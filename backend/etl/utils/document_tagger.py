"""
Document tagging module for Growth Lab Deep Search ETL pipeline.

Generates structured metadata tags for documents using an LLM and propagates
them to all chunks produced from each document.  Tags are stored both in the
PublicationTracking SQLite database and inside each chunk's ``metadata`` dict
as ``document_tags``, making them available as Qdrant payload filters.

Pipeline position: after text chunker, before embeddings generator.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI
from pydantic import BaseModel

from backend.etl.models.tracking import TaggingStatus
from backend.storage.base import StorageBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TaggingResult:
    """Result of the tagging operation for a single document."""

    publication_id: str
    status: TaggingStatus
    tags: dict[str, Any] | None
    processing_time: float
    chunks_tagged: int = 0
    text_source: str = "abstract"  # "abstract", "chunks", or "none"
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Pydantic output schema
# ---------------------------------------------------------------------------


class DocumentTags(BaseModel):
    """Structured output schema enforcing the four required tag fields."""

    regions: list[str]
    topics: list[str]
    document_type: str
    methodology: list[str]


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are a research document classifier for the Harvard Growth Lab publication library.

Given the following document information, classify it using the provided taxonomy.
{title_section}
{text_section}

## Taxonomy
Select ALL that apply from each category. You may add at most ONE new entry per \
category if nothing in the provided list fits.

regions (countries or world regions the document focuses on):
{regions}

topics (research themes covered):
{topics}

document_type (pick EXACTLY ONE best match):
{document_types}

methodology (research approaches used):
{methodologies}

## Instructions
- Choose all relevant items from each category.
- For document_type, select exactly one value (a string, not a list).
- Use ONLY values from the provided taxonomy lists above (you may add at most one \
new entry per category if nothing provided fits).
- Return values as arrays for regions, topics, and methodology; a single string \
for document_type."""


# ---------------------------------------------------------------------------
# DocumentTagger
# ---------------------------------------------------------------------------


class DocumentTagger:
    """Tags documents with structured metadata using an LLM.

    Reads publication title/abstract from the tracker (falls back to chunk
    sampling when the abstract is absent or too short), calls an LLM to
    classify the document against a configured taxonomy, and writes the
    resulting tag dict into every chunk's ``metadata["document_tags"]`` field.

    Tags are also persisted to the PublicationTracking SQLite table via the
    optional ``tracker`` dependency.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        tracker=None,
    ) -> None:
        self.tracker = tracker
        self._full_config = self._load_config(config_path)
        self.config: dict = (
            self._full_config.get("file_processing", {}).get("tagging", {})
        )

        self.abstract_min_length: int = self.config.get("abstract_min_length", 100)
        self.num_fallback_chunks: int = self.config.get("num_fallback_chunks", 3)

        llm_cfg = self.config.get("llm", {})
        self.llm_model: str = llm_cfg.get("model", "gpt-4o-mini")
        self.llm_temperature: float = llm_cfg.get("temperature", 0.1)
        self.llm_max_tokens: int = llm_cfg.get("max_tokens", 500)

        provider = llm_cfg.get("provider", "openai")
        llm_base_url: str | None = (
            "https://openrouter.ai/api/v1" if provider == "openrouter" else None
        )
        api_key: str | None = llm_cfg.get("api_key", os.environ.get("OPENAI_API_KEY"))
        self._client = AsyncOpenAI(api_key=api_key, base_url=llm_base_url)

        taxonomy = self.config.get("taxonomy", {})
        self.taxonomy_regions: list[str] = taxonomy.get("regions", [])
        self.taxonomy_topics: list[str] = taxonomy.get("topics", [])
        self.taxonomy_doc_types: list[str] = taxonomy.get("document_type", [])
        self.taxonomy_methodologies: list[str] = taxonomy.get("methodology", [])

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_config(self, config_path: str | Path | None) -> dict:
        """Load full config from YAML."""
        if not config_path:
            config_path = Path(__file__).parent.parent / "config.yaml"
        config_path = Path(config_path)
        try:
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Error loading tagger config from {config_path}: {e}")
            return {}

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, title: str | None, text: str) -> str:
        """Build the LLM classification prompt."""
        title_section = f"Title: {title}" if title else ""
        text_section = f"Abstract / Text:\n{text}"
        return _PROMPT_TEMPLATE.format(
            title_section=title_section,
            text_section=text_section,
            regions="\n".join(f"  - {r}" for r in self.taxonomy_regions),
            topics="\n".join(f"  - {t}" for t in self.taxonomy_topics),
            document_types="\n".join(f"  - {d}" for d in self.taxonomy_doc_types),
            methodologies="\n".join(f"  - {m}" for m in self.taxonomy_methodologies),
        )

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    async def _call_llm(self, prompt: str) -> dict[str, Any] | None:
        """Call the LLM using Pydantic structured outputs, or None after 3 failures."""
        for attempt in range(3):
            try:
                response = await self._client.beta.chat.completions.parse(
                    model=self.llm_model,
                    temperature=self.llm_temperature,
                    max_tokens=self.llm_max_tokens,
                    response_format=DocumentTags,
                    messages=[{"role": "user", "content": prompt}],
                )
                parsed: DocumentTags | None = response.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("LLM refused to generate structured output")
                return parsed.model_dump()

            except Exception as e:
                logger.warning(f"LLM call error (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    await asyncio.sleep(2**attempt)

        return None

    # ------------------------------------------------------------------
    # Text input resolution
    # ------------------------------------------------------------------

    def _get_text_input(
        self, pub_id: str, chunks_path: Path
    ) -> tuple[str | None, str, str]:
        """Resolve the text that will be sent to the LLM.

        Priority:
        1. Abstract from the tracker (if length >= abstract_min_length).
        2. Sample of representative chunks from the chunks.json file.

        Returns:
            (title, text, source) where source is "abstract", "chunks", or "none".
        """
        title: str | None = None
        abstract: str | None = None

        if self.tracker:
            try:
                pub = self.tracker.get_publication(pub_id)
                if pub:
                    title = pub.title
                    abstract = pub.abstract
            except Exception as e:
                logger.debug(
                    f"Could not fetch tracker record for {pub_id}: {e}"
                )

        if abstract and len(abstract) >= self.abstract_min_length:
            return title, abstract, "abstract"

        # Fallback: sample first, middle, and last chunks
        try:
            with open(chunks_path, encoding="utf-8") as f:
                chunks = json.load(f)
            if not chunks:
                return title, abstract or "", "none"
            n = len(chunks)
            indices = sorted({0, n // 2, n - 1})[: self.num_fallback_chunks]
            sampled_text = "\n\n---\n\n".join(
                chunks[i]["text_content"] for i in indices if i < n
            )
            return title, sampled_text, "chunks"
        except Exception as e:
            logger.warning(f"Could not sample chunks for {pub_id}: {e}")

        return title, abstract or "", "none"

    # ------------------------------------------------------------------
    # Chunk injection
    # ------------------------------------------------------------------

    def _inject_tags_into_chunks(
        self, chunks_path: Path, tags: dict[str, Any]
    ) -> int:
        """Load chunks.json, inject ``document_tags`` into every chunk's metadata.

        Rewrites the file in-place.

        Returns:
            Number of chunks updated.
        """
        with open(chunks_path, encoding="utf-8") as f:
            chunks = json.load(f)

        for chunk in chunks:
            chunk.setdefault("metadata", {})["document_tags"] = tags

        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        return len(chunks)

    # ------------------------------------------------------------------
    # Core tagging
    # ------------------------------------------------------------------

    async def tag_document(
        self, pub_id: str, chunks_path: Path
    ) -> TaggingResult:
        """Tag a single document and inject tags into its chunks.json.

        Args:
            pub_id: Publication identifier (used for tracker updates).
            chunks_path: Absolute path to the document's chunks.json file.

        Returns:
            TaggingResult with status, tags, and processing info.
        """
        start = time.monotonic()

        if self.tracker:
            try:
                self.tracker.update_tagging_status(pub_id, TaggingStatus.IN_PROGRESS)
            except Exception:
                pass

        title, text, text_source = self._get_text_input(pub_id, chunks_path)

        if not text:
            error = (
                "No text available for tagging (no abstract and no chunk samples)"
            )
            logger.warning(f"{pub_id}: {error}")
            if self.tracker:
                try:
                    self.tracker.update_tagging_status(
                        pub_id, TaggingStatus.FAILED, error=error
                    )
                except Exception:
                    pass
            return TaggingResult(
                publication_id=pub_id,
                status=TaggingStatus.FAILED,
                tags=None,
                processing_time=time.monotonic() - start,
                text_source=text_source,
                error_message=error,
            )

        prompt = self._build_prompt(title, text)
        tags = await self._call_llm(prompt)

        if tags is None:
            error = "LLM failed to return valid tags after 3 attempts"
            logger.error(f"{pub_id}: {error}")
            if self.tracker:
                try:
                    self.tracker.update_tagging_status(
                        pub_id, TaggingStatus.FAILED, error=error
                    )
                except Exception:
                    pass
            return TaggingResult(
                publication_id=pub_id,
                status=TaggingStatus.FAILED,
                tags=None,
                processing_time=time.monotonic() - start,
                text_source=text_source,
                error_message=error,
            )

        # Inject tags into every chunk
        try:
            chunks_tagged = self._inject_tags_into_chunks(chunks_path, tags)
        except Exception as e:
            error = f"Failed to inject tags into chunks: {e}"
            logger.error(f"{pub_id}: {error}")
            if self.tracker:
                try:
                    self.tracker.update_tagging_status(
                        pub_id, TaggingStatus.FAILED, error=error
                    )
                except Exception:
                    pass
            return TaggingResult(
                publication_id=pub_id,
                status=TaggingStatus.FAILED,
                tags=tags,
                processing_time=time.monotonic() - start,
                text_source=text_source,
                error_message=error,
            )

        # Persist tags and update tracker status
        if self.tracker:
            try:
                self.tracker.update_tags(pub_id, tags)
                self.tracker.update_tagging_status(pub_id, TaggingStatus.TAGGED)
            except Exception as e:
                logger.warning(f"Could not update tracker for {pub_id}: {e}")

        logger.info(
            f"Tagged {pub_id}: {chunks_tagged} chunks updated "
            f"(source: {text_source}, topics: {tags.get('topics', [])})"
        )
        return TaggingResult(
            publication_id=pub_id,
            status=TaggingStatus.TAGGED,
            tags=tags,
            processing_time=time.monotonic() - start,
            chunks_tagged=chunks_tagged,
            text_source=text_source,
        )

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    async def process_all_documents(
        self,
        storage: StorageBase,
        max_docs: int | None = None,
        overwrite: bool = False,
    ) -> list[TaggingResult]:
        """Tag all untagged documents found in the processed chunks directory.

        A document is considered already tagged when its first chunk has a
        ``metadata["document_tags"]`` key.  This makes the operation fully
        idempotent — re-running will skip already-tagged documents unless
        ``overwrite=True`` is passed.

        Args:
            storage: Storage backend used to locate chunks.json files.
            max_docs: Maximum documents to tag (None = all).  When None, the
                      value is read from ``file_processing.tagging.max_docs``
                      in the config, then falls back to processing everything.
            overwrite: When True, re-tag documents that have already been tagged,
                       replacing their existing ``document_tags`` in chunks.json.

        Returns:
            List of TaggingResult, one per document processed.
        """
        # Resolve limit
        effective_max = max_docs
        if effective_max is None:
            cfg_max = self.config.get("max_docs")
            if isinstance(cfg_max, int) and cfg_max > 0:
                effective_max = cfg_max

        # Discover all chunks.json files
        chunk_relatives = storage.glob("processed/chunks/**/chunks.json")
        if not chunk_relatives:
            logger.warning("No chunks.json files found — run text chunker first")
            return []

        logger.info(f"Found {len(chunk_relatives)} chunk file(s)")

        # Build list of (pub_id, local_path), skipping already-tagged docs
        to_process: list[tuple[str, Path]] = []
        already_tagged = 0
        for rel in chunk_relatives:
            local_path = storage.get_path(rel)
            # Path pattern: processed/chunks/documents/{source}/{pub_id}/chunks.json
            pub_id = Path(rel).parent.name

            try:
                with open(local_path, encoding="utf-8") as f:
                    first_chunk = json.load(f)[0]
                if not overwrite and "document_tags" in first_chunk.get("metadata", {}):
                    already_tagged += 1
                    logger.debug(f"Skipping already-tagged document: {pub_id}")
                    continue
            except Exception as e:
                logger.warning(
                    f"Could not read {local_path} for idempotency check: {e}"
                )

            to_process.append((pub_id, local_path))

        logger.info(
            f"{len(to_process)} document(s) need tagging "
            f"({already_tagged} already tagged, skipped)"
        )

        # Apply max_docs cap
        if effective_max is not None and len(to_process) > effective_max:
            logger.info(
                f"Limiting to {effective_max} of {len(to_process)} "
                f"documents (max_docs={effective_max})"
            )
            to_process = to_process[:effective_max]

        results: list[TaggingResult] = []
        for pub_id, chunks_path in to_process:
            result = await self.tag_document(pub_id, chunks_path)
            results.append(result)

        return results
