"""
OpenAlex API client for fetching academic publications
"""

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd
import yaml
from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator
from tqdm.asyncio import tqdm as async_tqdm

logger = logging.getLogger(__name__)


class OpenAlexPublication(BaseModel):
    """Represents a publication from the OpenAlex API"""

    paper_id: str  # OpenAlex ID (e.g., "W2741809807")
    openalex_id: str  # Full OpenAlex URL
    title: str | None = None
    authors: str | None = None
    year: int | None = None
    abstract: str | None = None
    pub_url: HttpUrl | None = None
    file_urls: list[HttpUrl] = Field(default_factory=list)
    source: str = "OpenAlex"
    content_hash: str | None = None  # Hash for detecting changes
    cited_by_count: int | None = None

    @field_validator("paper_id")
    def validate_id(cls, v):  # noqa: N805
        """Ensure paper_id is just the ID without the URL prefix"""
        if v.startswith("https://openalex.org/"):
            return v.replace("https://openalex.org/", "")
        return v

    @model_validator(mode="before")
    def set_openalex_id(cls, values: dict[str, Any]) -> dict[str, Any]:  # noqa: N805
        """Ensure openalex_id is the full URL"""
        if "paper_id" in values and not values.get("openalex_id"):
            paper_id = values["paper_id"]
            if not paper_id.startswith("https://openalex.org/"):
                values["openalex_id"] = f"https://openalex.org/{paper_id}"
        return values

    def generate_id(self) -> str:
        """Generate a stable ID for the publication

        This method is included for compatibility with the Growth Lab scraper,
        but we typically just use the OpenAlex ID directly since it's already stable.
        """
        # First choice: use the existing OpenAlex ID (already stable)
        if self.paper_id and self.paper_id.startswith("W"):
            return f"oa_{self.paper_id}"

        # Second choice: use DOI if available
        if self.file_urls:
            for url in self.file_urls:
                url_str = str(url).lower()
                if "doi.org" in url_str:
                    # Extract DOI and use it as identifier
                    doi = url_str.split("doi.org/")[-1]
                    # Remove any query parameters or fragments
                    doi = doi.split("?")[0].split("#")[0]
                    # Remove trailing slash if present
                    doi = doi.rstrip("/")
                    if doi:  # If we got a valid DOI
                        return f"oa_doi_{hashlib.sha256(doi.encode()).hexdigest()[:16]}"

        # Third choice: use the URL
        if self.pub_url:
            url_path = str(self.pub_url).lower()
            return f"oa_url_{hashlib.sha256(url_path.encode()).hexdigest()[:16]}"

        # Final fallback: use normalized metadata fields
        components = []

        # Normalize title - lowercase, remove punctuation and extra spaces
        if self.title:
            normalized_title = self._normalize_text(self.title)
            if normalized_title:
                components.append(f"t:{normalized_title}")

        # Normalize authors - lowercase, remove punctuation and extra spaces
        if self.authors:
            normalized_authors = self._normalize_text(self.authors)
            if normalized_authors:
                components.append(f"a:{normalized_authors}")

        # Add year if available
        if self.year:
            components.append(f"y:{self.year}")

        # Create a stable, normalized base for hashing
        base = "_".join(components)

        # If we don't have enough information to create a reliable hash
        if not base:
            # Use timestamp-based random ID as last resort
            import random
            import time

            random_id = f"{int(time.time())}_{random.randint(1000, 9999)}"
            return f"oa_unknown_{hashlib.sha256(random_id.encode()).hexdigest()[:16]}"

        # Create hash using SHA-256 for better collision resistance
        hash_id = hashlib.sha256(base.encode()).hexdigest()[:16]

        # Format: source_year_hash
        return f"oa_{self.year or '0000'}_{hash_id}"

    def _normalize_text(self, text: str) -> str:
        """Normalize text for stable ID generation

        Removes punctuation, extra spaces, and converts to lowercase.
        """
        if not text:
            return ""

        import re

        # Convert to lowercase
        text = text.lower()
        # Replace punctuation and special chars with spaces
        text = re.sub(r"[^\w\s]", " ", text)
        # Replace multiple spaces with a single space
        text = re.sub(r"\s+", " ", text)
        # Remove leading/trailing whitespace
        return text.strip()

    def generate_content_hash(self) -> str:
        """Generate a hash of the publication content to detect changes"""
        content = (
            f"{self.title}_{self.authors}_{self.year}_{self.abstract}_{self.pub_url}"
        )
        for url in self.file_urls:
            content += f"_{url}"
        return hashlib.sha256(content.encode()).hexdigest()


class OpenAlexClient:
    """Client for the OpenAlex API"""

    def __init__(self, config_path: Path | None = None):
        """Initialize the OpenAlex client with configuration"""
        self.config = self._load_config(config_path)
        self.author_id = self.config["author_id"]
        self.email = self.config["email"]
        self.max_retries_per_page = self.config["max_retries_per_page"]
        self.max_overall_retries = self.config["max_overall_retries"]

    def _load_config(self, config_path: Path | None = None) -> dict[str, Any]:
        """Load OpenAlex configuration"""
        if not config_path:
            config_path = Path(__file__).parent.parent / "config.yaml"

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                return config["sources"]["openalex"]
        except Exception as e:
            logger.warning(f"Error loading OpenAlex config: {e}. Using defaults.")
            return {
                "author_id": "A5034550995",
                "email": "example@example.com",
                "max_retries_per_page": 3,
                "max_overall_retries": 10,
            }

    def _build_url(self, cursor: str | None = None) -> str:
        """Build OpenAlex API URL with proper parameters"""
        base_url = (
            f"https://api.openalex.org/works?"
            f"filter=authorships.author.id:A{self.author_id.lstrip('A')},"
            f"primary_location.version:!submittedVersion"
            f"&mailto={self.email}"
        )

        if cursor:
            return f"{base_url}&cursor={cursor}&per-page=200"
        return f"{base_url}&per-page=200"

    async def fetch_page(
        self, session: aiohttp.ClientSession, cursor: str | None = None
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Fetch a single page of results from OpenAlex API"""
        url = self._build_url(cursor)
        retries = 0

        while retries < self.max_retries_per_page:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get("results", [])
                        next_cursor = data.get("meta", {}).get("next_cursor")
                        return results, next_cursor

                    logger.warning(f"Got status {response.status} for {url}")

                    # If rate limited, wait longer
                    if response.status == 429:
                        wait_time = 60  # 1 minute
                        logger.warning(f"Rate limited. Waiting {wait_time} seconds.")
                        await asyncio.sleep(wait_time)
                    else:
                        await asyncio.sleep(5)  # General retry delay
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                await asyncio.sleep(5)

            retries += 1

        logger.error(f"Failed to fetch {url} after {retries} retries")
        return [], None

    async def fetch_all_pages(self) -> list[dict[str, Any]]:
        """Fetch all pages of results from OpenAlex API"""
        all_results = []
        cursor: str | None = "*"  # Start with first page
        overall_retries = 0

        async with aiohttp.ClientSession() as session:
            with async_tqdm(desc="Fetching OpenAlex pages") as pbar:
                while cursor and overall_retries < self.max_overall_retries:
                    results, next_cursor = await self.fetch_page(session, cursor)

                    if results:
                        all_results.extend(results)
                        pbar.update(1)
                        pbar.set_postfix({"total": len(all_results)})
                        cursor = next_cursor
                    else:
                        overall_retries += 1
                        if not next_cursor:
                            # No more pages
                            break

        logger.info(f"Fetched {len(all_results)} results from OpenAlex")
        return all_results

    def _extract_abstract(self, abstract_dict: dict[str, list[int]]) -> str:
        """Extract abstract text from OpenAlex abstract_inverted_index"""
        if not abstract_dict:
            return ""

        # Find the maximum position
        max_pos = 0
        for positions in abstract_dict.values():
            if positions and max(positions) > max_pos:
                max_pos = max(positions)

        # Create a list with None placeholders
        abstract_list: list[str | None] = [None] * (max_pos + 1)

        # Fill in the words at their positions
        for word, positions in abstract_dict.items():
            for pos in positions:
                abstract_list[pos] = word

        # Join the words, skipping None values
        abstract = " ".join(word if word is not None else "" for word in abstract_list)
        return abstract

    def process_results(
        self, results: list[dict[str, Any]]
    ) -> list[OpenAlexPublication]:
        """Process raw API results into OpenAlexPublication objects"""
        publications = []

        for result in results:
            try:
                # Extract the OpenAlex ID
                openalex_id = result.get("id", "")
                paper_id = openalex_id.replace("https://openalex.org/", "")

                # Extract publication year
                year = result.get("publication_year")

                # Extract title
                title = result.get("title", "")

                # Extract authors
                authorships = result.get("authorships") or []
                # Build author string
                author_names = []
                for author in authorships:
                    author_obj = author.get("author", {})
                    if author_obj and "display_name" in author_obj:
                        author_names.append(author_obj.get("display_name", ""))
                authors = ", ".join(author_names)

                # Extract abstract
                abstract_dict = result.get("abstract_inverted_index", {})
                abstract = self._extract_abstract(abstract_dict)

                # Extract URL
                primary_location = result.get("primary_location") or {}
                pub_url = primary_location.get("landing_page_url", None)

                # Extract DOI as file URL
                doi = result.get("doi")
                file_urls = [doi] if doi else []

                # Create publication object
                pub = OpenAlexPublication(
                    paper_id=paper_id,
                    openalex_id=openalex_id,
                    title=title,
                    authors=authors,
                    year=year,
                    abstract=abstract,
                    pub_url=pub_url,
                    file_urls=file_urls,
                    source="OpenAlex",
                    cited_by_count=result.get("cited_by_count"),
                )

                # Use our improved ID generation if needed
                # Note: OpenAlex already has stable IDs, but this is here for
                # consistency with the rest of the system
                # If we don't have a standard OpenAlex ID
                if not pub.paper_id.startswith("W"):
                    pub.paper_id = pub.generate_id()

                # Generate content hash
                pub.content_hash = pub.generate_content_hash()

                publications.append(pub)
            except Exception as e:
                logger.error(f"Error processing result: {e}")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Problematic result: {json.dumps(result, indent=2)}")

        return publications

    async def fetch_publications(self) -> list[OpenAlexPublication]:
        """Fetch all publications from OpenAlex API"""
        results = await self.fetch_all_pages()
        publications = self.process_results(results)
        return publications

    def save_to_csv(
        self, publications: list[OpenAlexPublication], output_path: Path
    ) -> None:
        """Save publications to CSV file"""
        # Convert publications to dictionaries and handle HttpUrl objects
        pub_dicts = []
        for pub in publications:
            pub_dict = pub.model_dump()
            # Convert HttpUrl objects to strings
            if pub_dict.get("pub_url"):
                pub_dict["pub_url"] = str(pub_dict["pub_url"])
            if pub_dict.get("file_urls"):
                pub_dict["file_urls"] = [str(url) for url in pub_dict["file_urls"]]
            pub_dicts.append(pub_dict)

        df = pd.DataFrame(pub_dicts)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(publications)} publications to {output_path}")

    def load_from_csv(self, input_path: Path) -> list[OpenAlexPublication]:
        """Load publications from CSV file"""
        if not input_path.exists():
            logger.warning(f"CSV file {input_path} does not exist")
            return []

        try:
            df = pd.read_csv(input_path)

            # Convert string representation of lists to actual lists
            # Process list fields
            for list_field in ["file_urls", "concepts"]:
                if list_field in df.columns:
                    df[list_field] = df[list_field].apply(
                        lambda x: eval(x)
                        if isinstance(x, str) and x.startswith("[")
                        else []
                    )

            # Convert NaN values to None (or appropriate default values)
            df = df.replace({pd.NA: None})
            for col in df.columns:
                df[col] = df[col].apply(lambda x: None if pd.isna(x) else x)

            publications = []
            for _, row in df.iterrows():
                try:
                    # Create a clean dictionary without NaN values
                    clean_dict: dict[str, Any] = {}
                    for k, v in row.to_dict().items():
                        if pd.isna(v):
                            clean_dict[k] = None
                        else:
                            clean_dict[k] = v

                    pub = OpenAlexPublication(**clean_dict)
                    publications.append(pub)
                except Exception as e:
                    logger.warning(f"Could not load publication: {e}")

            logger.info(f"Loaded {len(publications)} publications from {input_path}")
            return publications
        except Exception as e:
            logger.error(f"Error loading publications from CSV: {e}")
            return []

    async def update_publications(
        self,
        existing_path: Path | None = None,
        output_path: Path | None = None,
        storage=None,
    ) -> list[OpenAlexPublication]:
        """
        Update publications by comparing existing ones with newly fetched ones

        This handles updates to existing publications by comparing content hashes

        Args:
            existing_path: Optional path to existing publications CSV
            output_path: Optional path to save updated publications
            storage: Optional storage instance (will use default if None)
        """
        # Import storage factory here to avoid circular imports
        from backend.storage.factory import get_storage

        # Get storage instance if not provided
        storage = storage or get_storage()

        # Default paths if not provided
        if not existing_path:
            existing_path = storage.get_path("intermediate/openalex_publications.csv")
        if not output_path:
            output_path = existing_path

        # Load existing publications if available
        existing_publications = (
            self.load_from_csv(existing_path)
            if existing_path and existing_path.exists()
            else []
        )
        existing_pub_map = {pub.paper_id: pub for pub in existing_publications}

        # Get new publications
        new_publications = await self.fetch_publications()

        # Merge publications, preferring new ones with different content
        updated_publications = []
        new_pub_map = {}

        for pub in new_publications:
            pub_id = pub.paper_id
            new_pub_map[pub_id] = pub

            if pub_id in existing_pub_map:
                # If content changed, use new publication
                if pub.content_hash != existing_pub_map[pub_id].content_hash:
                    logger.info(f"Publication {pub_id} updated")
                    updated_publications.append(pub)
                else:
                    # No changes, use existing
                    updated_publications.append(existing_pub_map[pub_id])
            else:
                # New publication
                logger.info(f"New publication found: {pub_id}")
                updated_publications.append(pub)

        # Check for publications that no longer exist in the API
        for pub_id, pub in existing_pub_map.items():
            if pub_id not in new_pub_map:
                logger.info(
                    f"Publication {pub_id} no longer available in OpenAlex, "
                    "retaining in database"
                )
                updated_publications.append(pub)

        # Save updated publications
        if output_path:
            # Ensure parent directory exists
            storage.ensure_dir(output_path.parent)
            self.save_to_csv(updated_publications, output_path)

        return updated_publications
