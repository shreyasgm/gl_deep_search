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
                authorships = result.get("authorships", [])
                authors = ", ".join(
                    author.get("author", {}).get("display_name", "")
                    for author in authorships
                    if "author" in author and "display_name" in author["author"]
                )

                # Extract abstract
                abstract_dict = result.get("abstract_inverted_index", {})
                abstract = self._extract_abstract(abstract_dict)

                # Extract URL
                primary_location = result.get("primary_location", {})
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
        df = pd.DataFrame([pub.dict() for pub in publications])
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(publications)} publications to {output_path}")

    def load_from_csv(self, input_path: Path) -> list[OpenAlexPublication]:
        """Load publications from CSV file"""
        if not input_path.exists():
            logger.warning(f"CSV file {input_path} does not exist")
            return []

        try:
            df = pd.read_csv(input_path)
            publications = [OpenAlexPublication(**row) for _, row in df.iterrows()]
            logger.info(f"Loaded {len(publications)} publications from {input_path}")
            return publications
        except Exception as e:
            logger.error(f"Error loading publications from CSV: {e}")
            return []

    async def update_publications(
        self, existing_path: Path | None = None, output_path: Path | None = None
    ) -> list[OpenAlexPublication]:
        """Update publications by comparing existing ones with newly fetched ones"""
        # Default paths if not provided
        if not existing_path:
            existing_path = (
                Path(__file__).parent.parent
                / "data"
                / "intermediate"
                / "openalex_publications.csv"
            )
        if not output_path:
            output_path = existing_path

        # Load existing publications if available
        existing_publications = (
            self.load_from_csv(existing_path) if existing_path.exists() else []
        )
        existing_pub_map = {pub.paper_id: pub for pub in existing_publications}

        # Get new publications
        new_publications = await self.fetch_publications()

        # Merge existing and new publications,
        # preferring new ones with different content
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
            self.save_to_csv(updated_publications, output_path)

        return updated_publications
