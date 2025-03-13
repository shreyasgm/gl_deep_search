"""
Scraper module for the Growth Lab website publications
"""

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import yaml
import aiohttp
from bs4 import BeautifulSoup
from pydantic import BaseModel, HttpUrl, Field, validator
import pandas as pd
from tqdm.asyncio import tqdm as async_tqdm
import hashlib
import uuid

logger = logging.getLogger(__name__)


class Publication(BaseModel):
    """Represents a publication with relevant details"""

    paper_id: Optional[str] = None
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    abstract: Optional[str] = None
    pub_url: Optional[HttpUrl] = None
    file_urls: List[HttpUrl] = Field(default_factory=list)
    source: str = "GrowthLab"
    content_hash: Optional[str] = None  # Hash for detecting changes in publication

    @validator("year")
    def validate_year(cls, v):
        """Validate year is reasonable"""
        if v and (v < 1900 or v > 2100):
            return None
        return v

    def generate_id(self) -> str:
        """Generate a stable ID for the publication"""
        if self.paper_id:
            return self.paper_id

        # Create base string for hashing
        base = f"{self.title}_{self.authors}_{self.year}_{self.pub_url}"

        # Create hash for stability
        hash_obj = hashlib.md5(base.encode())
        hash_id = hash_obj.hexdigest()[:10]

        # Format: source_year_hash
        return f"gl_{self.year or '0000'}_{hash_id}"

    def generate_content_hash(self) -> str:
        """Generate a hash of the publication content to detect changes"""
        content = (
            f"{self.title}_{self.authors}_{self.year}_{self.abstract}_{self.pub_url}"
        )
        for url in self.file_urls:
            content += f"_{url}"
        return hashlib.sha256(content.encode()).hexdigest()


class GrowthLabScraper:
    """Scraper for Growth Lab website publications"""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the scraper with configuration"""
        self.config = self._load_config(config_path)
        self.base_url = self.config["base_url"]
        self.scrape_delay = self.config["scrape_delay"]

        # Hardcoded year corrections for publications with missing years
        self.year_corrections = {
            "https://growthlab.hks.harvard.edu/publications/sri-lanka-growth-diagnostic": 2018,
            "https://growthlab.hks.harvard.edu/publications/recommendations-trade-adjustment-assistance-sri-lanka": 2017,
            "https://growthlab.hks.harvard.edu/publications/immigration-policy-research": 2017,
            "https://growthlab.hks.harvard.edu/publications/sri-lanka%E2%80%99s-edible-oils-exports": 2016,
            "https://growthlab.hks.harvard.edu/publications/targeting-investment-japan-promising-leads-targeted-sectors-sri-lanka": 2016,
            "https://growthlab.hks.harvard.edu/publications/colombia-atlas-economic-complexity-datlas": 2014,
            "https://growthlab.hks.harvard.edu/publications/economic-complexity-brief": 2013,
            "https://growthlab.hks.harvard.edu/publications/journey-through-time-story-behind-%E2%80%98eight-decades-changes-occupational-tasks": 2024,
        }

    def _load_config(self, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load scraper configuration"""
        if not config_path:
            config_path = Path(__file__).parent.parent / "config.yaml"

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                return config["sources"]["growth_lab"]
        except Exception as e:
            logger.warning(f"Error loading scraper config: {e}. Using defaults.")
            return {
                "base_url": "https://growthlab.hks.harvard.edu/publications",
                "scrape_delay": 2.5,
            }

    @staticmethod
    async def get_max_page_num(session: aiohttp.ClientSession, url: str) -> int:
        """Get the maximum page number from pagination"""
        async with session.get(url) as response:
            if response.status != 200:
                logger.error(f"Failed to fetch {url}: {response.status}")
                return 0

            html = await response.text()
            soup = BeautifulSoup(html, "html.parser")
            pagination = soup.find("ul", {"class": "pager"})

            if pagination:
                last_page_link = pagination.find("li", {"class": "pager-last"})
                if last_page_link and last_page_link.find("a"):
                    last_page_url = last_page_link.find("a").get("href")
                    match = re.search(r"\d+", last_page_url)
                    if match:
                        return int(match.group())

            return 0

    async def parse_publication(
        self, pub_element: BeautifulSoup, base_url: str
    ) -> Optional[Publication]:
        """Parse a single publication element"""
        try:
            title_element = pub_element.find("span", {"class": "biblio-title"})
            if not title_element:
                return None

            title = title_element.text.strip()
            title_link = title_element.find("a")
            pub_url = title_link.get("href") if title_link else None

            # Ensure URL is absolute
            if pub_url and not pub_url.startswith(("http://", "https://")):
                pub_url = f"{base_url.split('/publications')[0]}{pub_url}"

            authors_element = pub_element.find("span", {"class": "biblio-authors"})
            authors = authors_element.text.strip() if authors_element else None

            # Extract year
            year = None
            if authors_element:
                sibling_text = authors_element.next_sibling
                if sibling_text:
                    year_match = re.search(r"\b\d{4}\b", sibling_text)
                    if year_match:
                        year = int(year_match.group())

            # Apply year correction if available
            if pub_url in self.year_corrections:
                year = self.year_corrections[pub_url]

            abstract_element = pub_element.find(
                "div", {"class": "biblio-abstract-display"}
            )
            abstract = abstract_element.text.strip() if abstract_element else None

            # Get file URLs
            file_urls = []
            if pub_element.find_all("span", {"class": "file"}):
                for file_elem in pub_element.find_all("span", {"class": "file"}):
                    file_link = file_elem.find("a")
                    if file_link and file_link.get("href"):
                        file_url = file_link["href"]
                        # Ensure URL is absolute
                        if not file_url.startswith(("http://", "https://")):
                            file_url = f"{base_url.split('/publications')[0]}{file_url}"
                        file_urls.append(file_url)

            pub = Publication(
                title=title,
                authors=authors,
                year=year,
                abstract=abstract,
                pub_url=pub_url,
                file_urls=file_urls,
                source="GrowthLab",
            )

            # Generate stable ID and content hash
            pub.paper_id = pub.generate_id()
            pub.content_hash = pub.generate_content_hash()

            return pub
        except Exception as e:
            logger.error(f"Error parsing publication: {e}")
            return None

    async def fetch_page(
        self, session: aiohttp.ClientSession, page_num: int
    ) -> List[Publication]:
        """Fetch a single page of publications"""
        url = self.base_url if page_num == 0 else f"{self.base_url}?page={page_num}"
        publications = []

        try:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch page {page_num}: {response.status}")
                    return []

                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                pub_elements = soup.find_all("div", {"class": "biblio-entry"})

                for pub_element in pub_elements:
                    pub = await self.parse_publication(pub_element, self.base_url)
                    if pub:
                        publications.append(pub)

                # Sleep to prevent overwhelming the server
                await asyncio.sleep(self.scrape_delay)
                return publications
        except Exception as e:
            logger.error(f"Error fetching page {page_num}: {e}")
            return []

    async def extract_publications(self) -> List[Publication]:
        """Extract all publications from the Growth Lab website"""
        async with aiohttp.ClientSession() as session:
            # Get the maximum page number
            max_page_num = await self.get_max_page_num(session, self.base_url)
            logger.info(f"Found {max_page_num} pages of publications")

            # Fetch all pages
            tasks = []
            for page_num in range(max_page_num + 1):
                tasks.append(self.fetch_page(session, page_num))

            # Process pages with progress bar
            all_publications = []
            for publications_list in await async_tqdm.gather(
                *tasks, desc="Scraping pages"
            ):
                all_publications.extend(publications_list)

            logger.info(f"Extracted {len(all_publications)} publications")
            return all_publications

    async def get_endnote_file_url(
        self, session: aiohttp.ClientSession, publication_url: str
    ) -> Optional[str]:
        """Fetch Endnote file URL from a publication page"""
        try:
            async with session.get(publication_url) as response:
                if response.status != 200:
                    return None

                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                endnote_link = soup.find("li", class_="biblio_tagged")

                if endnote_link and endnote_link.find("a"):
                    return endnote_link.find("a")["href"]
                return None
        except Exception as e:
            logger.error(f"Error fetching Endnote URL for {publication_url}: {e}")
            return None

    async def parse_endnote_content(self, content: str) -> Dict[str, Any]:
        """Parse Endnote file content"""
        record: Dict[str, Any] = {}
        lines = content.split("\n")

        for line in lines:
            if line.startswith("%"):
                key = line[1]
                value = line[3:].strip()

                if key == "A":  # Author
                    name_parts = value.split(", ")
                    if len(name_parts) == 2:
                        value = f"{name_parts[1]} {name_parts[0]}"
                    record["author"] = record.get("author", []) + [value]
                elif key == "T":  # Title
                    record["title"] = value
                elif key == "D":  # Date
                    record["date"] = value
                elif key == "X":  # Abstract
                    soup = BeautifulSoup(value, "html.parser")
                    for tag in soup.find_all(["b", "strong"]):
                        tag.unwrap()

                    abstract = "\n".join(
                        p.get_text(separator=" ", strip=True)
                        for p in soup.find_all("p")
                        if p.get_text(strip=True)
                    )
                    record["abstract"] = abstract.strip()

        if "author" in record:
            record["author"] = ", ".join(record["author"])

        return record

    async def enrich_publication(
        self, session: aiohttp.ClientSession, pub: Publication
    ) -> Publication:
        """Enrich publication with data from Endnote file"""
        if not pub.pub_url:
            return pub

        endnote_url = await self.get_endnote_file_url(session, pub.pub_url)
        if not endnote_url:
            return pub

        try:
            async with session.get(endnote_url) as response:
                if response.status != 200:
                    return pub

                content = await response.text()
                endnote_data = await self.parse_endnote_content(content)

                # Update publication with Endnote data if missing
                if not pub.title and "title" in endnote_data:
                    pub.title = endnote_data["title"]

                if not pub.authors and "author" in endnote_data:
                    pub.authors = endnote_data["author"]

                if not pub.abstract and "abstract" in endnote_data:
                    pub.abstract = endnote_data["abstract"]

                # Update content hash
                pub.content_hash = pub.generate_content_hash()

                return pub
        except Exception as e:
            logger.error(f"Error enriching publication from Endnote: {e}")
            return pub

    async def extract_and_enrich_publications(self) -> List[Publication]:
        """Extract all publications and enrich them with Endnote data"""
        publications = await self.extract_publications()

        async with aiohttp.ClientSession() as session:
            tasks = [self.enrich_publication(session, pub) for pub in publications]
            enriched_publications = await async_tqdm.gather(
                *tasks, desc="Enriching publications"
            )

        return enriched_publications

    def save_to_csv(self, publications: List[Publication], output_path: Path) -> None:
        """Save publications to CSV file"""
        df = pd.DataFrame([pub.dict() for pub in publications])
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(publications)} publications to {output_path}")

    def load_from_csv(self, input_path: Path) -> List[Publication]:
        """Load publications from CSV file"""
        if not input_path.exists():
            logger.warning(f"CSV file {input_path} does not exist")
            return []

        try:
            df = pd.read_csv(input_path)
            publications = [Publication(**row) for _, row in df.iterrows()]
            logger.info(f"Loaded {len(publications)} publications from {input_path}")
            return publications
        except Exception as e:
            logger.error(f"Error loading publications from CSV: {e}")
            return []

    async def update_publications(
        self, existing_path: Optional[Path] = None, output_path: Optional[Path] = None
    ) -> List[Publication]:
        """
        Update publications by comparing existing ones with newly scraped ones

        This handles updates to existing publications by comparing content hashes
        """
        # Default paths if not provided
        if not existing_path:
            existing_path = (
                Path(__file__).parent.parent
                / "data"
                / "intermediate"
                / "growth_lab_publications.csv"
            )
        if not output_path:
            output_path = existing_path

        # Load existing publications if available
        existing_publications = (
            self.load_from_csv(existing_path) if existing_path.exists() else []
        )
        existing_pub_map = {pub.paper_id: pub for pub in existing_publications}

        # Get new publications
        new_publications = await self.extract_and_enrich_publications()

        # Merge existing and new publications, preferring new ones with different content
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

        # Check for publications that no longer exist on the website
        for pub_id, pub in existing_pub_map.items():
            if pub_id not in new_pub_map:
                logger.info(
                    f"Publication {pub_id} no longer available, retaining in database"
                )
                updated_publications.append(pub)

        # Save updated publications
        if output_path:
            self.save_to_csv(updated_publications, output_path)

        return updated_publications
