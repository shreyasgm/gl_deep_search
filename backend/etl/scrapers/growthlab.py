"""
Scraper module for the Growth Lab website publications
"""

import asyncio
import hashlib
import logging
import random
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import aiohttp
import pandas as pd
import yaml
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, HttpUrl, field_validator
from tqdm.asyncio import tqdm as async_tqdm

logger = logging.getLogger(__name__)

# Type variable for generic retry function
T = TypeVar("T")


async def retry_with_backoff(
    func: Callable[..., T],
    *args,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retry_on: tuple = (aiohttp.ClientError, TimeoutError),
    **kwargs,
) -> T:
    """
    Retry an async function with exponential backoff.

    Args:
        func: The async function to retry
        *args: Arguments to pass to the function
        max_retries: Maximum number of retries
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        retry_on: Tuple of exceptions to retry on
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The return value of the function

    Raises:
        The last exception encountered if max_retries is exceeded
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result
        except retry_on as e:
            if attempt == max_retries:
                # On last attempt, re-raise the exception
                logger.error(
                    f"Max retries ({max_retries}) exceeded for {func.__name__}: {e}"
                )
                raise

            # Calculate delay with jitter to avoid thundering herd
            delay = min(base_delay * (2**attempt) + random.uniform(0, 1), max_delay)

            # Log retry attempt
            logger.warning(
                f"Request failed with {e.__class__.__name__}: {e}. "
                f"Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})"
            )

            last_exception = e
            await asyncio.sleep(delay)

    # This should not be reached due to the re-raise in the loop,
    # but just in case, we raise the last exception here
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected failure in retry logic")


class Publication(BaseModel):
    """Represents a publication with relevant details"""

    paper_id: str | None = None
    title: str | None = None
    authors: str | None = None
    year: int | None = None
    abstract: str | None = None
    pub_url: HttpUrl | None = None
    file_urls: list[HttpUrl] = Field(default_factory=list)
    source: str = "GrowthLab"
    content_hash: str | None = None  # Hash for detecting changes in publication

    @field_validator("year")
    def validate_year(cls, v):  # noqa: N805
        """Validate year is reasonable"""
        if v and (v < 1900 or v > 2100):
            raise ValueError(f"Year {v} is not in valid range (1900-2100)")
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

    def __init__(
        self, config_path: Path | None = None, concurrency_limit: int | None = None
    ):
        """
        Initialize the scraper with configuration

        Args:
            config_path: Path to the configuration file
            concurrency_limit: Maximum number of concurrent requests (default from config)
        """
        self.config = self._load_config(config_path)
        self.base_url = self.config["base_url"]
        self.scrape_delay = self.config["scrape_delay"]

        # Set concurrency limit (from parameter or config)
        self.concurrency_limit = concurrency_limit or self.config.get(
            "concurrency_limit", 5
        )
        # Create a semaphore for concurrency control
        self.semaphore = asyncio.Semaphore(self.concurrency_limit)

        # fmt: off
        # ruff: noqa: E501
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
        # fmt: on

    def _load_config(self, config_path: Path | None = None) -> dict[str, Any]:
        """Load scraper configuration"""
        if not config_path:
            config_path = Path(__file__).parent.parent / "config.yaml"

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                return config["sources"]["growth_lab"]
        except Exception as e:
            logger.warning(f"Error loading scraper config: {e}. Using defaults.")
            return {
                "base_url": "https://growthlab.hks.harvard.edu/publications",
                "scrape_delay": 2.5,
                "concurrency_limit": 5,
                "max_retries": 3,
                "retry_base_delay": 1.0,
                "retry_max_delay": 30.0,
            }

    async def _get_max_page_num_impl(
        self, session: aiohttp.ClientSession, url: str
    ) -> int:
        """Implementation to get the maximum page number from pagination"""
        async with self.semaphore:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch {url}: {response.status}")
                    # Raise exception for non-200 to allow retry mechanism to work
                    if response.status == 429 or response.status >= 500:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"Rate limited or server error: {response.status}",
                        )
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

    async def get_max_page_num(self, session: aiohttp.ClientSession, url: str) -> int:
        """Get the maximum page number from pagination with retry mechanism"""
        max_retries = self.config.get("max_retries", 3)
        base_delay = self.config.get("retry_base_delay", 1.0)
        max_delay = self.config.get("retry_max_delay", 30.0)

        return await retry_with_backoff(
            self._get_max_page_num_impl,
            session,
            url,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            retry_on=(aiohttp.ClientError, TimeoutError),
        )

    async def parse_publication(
        self, pub_element: BeautifulSoup, base_url: str
    ) -> Publication | None:
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

    async def _fetch_page_impl(
        self, session: aiohttp.ClientSession, page_num: int
    ) -> list[Publication]:
        """Implementation to fetch a single page of publications"""
        url = self.base_url if page_num == 0 else f"{self.base_url}?page={page_num}"
        publications = []

        # Use the semaphore to limit concurrency
        async with self.semaphore:
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(
                            f"Failed to fetch page {page_num}: {response.status}"
                        )
                        # Raise exception for non-200 to allow retry mechanism to work
                        if response.status == 429 or response.status >= 500:
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status,
                                message=f"Rate limited or server error: {response.status}",
                            )
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
                # Log and re-raise to allow the retry mechanism to work
                logger.error(f"Error fetching page {page_num}: {e}")
                raise

    async def fetch_page(
        self, session: aiohttp.ClientSession, page_num: int
    ) -> list[Publication]:
        """Fetch a single page of publications with retry mechanism"""
        max_retries = self.config.get("max_retries", 3)
        base_delay = self.config.get("retry_base_delay", 1.0)
        max_delay = self.config.get("retry_max_delay", 30.0)

        try:
            return await retry_with_backoff(
                self._fetch_page_impl,
                session,
                page_num,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                retry_on=(aiohttp.ClientError, TimeoutError, Exception),
            )
        except Exception as e:
            logger.error(f"All retries failed for page {page_num}: {e}")
            return []

    async def extract_publications(self) -> list[Publication]:
        """Extract all publications from the Growth Lab website"""
        # Create more robust session with timeouts
        timeout = aiohttp.ClientTimeout(
            total=60, connect=20, sock_connect=20, sock_read=20
        )
        connector = aiohttp.TCPConnector(
            limit=self.concurrency_limit,
            ttl_dns_cache=300,
            force_close=False,
            enable_cleanup_closed=True,
        )

        # Create a session with custom headers by default
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=headers,
            cookie_jar=aiohttp.CookieJar(unsafe=True),
        ) as session:
            # Get the maximum page number
            max_page_num = await self.get_max_page_num(session, self.base_url)
            logger.info(f"Found {max_page_num} pages of publications")

            # Create a list of pages to process
            all_pages = list(range(max_page_num + 1))

            # Process pages using semaphore-controlled concurrency
            all_publications = []
            total_file_urls = 0
            failed_pages = 0

            # Create tasks for each page but with concurrency control via semaphore
            tasks = [self.fetch_page(session, page_num) for page_num in all_pages]

            # Process tasks with progress bar using as_completed for better error handling
            with async_tqdm(total=len(tasks), desc="Scraping pages") as pbar:
                # Use a helper to process tasks and update progress
                for publications_future in asyncio.as_completed(tasks):
                    try:
                        publications = await publications_future
                        all_publications.extend(publications)
                        # Count file URLs
                        for pub in publications:
                            total_file_urls += len(pub.file_urls)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error processing page: {e}")
                        failed_pages += 1
                        pbar.update(1)

            if failed_pages > 0:
                logger.warning(f"Failed to process {failed_pages} pages due to errors")

            logger.info(
                f"Extracted {len(all_publications)} publications with {total_file_urls} total file URLs"
            )
            return all_publications

    async def _get_endnote_file_url_impl(
        self, session: aiohttp.ClientSession, publication_url: str
    ) -> str | None:
        """Implementation to fetch Endnote file URL from a publication page"""
        async with self.semaphore:
            try:
                # Add custom headers to mimic a browser
                headers = {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "Cache-Control": "max-age=0",
                }

                async with session.get(publication_url, headers=headers) as response:
                    if response.status != 200:
                        # Raise exception for retry on rate limits or server errors
                        if response.status == 429 or response.status >= 500:
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status,
                                message=f"Rate limited or server error: {response.status}",
                            )
                        return None

                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")
                    endnote_link = soup.find("li", class_="biblio_tagged")

                    if endnote_link and endnote_link.find("a"):
                        return endnote_link.find("a")["href"]
                    return None
            except (aiohttp.ClientError, TimeoutError) as e:
                # Log and re-raise to allow the retry mechanism to work
                logger.error(f"Error fetching Endnote URL for {publication_url}: {e}")
                raise
            except Exception as e:
                logger.error(f"Error fetching Endnote URL for {publication_url}: {e}")
                return None

    async def get_endnote_file_url(
        self, session: aiohttp.ClientSession, publication_url: str
    ) -> str | None:
        """Fetch Endnote file URL from a publication page with retry mechanism"""
        max_retries = self.config.get("max_retries", 3)
        base_delay = self.config.get("retry_base_delay", 1.0)
        max_delay = self.config.get("retry_max_delay", 30.0)

        try:
            return await retry_with_backoff(
                self._get_endnote_file_url_impl,
                session,
                publication_url,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                retry_on=(aiohttp.ClientError, TimeoutError),
            )
        except Exception as e:
            logger.error(
                f"All retries failed for getting endnote URL {publication_url}: {e}"
            )
            return None

    async def parse_endnote_content(self, content: str) -> dict[str, Any]:
        """Parse Endnote file content"""
        record: dict[str, Any] = {}
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

    async def _enrich_publication_impl(
        self, session: aiohttp.ClientSession, pub: Publication, endnote_url: str
    ) -> Publication:
        """Implementation to enrich publication with data from Endnote file"""
        async with self.semaphore:
            try:
                # Add custom headers to mimic a browser
                headers = {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Connection": "keep-alive",
                    "Referer": str(pub.pub_url)
                    if pub.pub_url
                    else "https://growthlab.hks.harvard.edu/publications",
                }

                # Log endnote download attempt
                logger.info(f"Downloading endnote file from {endnote_url}")

                async with session.get(
                    endnote_url, headers=headers, timeout=30
                ) as response:
                    if response.status != 200:
                        logger.warning(
                            f"Non-200 status code ({response.status}) when fetching endnote: {endnote_url}"
                        )
                        # Raise exception for retry on rate limits or server errors
                        if response.status == 429 or response.status >= 500:
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status,
                                message=f"Rate limited or server error: {response.status}",
                            )
                        return pub

                    content = await response.text()
                    if not content or len(content.strip()) < 10:
                        logger.warning(
                            f"Empty or too short endnote content from {endnote_url}"
                        )
                        return pub

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

                    # Log successful endnote enrichment
                    logger.info(
                        f"Successfully enriched publication {pub.paper_id} with endnote data"
                    )

                    return pub
            except (aiohttp.ClientError, TimeoutError) as e:
                # Log and re-raise to allow the retry mechanism to work
                logger.error(f"Error enriching publication from Endnote: {e}")
                raise
            except Exception as e:
                logger.error(f"Error enriching publication from Endnote: {e}")
                return pub

    async def enrich_publication(
        self, session: aiohttp.ClientSession, pub: Publication
    ) -> Publication:
        """Enrich publication with data from Endnote file with retry mechanism"""
        if not pub.pub_url:
            return pub

        endnote_url = await self.get_endnote_file_url(session, str(pub.pub_url))
        if not endnote_url:
            return pub

        max_retries = self.config.get("max_retries", 3)
        base_delay = self.config.get("retry_base_delay", 1.0)
        max_delay = self.config.get("retry_max_delay", 30.0)

        try:
            return await retry_with_backoff(
                self._enrich_publication_impl,
                session,
                pub,
                endnote_url,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                retry_on=(aiohttp.ClientError, TimeoutError),
            )
        except Exception as e:
            logger.error(
                f"All retries failed for enriching publication {pub.paper_id}: {e}"
            )
            return pub

    async def extract_and_enrich_publications(self) -> list[Publication]:
        """Extract all publications and enrich them with Endnote data"""
        publications = await self.extract_publications()

        # Create more robust session with timeouts
        timeout = aiohttp.ClientTimeout(
            total=60, connect=20, sock_connect=20, sock_read=20
        )
        connector = aiohttp.TCPConnector(
            limit=self.concurrency_limit,
            ttl_dns_cache=300,
            force_close=False,
            enable_cleanup_closed=True,
        )

        # Create a session with custom headers by default
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=headers,
            cookie_jar=aiohttp.CookieJar(unsafe=True),
        ) as session:
            # Log session start
            logger.info(
                f"Starting publication enrichment with concurrency limit {self.concurrency_limit}"
            )

            # Create tasks but limit concurrency through semaphore in enrich_publication
            tasks = [self.enrich_publication(session, pub) for pub in publications]

            # Process enrichment tasks with better error handling
            enriched_publications = []
            endnote_urls_found = 0
            successful_endnote_parses = 0
            failed_enrichments = 0

            # Process with progress bar using as_completed for better error handling
            with async_tqdm(total=len(tasks), desc="Enriching publications") as pbar:
                for future in asyncio.as_completed(tasks):
                    try:
                        pub = await future
                        enriched_publications.append(pub)

                        # Check if the publication has a pub_url (needed for endnote)
                        if pub.pub_url:
                            # Use a simple check to determine if endnote was found and parsed
                            if (
                                pub.abstract
                            ):  # If abstract is present, likely from endnote
                                successful_endnote_parses += 1

                            # Increment URL counter if pub has a URL
                            endnote_urls_found += 1

                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error enriching publication: {e}")
                        failed_enrichments += 1
                        pbar.update(1)

            # Log enrichment statistics
            logger.info(f"Failed enrichments: {failed_enrichments}")
            if failed_enrichments > 0:
                logger.warning(
                    f"Failed to enrich {failed_enrichments} publications due to errors"
                )

        # Calculate percentages for reporting
        endnote_success_rate = (
            (successful_endnote_parses / len(publications) * 100) if publications else 0
        )
        endnote_url_rate = (
            (endnote_urls_found / len(publications) * 100) if publications else 0
        )

        logger.info(f"Enriched {len(enriched_publications)} publications")
        logger.info(
            f"Found {endnote_urls_found} EndNote URLs ({endnote_url_rate:.1f}% of publications)"
        )
        logger.info(
            f"Successfully parsed {successful_endnote_parses} EndNote files ({endnote_success_rate:.1f}% of publications)"
        )
        return enriched_publications

    def save_to_csv(self, publications: list[Publication], output_path: Path) -> None:
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

    def load_from_csv(self, input_path: Path) -> list[Publication]:
        """Load publications from CSV file"""
        if not input_path.exists():
            logger.warning(f"CSV file {input_path} does not exist")
            return []

        try:
            df = pd.read_csv(input_path)
            # Fill NaN values with appropriate defaults to avoid validation errors
            df = df.fillna(
                {
                    "abstract": "",
                    "title": "",
                    "authors": "",
                    "paper_id": "",
                    "content_hash": "",
                    "pub_url": "",
                }
            )
            # Convert string representation of list to actual list for file_urls
            df["file_urls"] = df["file_urls"].apply(
                lambda x: eval(x) if isinstance(x, str) else []
            )
            publications = [Publication(**row) for _, row in df.iterrows()]
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
    ) -> list[Publication]:
        """
        Update publications by comparing existing ones with newly scraped ones

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
            existing_path = storage.get_path("intermediate/growth_lab_publications.csv")
        if not output_path:
            output_path = existing_path

        # Load existing publications if available
        existing_publications = (
            self.load_from_csv(existing_path)
            if existing_path and existing_path.exists()
            else []
        )
        existing_pub_map = {pub.paper_id: pub for pub in existing_publications}
        logger.info(f"Loaded {len(existing_publications)} existing publications")

        # Get new publications
        new_publications = await self.extract_and_enrich_publications()

        # Merge existing and new publications, preferring new ones with different content
        updated_publications = []
        new_pub_map = {}
        updated_count = 0
        new_count = 0
        retained_count = 0
        unchanged_count = 0

        for pub in new_publications:
            pub_id = pub.paper_id
            new_pub_map[pub_id] = pub

            if pub_id in existing_pub_map:
                # If content changed, use new publication
                if pub.content_hash != existing_pub_map[pub_id].content_hash:
                    logger.info(f"Publication {pub_id} updated")
                    updated_publications.append(pub)
                    updated_count += 1
                else:
                    # No changes, use existing
                    updated_publications.append(existing_pub_map[pub_id])
                    unchanged_count += 1
            else:
                # New publication
                logger.info(f"New publication found: {pub_id}")
                updated_publications.append(pub)
                new_count += 1

        # Check for publications that no longer exist on the website
        for pub_id, pub in existing_pub_map.items():
            if pub_id not in new_pub_map:
                logger.info(
                    f"Publication {pub_id} no longer available, retaining in database"
                )
                updated_publications.append(pub)
                retained_count += 1

        # Log summary of publication updates
        total_file_urls = sum(len(pub.file_urls) for pub in updated_publications)
        logger.info("-" * 50)
        logger.info("Publication Update Summary:")
        logger.info(f"Total publications: {len(updated_publications)}")
        logger.info(f"  - New publications: {new_count}")
        logger.info(f"  - Updated publications: {updated_count}")
        logger.info(f"  - Unchanged publications: {unchanged_count}")
        logger.info(f"  - Retained publications (not on website): {retained_count}")
        logger.info(f"Total file URLs across all publications: {total_file_urls}")
        logger.info("-" * 50)

        # Save updated publications
        if output_path:
            # Ensure parent directory exists
            storage.ensure_dir(output_path.parent)
            self.save_to_csv(updated_publications, output_path)

        return updated_publications
