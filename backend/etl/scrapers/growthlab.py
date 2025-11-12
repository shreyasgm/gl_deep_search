"""
Scraper module for the Growth Lab website publications
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, TypeVar

import aiohttp
import pandas as pd
import yaml
from bs4 import BeautifulSoup
from bs4.element import ResultSet, Tag
from tqdm.asyncio import tqdm as async_tqdm

from backend.etl.models.publications import GrowthLabPublication
from backend.etl.utils.publication_tracker import PublicationTracker
from backend.etl.utils.retry import retry_with_backoff
from backend.storage.factory import get_storage

logger = logging.getLogger(__name__)

# Type variable for generic retry function
T = TypeVar("T")


class GrowthLabScraper:
    """Scraper for Growth Lab website publications"""

    def __init__(
        self,
        config_path: Path | None = None,
        concurrency_limit: int | None = None,
        tracker: PublicationTracker | None = None,
    ):
        """
        Initialize the scraper with configuration

        Args:
            config_path: Path to the configuration file
            concurrency_limit: Maximum number of concurrent requests (default from config)
            tracker: Optional PublicationTracker instance for registering publications
        """
        self.config = self._load_config(config_path)
        self.base_url = self.config["base_url"]
        self.scrape_delay = self.config["scrape_delay"]
        self.tracker = tracker

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
                "base_url": "https://growthlab.hks.harvard.edu/publications-home/repository",
                "scrape_delay": 2.5,
                "concurrency_limit": 5,
                "max_retries": 3,
                "retry_base_delay": 1.0,
                "retry_max_delay": 30.0,
            }

    async def _get_max_page_num_impl(
        self, session: aiohttp.ClientSession, url: str
    ) -> int:
        """Implementation to get the maximum page number from pagination

        For FacetWP-based sites without visible pagination links,
        we discover pages by trying sequential pages until we get an empty result.
        """
        async with self.semaphore:
            # First try to find pagination links in HTML (old structure)
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch {url}: {response.status}")
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

                # Try old structure: Drupal pager
                pagination = soup.find("ul", {"class": "pager"})
                if pagination:
                    last_page_link = pagination.find("li", {"class": "pager-last"})
                    if last_page_link and last_page_link.find("a"):
                        last_page_url = last_page_link.find("a").get("href")
                        match = re.search(r"\d+", last_page_url)
                        if match:
                            return int(match.group())

                # Try to find pagination links (may not exist with JavaScript-loaded pagination)
                pagination_links = soup.find_all(
                    "a",
                    href=lambda x: x and ("fwp_paged" in str(x) or "/page/" in str(x)),
                )
                if pagination_links:
                    max_page = 0
                    for link in pagination_links:
                        href = link.get("href", "")
                        match = re.search(r"(?:fwp_paged=|/page/)(\d+)", href)
                        if match:
                            page_num = int(match.group(1))
                            max_page = max(max_page, page_num)
                    if max_page > 0:
                        return max_page

                # Pagination links not found - use binary search discovery for FacetWP
                logger.info(
                    "Pagination links not found, using binary search discovery..."
                )

                # Binary search to find the last page
                low, high = 0, 200  # Search up to page 200
                max_page_found = 0

                while low <= high:
                    mid = (low + high) // 2
                    test_url = url if mid == 0 else f"{url}?fwp_paged={mid}"

                    try:
                        async with session.get(test_url) as test_response:
                            if test_response.status == 200:
                                test_html = await test_response.text()
                                test_soup = BeautifulSoup(test_html, "html.parser")
                                test_items = test_soup.find_all(
                                    "li",
                                    {
                                        "class": lambda x: x
                                        and "wp-block-post" in str(x)
                                    },
                                )
                                if not test_items:
                                    test_items = test_soup.find_all(
                                        "div", {"class": "cp-publication"}
                                    )

                                if len(test_items) > 0:
                                    max_page_found = mid
                                    low = mid + 1  # Try higher pages
                                else:
                                    high = mid - 1  # No items, try lower pages
                                await asyncio.sleep(0.2)  # Small delay
                            else:
                                high = mid - 1  # Error, try lower
                    except Exception as e:
                        logger.debug(f"Error checking page {mid}: {e}")
                        high = mid - 1

                return max_page_found

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
    ) -> GrowthLabPublication | None:
        """Parse a single publication element

        Supports both old (biblio-entry) and new (cp-publication) HTML structures
        """
        try:
            # Try new structure first (cp-publication)
            title_element = pub_element.find("h2", {"class": "publication-title"})
            if not title_element:
                # Fall back to old structure
                title_element = pub_element.find("span", {"class": "biblio-title"})

            if not title_element:
                return None

            # Extract title and URL
            title_link = title_element.find("a")
            if title_link:
                title = title_link.text.strip()
                pub_url = title_link.get("href")
            else:
                title = title_element.text.strip()
                pub_url = None

            # Ensure URL is absolute
            if pub_url and not pub_url.startswith(("http://", "https://")):
                base_domain = (
                    base_url.split("/publications")[0]
                    if "/publications" in base_url
                    else base_url.split("/repository")[0]
                )
                pub_url = f"{base_domain}{pub_url}"

            # Extract authors (new structure: p.publication-authors, old: span.biblio-authors)
            authors_element = pub_element.find("p", {"class": "publication-authors"})
            if not authors_element:
                authors_element = pub_element.find("span", {"class": "biblio-authors"})

            authors = None
            year = None

            if authors_element:
                # In new structure, year is inside publication-authors as a span
                year_span = authors_element.find("span", {"class": "publication-year"})
                if year_span:
                    # Extract year from the span text (format: ", 2025")
                    year_text = year_span.text.strip()
                    year_match = re.search(r"\b\d{4}\b", year_text)
                    if year_match:
                        year = int(year_match.group())
                    # Remove year from authors text
                    authors = (
                        authors_element.text.replace(year_span.text, "")
                        .strip()
                        .rstrip(",")
                        .strip()
                    )
                else:
                    # Old structure: year is sibling of authors_element
                    authors = authors_element.text.strip()
                    sibling_text = authors_element.next_sibling
                    if sibling_text:
                        year_match = re.search(r"\b\d{4}\b", sibling_text)
                        if year_match:
                            year = int(year_match.group())

            # Apply year correction if available
            if pub_url in self.year_corrections:
                year = self.year_corrections[pub_url]

            # Extract abstract (new: div.publication-excerpt, old: div.biblio-abstract-display)
            abstract_element = pub_element.find("div", {"class": "publication-excerpt"})
            if not abstract_element:
                abstract_element = pub_element.find(
                    "div", {"class": "biblio-abstract-display"}
                )

            abstract = None
            if abstract_element:
                # In new structure, abstract might be in a nested div
                abstract_div = abstract_element.find("div")
                if abstract_div:
                    abstract = abstract_div.text.strip()
                else:
                    abstract = abstract_element.text.strip()

            # Get file URLs (new: div.publication-links, old: span.file)
            file_urls = []
            links_container = pub_element.find("div", {"class": "publication-links"})
            if links_container:
                # Look for links that aren't abstract buttons
                for link in links_container.find_all("a", href=True):
                    href = link.get("href")
                    # Skip abstract buttons and internal links
                    if (
                        href
                        and not href.startswith("#")
                        and "abstract" not in link.get("class", [])
                    ):
                        if not href.startswith(("http://", "https://")):
                            base_domain = (
                                base_url.split("/publications")[0]
                                if "/publications" in base_url
                                else base_url.split("/repository")[0]
                            )
                            href = f"{base_domain}{href}"
                        file_urls.append(href)
            else:
                # Old structure: span.file
                for file_elem in pub_element.find_all("span", {"class": "file"}):
                    file_link = file_elem.find("a")
                    if file_link and file_link.get("href"):
                        file_url = file_link["href"]
                        if not file_url.startswith(("http://", "https://")):
                            base_domain = (
                                base_url.split("/publications")[0]
                                if "/publications" in base_url
                                else base_url.split("/repository")[0]
                            )
                            file_url = f"{base_domain}{file_url}"
                        file_urls.append(file_url)

            pub = GrowthLabPublication(
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
    ) -> list[GrowthLabPublication]:
        """Implementation to fetch a single page of publications"""
        # Build URL with pagination (try FacetWP format first, then fall back to ?page=)
        if page_num == 0:
            url = self.base_url
        else:
            # Try FacetWP pagination format
            if "fwp_paged" not in self.base_url:
                url = f"{self.base_url}?fwp_paged={page_num}"
            else:
                url = f"{self.base_url}?page={page_num}"

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

                    # Try new structure first: look for cp-publication divs or li.wp-block-post
                    pub_elements: ResultSet[Tag] = soup.find_all(
                        "div", {"class": "cp-publication"}
                    )

                    # If not found, try finding li elements with cp-publication nested
                    if not pub_elements:
                        post_items = soup.find_all(
                            "li", {"class": lambda x: x and "wp-block-post" in str(x)}
                        )
                        pub_elements = ResultSet([])  # type: ignore[call-overload]
                        for item in post_items:
                            cp_pub = item.find("div", {"class": "cp-publication"})
                            if cp_pub:
                                pub_elements.append(cp_pub)

                    # Fall back to old structure
                    if not pub_elements:
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
    ) -> list[GrowthLabPublication]:
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

    async def extract_publications(
        self, limit: int | None = None
    ) -> list[GrowthLabPublication]:
        """
        Extract publications from the Growth Lab website.

        Args:
            limit: Optional limit on number of publications to extract.
                   If set, stops fetching pages once limit is reached.

        Returns:
            List of extracted publications
        """
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

            # If limit is set, estimate how many pages we need
            # (typically ~10-20 publications per page)
            if limit:
                estimated_pages_needed = max(1, (limit // 15) + 1)
                pages_to_fetch = min(estimated_pages_needed, max_page_num + 1)
                logger.info(
                    f"Limiting to {limit} publications, fetching approximately "
                    f"{pages_to_fetch} pages"
                )
            else:
                pages_to_fetch = max_page_num + 1

            # Create a list of pages to process
            all_pages = list(range(pages_to_fetch))

            # Process pages using semaphore-controlled concurrency
            all_publications: list[GrowthLabPublication] = []
            total_file_urls = 0
            failed_pages = 0

            # Process pages sequentially if limit is small to avoid unnecessary work
            # Otherwise use concurrent processing
            if limit and limit <= 10:
                # For small limits, process pages sequentially to stop early
                with async_tqdm(total=pages_to_fetch, desc="Scraping pages") as pbar:
                    for page_num in all_pages:
                        if limit and len(all_publications) >= limit:
                            logger.info(
                                f"Reached limit of {limit} publications, "
                                f"stopping page fetching"
                            )
                            break
                        try:
                            publications = await self.fetch_page(session, page_num)
                            all_publications.extend(publications)
                            # Count file URLs
                            for pub in publications:
                                total_file_urls += len(pub.file_urls)
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"Error processing page {page_num}: {e}")
                            failed_pages += 1
                            pbar.update(1)
            else:
                # For larger limits or no limit, use concurrent processing
                tasks = [self.fetch_page(session, page_num) for page_num in all_pages]

                # Process tasks with progress bar using as_completed
                with async_tqdm(total=len(tasks), desc="Scraping pages") as pbar:
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

            # Apply limit if we exceeded it
            if limit and len(all_publications) > limit:
                all_publications = all_publications[:limit]
                total_file_urls = sum(
                    len(pub.file_urls) for pub in all_publications if pub.file_urls
                )

            if failed_pages > 0:
                logger.warning(f"Failed to process {failed_pages} pages due to errors")

            logger.info(
                f"Extracted {len(all_publications)} publications with "
                f"{total_file_urls} total file URLs"
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
        self,
        session: aiohttp.ClientSession,
        pub: GrowthLabPublication,
        endnote_url: str,
    ) -> GrowthLabPublication:
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

    async def enrich_publication_from_page(
        self, session: aiohttp.ClientSession, pub: GrowthLabPublication
    ) -> GrowthLabPublication:
        """Enrich publication by extracting full metadata from the publication page

        This replaces the old Endnote-based enrichment since Endnote files
        are no longer available on the new website structure.
        """
        if not pub.pub_url:
            return pub

        async with self.semaphore:
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                }

                async with session.get(str(pub.pub_url), headers=headers) as response:
                    if response.status != 200:
                        logger.warning(
                            f"Failed to fetch publication page {pub.pub_url}: {response.status}"
                        )
                        return pub

                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")

                    # Extract full abstract from publication page
                    abstract_divs = soup.find_all(
                        "div", {"class": lambda x: x and "abstract" in str(x).lower()}
                    )
                    for div in abstract_divs:
                        text = div.get_text().strip()
                        # Skip the word "Abstract" itself, get the actual content
                        if text.lower().startswith("abstract"):
                            text = text[8:].strip()  # Remove "Abstract" prefix
                        # Use the longest abstract we find (likely the full one)
                        if len(text) > len(pub.abstract or ""):
                            pub.abstract = text

                    # Extract metadata from JSON-LD if available
                    scripts = soup.find_all("script", {"type": "application/ld+json"})
                    for script in scripts:
                        try:
                            data = json.loads(script.string)
                            # Handle @graph format
                            graph = data.get("@graph", [])
                            if (
                                isinstance(data, dict)
                                and data.get("@type") == "ScholarlyArticle"
                            ):
                                graph = [data]  # Single item as list

                            for item in graph:
                                if item.get("@type") == "ScholarlyArticle":
                                    # Extract datePublished if year is missing
                                    if not pub.year and "datePublished" in item:
                                        date_str = item["datePublished"]
                                        year_match = re.search(r"\b\d{4}\b", date_str)
                                        if year_match:
                                            pub.year = int(year_match.group())

                                    # Extract authors if missing or incomplete
                                    if "author" in item:
                                        author = item.get("author", {})
                                        if isinstance(author, dict):
                                            author_name = author.get("name", "")
                                            # Only update if we don't have authors or if JSON-LD has better data
                                            if not pub.authors or (
                                                author_name
                                                and len(author_name)
                                                > len(pub.authors or "")
                                            ):
                                                pub.authors = author_name
                                        elif isinstance(author, list) and author:
                                            # Handle list of authors
                                            author_names = [
                                                a.get("name", "")
                                                if isinstance(a, dict)
                                                else str(a)
                                                for a in author
                                            ]
                                            if author_names:
                                                pub.authors = ", ".join(author_names)
                        except Exception as e:
                            logger.debug(f"Failed to parse JSON-LD: {e}")
                            continue

                    # Update content hash after enrichment
                    pub.content_hash = pub.generate_content_hash()

                    logger.debug(f"Enriched publication {pub.paper_id} from page")
                    return pub
            except Exception as e:
                logger.error(f"Error enriching publication from page: {e}")
                return pub

    async def enrich_publication(
        self, session: aiohttp.ClientSession, pub: GrowthLabPublication
    ) -> GrowthLabPublication:
        """Enrich publication with data from Endnote file or publication page

        First tries to get Endnote file (for backward compatibility with old structure).
        If no Endnote file exists, extracts metadata directly from the publication page.
        """
        if not pub.pub_url:
            return pub

        # Try Endnote file first (for old publications)
        endnote_url = await self.get_endnote_file_url(session, str(pub.pub_url))
        if endnote_url:
            # Use old Endnote-based enrichment
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
                    f"All retries failed for enriching publication {pub.paper_id} from Endnote: {e}"
                )
                # Fall through to page-based enrichment

        # If no Endnote file, enrich from publication page directly
        return await self.enrich_publication_from_page(session, pub)

    async def extract_and_enrich_publications(
        self, limit: int | None = None
    ) -> list[GrowthLabPublication]:
        """
        Extract publications and enrich them with metadata from publication pages.

        Args:
            limit: Optional limit on number of publications to extract.

        Returns:
            List of enriched publications
        """
        publications = await self.extract_publications(limit=limit)

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

                        # Track enrichment success
                        if pub.pub_url:
                            # Check if enrichment improved the abstract (full abstract from page)
                            if pub.abstract and len(pub.abstract) > 200:
                                successful_endnote_parses += 1

                            # Count publications with URLs (can be enriched)
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
            f"Publications with URLs (can be enriched): {endnote_urls_found} ({endnote_url_rate:.1f}% of publications)"
        )
        logger.info(
            f"Successfully enriched {successful_endnote_parses} publications with full metadata ({endnote_success_rate:.1f}% of publications)"
        )
        return enriched_publications

    def save_to_csv(
        self, publications: list[GrowthLabPublication], output_path: Path
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

    def load_from_csv(self, input_path: Path) -> list[GrowthLabPublication]:
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
            publications = [GrowthLabPublication(**row) for _, row in df.iterrows()]
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
        limit: int | None = None,
    ) -> list[GrowthLabPublication]:
        """
        Update publications by comparing existing ones with newly scraped ones

        This handles updates to existing publications by comparing content hashes

        Args:
            existing_path: Optional path to existing publications CSV
            output_path: Optional path to save updated publications
            storage: Optional storage instance (will use default if None)
            limit: Optional limit on number of publications to scrape
        """
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

        # Get new publications (with limit if specified)
        new_publications = await self.extract_and_enrich_publications(limit=limit)

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

        # Register publications with tracker if provided
        if self.tracker:
            logger.info("Registering publications with tracker")
            for pub in updated_publications:
                try:
                    self.tracker.add_publication(pub)
                    logger.debug(f"Registered publication {pub.paper_id} with tracker")
                except Exception as e:
                    logger.error(
                        f"Failed to register publication {pub.paper_id} with tracker: {e}"
                    )
                    # Continue processing other publications even if one fails

        return updated_publications
