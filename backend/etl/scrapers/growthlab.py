"""
Scraper module for the Growth Lab website publications
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, TypeVar

import aiohttp
import pandas as pd
import yaml
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm as async_tqdm

from backend.etl.models.publications import GrowthLabPublication
from backend.etl.utils.retry import retry_with_backoff
from backend.storage.factory import get_storage

logger = logging.getLogger(__name__)

# Type variable for generic retry function
T = TypeVar("T")

# adding selector config here - endnote section needs updating with second round, all title bits seemed to work fine, files had maybes
SELECTOR_CONFIG = {
    "publication": {
        "container": {
            "primary": "div.biblio-entry",
            "fallbacks": [
                "div.node-biblio",
                "article.publication",
                "div.publication-item",
            ],
            "xpath": "//div[contains(@class, 'biblio-entry')]",
            "description": "Publication container",
        },
        "title": {
            "primary": "span.biblio-title",
            "fallbacks": [
                "h1.page-title",
                "h2.publication-title",
                "h3.title",
                "div.title",
            ],
            "xpath": "//span[contains(@class, 'biblio-title')] | //h1[contains(@class, 'page-title')]",
            "description": "Publication title",
        },
        "authors": {
            "primary": "span.biblio-authors",
            "fallbacks": [
                "div.field-name-field-biblio-authors .field-item",
                "div.authors",
                "p.author-list",
                "div.publication-authors",
            ],
            "xpath": "//span[contains(@class, 'biblio-authors')] | //div[contains(@class, 'field-name-field-biblio-authors')]//div[@class='field-item']",
            "description": "Publication authors",
        },
        # abstract might need another try with selectorgadget
        "abstract": {
            "primary": "div.biblio-abstract-display",
            "fallbacks": [
                "div.field-name-field-abstract",
                "div.abstract",
                "div.publication-abstract",
                "p.abstract",
            ],
            "xpath": "//div[contains(@class, 'biblio-abstract-display')] | //div[contains(@class, 'field-name-field-abstract')]",
            "description": "Publication abstract",
        },
        "file": {
            "primary": "span.file",
            "fallbacks": [
                "#pub-cover-content-wrapper a",  # SelectorGadget discovery
                ".Z3988+ a",  # Element after Z3988
                "a.biblio-download",
                "a[href$='.pdf']",
                "a[href*='files']",
            ],
            "xpath": "//span[contains(@class, 'file')] | //*[@id='pub-cover-content-wrapper']//a | //*[contains(@class, 'Z3988')]/following-sibling::a | //a[contains(@href, '.pdf')]",
            "description": "Publication files",
        },
        "citation": {
            "primary": ".biblio-citation",
            "fallbacks": ["div.field-name-field-citation", "div.citation"],
            "xpath": "//div[contains(@class, 'biblio-citation')] | //div[contains(@class, 'field-name-field-citation')]",
            "description": "Citation information",
        },
    },
    "pagination": {
        "container": {
            "primary": "ul.pager",
            "fallbacks": ["div.pagination", "nav.pagination"],
            "xpath": "//ul[contains(@class, 'pager')]",
            "description": "Pagination container",
        },
        "last_page": {
            "primary": "li.pager-last",
            "fallbacks": ["li.page-item:last-child", "a.page-link:last-child"],
            "xpath": "//li[contains(@class, 'pager-last')]",
            "description": "Last page link",
        },
    },
    "endnote": {
        "link": {
            "primary": "li.biblio_tagged a",
            "fallbacks": [
                "a[href*='tagged=1']",
                "a[href*='endnote']",
                "a.endnote-link",
            ],
            "xpath": "//a[contains(@href, 'tagged=1')] | //a[contains(@href, 'endnote')]",
            "description": "Endnote link",
        }
    },
}


class SelectorMonitor:
    """Class to monitor selector performance and detect failures"""

    def __init__(self, selectors=None):
        self.selectors = selectors or SELECTOR_CONFIG
        self.stats = {
            "total_pages": 0,
            "total_publications": 0,
            "selector_success": {},
            "selector_failure": {},
            "alerts": [],
        }

        # Initialize stats for each selector
        for section, section_config in self.selectors.items():
            for name, _ in section_config.items():
                key = f"{section}.{name}"
                self.stats["selector_success"][key] = 0
                self.stats["selector_failure"][key] = 0

        # Track which selectors are actually used
        self.selector_usage = {}

    def record_success(self, section, name, used_selector=None):
        """Record a successful selector use"""
        key = f"{section}.{name}"
        if key in self.stats["selector_success"]:
            self.stats["selector_success"][key] += 1

            # Record which selector was actually used
            if used_selector:
                if key not in self.selector_usage:
                    self.selector_usage[key] = {}

                self.selector_usage[key][used_selector] = (
                    self.selector_usage[key].get(used_selector, 0) + 1
                )

    def record_failure(self, section, name):
        """Record a failed selector use"""
        key = f"{section}.{name}"
        if key in self.stats["selector_failure"]:
            self.stats["selector_failure"][key] += 1

            # Check if failure rate is high enough to trigger alert
            total = (
                self.stats["selector_success"][key]
                + self.stats["selector_failure"][key]
            )
            if total >= 5:  # Only check after a minimum sample
                failure_rate = self.stats["selector_failure"][key] / total
                if failure_rate > 0.5:  # Alert if more than 50% failure
                    self.create_alert(section, name, failure_rate)

    def create_alert(self, section, name, failure_rate):
        """Create an alert for a failing selector"""
        selector_config = self.selectors[section][name]
        alert = {
            "selector": f"{section}.{name}",
            "failure_rate": failure_rate,
            "primary": selector_config["primary"],
            "fallbacks": selector_config["fallbacks"],
            "message": f"Selector {section}.{name} is failing at a rate of {failure_rate:.2%}",
        }

        # Check if we already have an alert for this selector
        existing_alerts = [
            a for a in self.stats["alerts"] if a["selector"] == alert["selector"]
        ]
        if not existing_alerts:
            self.stats["alerts"].append(alert)
            logger.warning(f"SELECTOR ALERT: {alert['message']}")

    def record_page_processed(self):
        """Record that a page was processed"""
        self.stats["total_pages"] += 1

    def record_publication_processed(self):
        """Record that a publication was processed"""
        self.stats["total_publications"] += 1

    def check_selector_health(self):
        """Check the health of all selectors"""
        logger.info("\nSelector Health Check:")

        for section, section_config in self.selectors.items():
            logger.info(f"\n{section.upper()} Selectors:")

            for name, _ in section_config.items():
                key = f"{section}.{name}"
                success = self.stats["selector_success"].get(key, 0)
                failure = self.stats["selector_failure"].get(key, 0)
                total = success + failure

                if total > 0:
                    success_rate = success / total
                    status = (
                        "GOOD"
                        if success_rate >= 0.9
                        else "WARNING"
                        if success_rate >= 0.5
                        else "FAILING"
                    )
                    logger.info(
                        f"  - {key}: {status} ({success}/{total}, {success_rate:.1%})"
                    )
                else:
                    logger.info(f"  - {key}: NO DATA")

    def generate_report(self):
        """Generate a full report of selector performance"""
        logger.info("\nSelector Performance Report")
        logger.info(f"Pages processed: {self.stats['total_pages']}")
        logger.info(f"Publications processed: {self.stats['total_publications']}")

        # Calculate overall selector success rate
        total_success = sum(self.stats["selector_success"].values())
        total_failure = sum(self.stats["selector_failure"].values())
        total_attempts = total_success + total_failure

        if total_attempts > 0:
            overall_rate = total_success / total_attempts
            logger.info(f"Overall selector success rate: {overall_rate:.2%}")

        # Print selector-specific stats
        logger.info("\nSelector Performance:")
        for key in sorted(self.stats["selector_success"].keys()):
            success = self.stats["selector_success"][key]
            failure = self.stats["selector_failure"][key]
            total = success + failure

            if total > 0:
                rate = success / total
                status = (
                    "GOOD" if rate >= 0.9 else "WARNING" if rate >= 0.5 else "FAILING"
                )
                logger.info(
                    f"  - {key}: {status} {rate:.2%} success ({success}/{total})"
                )

        # Print selector usage statistics
        logger.info("\nSelector Usage Statistics:")
        for key, usage in self.selector_usage.items():
            if not usage:
                continue

            logger.info(f"\n{key}:")
            total_uses = sum(usage.values())

            for selector, count in sorted(
                usage.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = count / total_uses * 100
                logger.info(f"  - {selector}: {count} times ({percentage:.1f}%)")


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

        # set concurrency limit (from parameter or config)
        self.concurrency_limit = concurrency_limit or self.config.get(
            "concurrency_limit", 5
        )
        # create a semaphore for concurrency control
        self.semaphore = asyncio.Semaphore(self.concurrency_limit)

        # create selector monitor
        self.monitor = SelectorMonitor(SELECTOR_CONFIG)

        # fmt: off
        #ruff: noqa: E501
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

    # Selector utility functions
    def _find_with_selectors(self, element, selector_config, section=None, name=None):
        """Find an element using selectors with fallbacks

        Args:
            element: BeautifulSoup element to search within
            selector_config: Configuration with primary, fallbacks, xpath
            section: Section name for monitoring
            name: Selector name for monitoring

        Returns:
            Tuple of (element, selector_used) or (None, None) if not found
        """
        # Try primary selector
        if selector_config["primary"]:
            try:
                results = element.select(selector_config["primary"])
                if results:
                    if section and name:
                        self.monitor.record_success(
                            section, name, selector_config["primary"]
                        )
                    return results[0], selector_config["primary"]
            except Exception as e:
                logger.debug(
                    f"Error with primary selector {selector_config['primary']}: {e}"
                )

        # Try fallbacks
        for fallback in selector_config.get("fallbacks", []):
            try:
                results = element.select(fallback)
                if results:
                    if section and name:
                        self.monitor.record_success(section, name, fallback)
                    return results[0], fallback
            except Exception as e:
                logger.debug(f"Error with fallback selector {fallback}: {e}")

        # Try XPath as last resort
        if "xpath" in selector_config and selector_config["xpath"]:
            try:
                import lxml.html
                from lxml import etree

                # Parse the HTML of the element
                dom = lxml.html.fromstring(str(element))
                xpath_results = dom.xpath(selector_config["xpath"])

                if xpath_results:
                    if section and name:
                        self.monitor.record_success(section, name, "xpath")

                    # Convert back to BeautifulSoup for consistency
                    result_html = etree.tostring(xpath_results[0])
                    result_soup = BeautifulSoup(result_html, "html.parser")
                    if result_soup.contents:
                        return result_soup.contents[0], "xpath"
            except ImportError:
                logger.debug("lxml not available for XPath queries")
            except Exception as e:
                logger.debug(f"Error with XPath: {e}")

        # Record failure if we got here
        if section and name:
            self.monitor.record_failure(section, name)

        return None, None

    def _find_all_with_selectors(
        self, element, selector_config, section=None, name=None
    ):
        """Find all elements matching selectors with fallbacks

        Returns:
            Tuple of (elements, selector_used) or ([], None) if none found
        """
        # Try primary selector
        if selector_config["primary"]:
            try:
                results = element.select(selector_config["primary"])
                if results:
                    if section and name:
                        self.monitor.record_success(
                            section, name, selector_config["primary"]
                        )
                    return results, selector_config["primary"]
            except Exception as e:
                logger.debug(
                    f"Error with primary selector {selector_config['primary']}: {e}"
                )

        # Try fallbacks
        for fallback in selector_config.get("fallbacks", []):
            try:
                results = element.select(fallback)
                if results:
                    if section and name:
                        self.monitor.record_success(section, name, fallback)
                    return results, fallback
            except Exception as e:
                logger.debug(f"Error with fallback selector {fallback}: {e}")

        # Try XPath
        if "xpath" in selector_config and selector_config["xpath"]:
            try:
                import lxml.html
                from lxml import etree

                dom = lxml.html.fromstring(str(element))
                xpath_results = dom.xpath(selector_config["xpath"])

                if xpath_results:
                    if section and name:
                        self.monitor.record_success(section, name, "xpath")

                    bs_results = []
                    for result in xpath_results:
                        result_html = etree.tostring(result)
                        result_soup = BeautifulSoup(result_html, "html.parser")
                        if result_soup.contents:
                            bs_results.append(result_soup.contents[0])

                    if bs_results:
                        return bs_results, "xpath"
            except ImportError:
                logger.debug("lxml not available for XPath queries")
            except Exception as e:
                logger.debug(f"Error with XPath: {e}")

        # Record failure
        if section and name:
            self.monitor.record_failure(section, name)

        return [], None

    # Extract components from citation
    def _extract_from_citation(self, citation_text):
        """Extract author, year, and title from a citation string"""
        result = {}

        if not citation_text:
            return result

        # Try to extract year
        year_match = re.search(r"(\d{4})\. ", citation_text)
        if year_match:
            year_pos = year_match.start()
            after_year = year_match.end()

            # Author is everything before the year
            result["authors"] = citation_text[:year_pos].strip()
            if result["authors"].endswith(","):
                result["authors"] = result["authors"][:-1]

            # Title is everything after the year until next period or end
            title_end = citation_text.find(".", after_year)
            if title_end > after_year:
                result["title"] = citation_text[after_year:title_end].strip()
            else:
                result["title"] = citation_text[after_year:].strip()

            # Extract year as integer
            try:
                result["year"] = int(year_match.group(1))
            except:
                pass

        return result

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

                # Find pagination container using configurable selectors
                pagination, pagination_selector = self._find_with_selectors(
                    soup,
                    SELECTOR_CONFIG["pagination"]["container"],
                    "pagination",
                    "container",
                )

                if not pagination:
                    logger.error("No pagination element found")
                    return 0

                # Find last page link
                last_page, last_page_selector = self._find_with_selectors(
                    pagination,
                    SELECTOR_CONFIG["pagination"]["last_page"],
                    "pagination",
                    "last_page",
                )

                if last_page and last_page.find("a"):
                    last_page_url = last_page.find("a").get("href")
                    match = re.search(r"\d+", last_page_url)
                    if match:
                        return int(match.group())

                return 0

    async def get_max_page_num(self, session: aiohttp.ClientSession, url: str) -> int:
        """Get the maximum page number from pagination with retry mechanism"""
        max_retries = self.config.get("max_retries", 3)
        base_delay = self.config.get("retry_base_delay", 1.0)
        max_delay = self.config.get("retry_max_delay", 30.0)

        # Record that we're processing a page
        self.monitor.record_page_processed()

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
        """Parse a single publication element using configurable selectors"""
        try:
            # Record that we're processing a publication
            self.monitor.record_publication_processed()

            # 1. Find title using selectors
            title_element, title_selector = self._find_with_selectors(
                pub_element,
                SELECTOR_CONFIG["publication"]["title"],
                "publication",
                "title",
            )

            if not title_element:
                logger.warning("No title element found, skipping publication")
                return None

            title = title_element.text.strip()
            title_link = title_element.find("a")
            pub_url = title_link.get("href") if title_link else None

            # Ensure URL is absolute
            if pub_url and not pub_url.startswith(("http://", "https://")):
                pub_url = f"{base_url.split('/publications')[0]}{pub_url}"

            # 2. Find authors using selectors
            authors_element, authors_selector = self._find_with_selectors(
                pub_element,
                SELECTOR_CONFIG["publication"]["authors"],
                "publication",
                "authors",
            )

            authors = authors_element.text.strip() if authors_element else None

            # 3. Extract year from text after authors or using selectors
            year = None
            if authors_element:
                sibling_text = authors_element.next_sibling
                if sibling_text:
                    year_match = re.search(r"\b\d{4}\b", str(sibling_text))
                    if year_match:
                        year = int(year_match.group())

            # Apply year correction if available
            if pub_url in self.year_corrections:
                year = self.year_corrections[pub_url]

            # 4. Find abstract using selectors
            abstract_element, abstract_selector = self._find_with_selectors(
                pub_element,
                SELECTOR_CONFIG["publication"]["abstract"],
                "publication",
                "abstract",
            )

            abstract = abstract_element.text.strip() if abstract_element else None

            # 5. Find file URLs using selectors
            file_elements, file_selector = self._find_all_with_selectors(
                pub_element,
                SELECTOR_CONFIG["publication"]["file"],
                "publication",
                "file",
            )

            file_urls = []
            for elem in file_elements:
                if elem.name == "a" and elem.get("href"):
                    # Direct link
                    file_url = elem["href"]
                    if not file_url.startswith(("http://", "https://")):
                        file_url = f"{base_url.split('/publications')[0]}{file_url}"
                    file_urls.append(file_url)
                else:
                    # Container with links
                    for file_link in elem.find_all("a"):
                        if file_link and file_link.get("href"):
                            file_url = file_link["href"]
                            if not file_url.startswith(("http://", "https://")):
                                file_url = (
                                    f"{base_url.split('/publications')[0]}{file_url}"
                                )
                            file_urls.append(file_url)

            # 6. Fallback to citation if critical fields are missing
            if not title or not authors or not year:
                citation_element, citation_selector = self._find_with_selectors(
                    pub_element,
                    SELECTOR_CONFIG["publication"]["citation"],
                    "publication",
                    "citation",
                )

                if citation_element:
                    citation_text = citation_element.text.strip()
                    citation_data = self._extract_from_citation(citation_text)

                    # Only use citation data for missing fields
                    if not title and "title" in citation_data:
                        title = citation_data["title"]
                    if not authors and "authors" in citation_data:
                        authors = citation_data["authors"]
                    if not year and "year" in citation_data:
                        year = citation_data["year"]

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

                    # Find publication containers using configurable selectors
                    pub_elements, container_selector = self._find_all_with_selectors(
                        soup,
                        SELECTOR_CONFIG["publication"]["container"],
                        "publication",
                        "container",
                    )

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

        # Record that we're processing a page
        self.monitor.record_page_processed()

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

    async def extract_publications(self) -> list[GrowthLabPublication]:
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

            # Generate a report on selector performance
            self.monitor.check_selector_health()

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

                    # Use configurable selectors to find endnote link
                    endnote_element, endnote_selector = self._find_with_selectors(
                        soup, SELECTOR_CONFIG["endnote"]["link"], "endnote", "link"
                    )

                    if endnote_element and endnote_element.get("href"):
                        return endnote_element.get("href")

                    # Fallback to old method if no element found with configurable selectors
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

        # Skip empty content
        if not lines or all(not line.strip() for line in lines):
            logger.warning("Empty EndNote content received")
            return record

        # Initialize current key to handle multiline values
        current_key = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("%"):
                # New key detected
                key = line[1]
                value = line[3:].strip() if len(line) > 2 else ""
                current_key = key

                if key == "A":  # Author
                    name_parts = value.split(", ")
                    if len(name_parts) == 2:
                        value = f"{name_parts[1]} {name_parts[0]}"
                    record["author"] = (
                        record.get("author", []) + [value] if value else []
                    )
                elif key == "T":  # Title
                    record["title"] = value
                elif key == "D":  # Date
                    record["date"] = value
                elif key == "X":  # Abstract
                    record["abstract"] = record.get("abstract", "") + value
            elif current_key == "X":
                # Append to existing abstract for multiline abstracts
                record["abstract"] = record.get("abstract", "") + " " + line

        # Process HTML in abstract if present
        if "abstract" in record and record["abstract"]:
            try:
                soup = BeautifulSoup(record["abstract"], "html.parser")
                for tag in soup.find_all(["b", "strong"]):
                    tag.unwrap()

                abstract_text = []
                for p in soup.find_all("p"):
                    if p.get_text(strip=True):
                        abstract_text.append(p.get_text(separator=" ", strip=True))

                if abstract_text:
                    record["abstract"] = "\n".join(abstract_text)
                else:
                    # If no <p> tags found, use the entire text
                    record["abstract"] = soup.get_text(separator=" ", strip=True)
            except Exception as e:
                logger.warning(f"Error processing HTML in abstract: {e}")

        if "author" in record and isinstance(record["author"], list):
            record["author"] = ", ".join(filter(None, record["author"]))

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
                    "Accept": "text/plain,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Connection": "keep-alive",
                    "Referer": str(pub.pub_url)
                    if pub.pub_url
                    else "https://growthlab.hks.harvard.edu/publications",
                }

                # Log endnote download attempt with details
                logger.info(
                    f"Downloading endnote file from {endnote_url} for publication: {pub.title or 'Unknown title'}"
                )

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
                    content_length = len(content) if content else 0

                    if not content or content_length < 10:
                        logger.warning(
                            f"Empty or too short endnote content ({content_length} bytes) from {endnote_url}"
                        )
                        return pub

                    # Log the first 100 chars for debugging
                    logger.debug(
                        f"First 100 chars of EndNote content: {content[:100].replace('\n', ' ')}"
                    )

                    endnote_data = await self.parse_endnote_content(content)

                    # Log what we found
                    fields_found = ", ".join(endnote_data.keys())
                    logger.info(f"EndNote data fields found: {fields_found}")

                    # Update publication with Endnote data if missing
                    if not pub.title and "title" in endnote_data:
                        pub.title = endnote_data["title"]
                        logger.debug(f"Updated title to: {pub.title}")

                    if not pub.authors and "author" in endnote_data:
                        pub.authors = endnote_data["author"]
                        logger.debug(f"Updated authors to: {pub.authors}")

                    if not pub.abstract and "abstract" in endnote_data:
                        pub.abstract = endnote_data["abstract"]
                        logger.debug(
                            f"Updated abstract with {len(pub.abstract)} characters"
                        )

                    if "date" in endnote_data and not pub.year:
                        try:
                            # Try to extract year from date
                            year_match = re.search(r"\b\d{4}\b", endnote_data["date"])
                            if year_match:
                                pub.year = int(year_match.group())
                                logger.debug(f"Updated year to: {pub.year}")
                        except Exception as e:
                            logger.warning(f"Failed to extract year from date: {e}")

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
        self, session: aiohttp.ClientSession, pub: GrowthLabPublication
    ) -> GrowthLabPublication:
        """Enrich publication with data from Endnote file with retry mechanism"""
        if not pub.pub_url:
            return pub

        # Add delay between requests to avoid being rate-limited
        await asyncio.sleep(self.scrape_delay)

        endnote_url = await self.get_endnote_file_url(session, str(pub.pub_url))
        if not endnote_url:
            return pub

        # Add additional delay before downloading the endnote file
        await asyncio.sleep(self.scrape_delay / 2)

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

    async def extract_and_enrich_publications(self) -> list[GrowthLabPublication]:
        """Extract all publications and enrich them with Endnote data"""
        publications = await self.extract_publications()

        # Create more robust session with timeouts
        timeout = aiohttp.ClientTimeout(
            total=120,  # Increased from 60
            connect=30,  # Increased from 20
            sock_connect=30,  # Increased from 20
            sock_read=30,  # Increased from 20
        )
        connector = aiohttp.TCPConnector(
            limit=self.concurrency_limit,
            ttl_dns_cache=300,
            force_close=False,
            enable_cleanup_closed=True,
        )

        # Create a session with custom headers by default
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Cache-Control": "max-age=0",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
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

        # Generate a report on selector performance
        self.monitor.generate_report()

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
    ) -> list[GrowthLabPublication]:
        """
        Update publications by comparing existing ones with newly scraped ones

        This handles updates to existing publications by comparing content hashes

        Args:
            existing_path: Optional path to existing publications CSV
            output_path: Optional path to save updated publications
            storage: Optional storage instance (will use default if None)
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
