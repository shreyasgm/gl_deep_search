"""
Publication models for Growth Lab Deep Search.

This module contains the data models for publications from various sources,
ensuring consistent structure and behavior across the application.
"""

import hashlib
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator
from sqlmodel import MetaData, SQLModel

# Create metadata instance
metadata = MetaData()


# Create Base class
class Base(SQLModel):
    """Base class for all models"""

    metadata = metadata


class GrowthLabPublication(BaseModel):
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

        # Prefer URL-based ID for stability if available
        if self.pub_url:
            # Extract the publication slug from the URL
            url_path = str(self.pub_url).lower()
            # Remove the domain and get the path
            # Handle both /publication/ (singular) and /publications/ (plural) patterns
            slug = None
            if "/publications/" in url_path:
                slug = url_path.split("/publications/")[-1]
            elif "/publication/" in url_path:
                slug = url_path.split("/publication/")[-1]

            if slug:
                # Remove any query parameters or fragments
                slug = slug.split("?")[0].split("#")[0]
                # Remove trailing slash if present
                slug = slug.rstrip("/")
                if slug:  # If we got a valid slug
                    return f"gl_url_{hashlib.sha256(slug.encode()).hexdigest()[:16]}"

        # Create normalized base string for hashing
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
            return f"gl_unknown_{hashlib.sha256(random_id.encode()).hexdigest()[:16]}"

        # Create hash using SHA-256 for better collision resistance
        hash_id = hashlib.sha256(base.encode()).hexdigest()[:16]

        # Format: source_year_hash
        return f"gl_{self.year or '0000'}_{hash_id}"

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
