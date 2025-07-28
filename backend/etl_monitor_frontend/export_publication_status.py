"""
Export Publication Status to CSV

This script connects to the Publication Tracking API and exports all publication
status data to a CSV file.
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_OUTPUT_DIR = Path(__file__).parent


class PublicationStatusExporter:
    """Handles exporting publication status data from the API to CSV"""

    def __init__(self, api_base_url: str):
        self.api_base_url = api_base_url.rstrip("/")
        self.session = requests.Session()

    def check_api_health(self) -> bool:
        """Check if the API is running and accessible"""
        try:
            health_url = urljoin(self.api_base_url, "/api/health")
            response = self.session.get(health_url)
            response.raise_for_status()

            health_data = response.json()
            is_healthy = health_data.get("status") == "healthy"

            if is_healthy:
                logger.info("API health check passed")
            else:
                logger.error(f"API health check failed: {health_data}")

            return is_healthy

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to API at {self.api_base_url}: {e}")
            return False

    def fetch_all_publications(self) -> list[dict[str, Any]]:
        """Fetch all publication status records from the API using pagination"""
        publications = []
        page = 1
        page_size = 100

        logger.info("Starting to fetch publication data...")

        while True:
            try:
                params = {
                    "page": page,
                    "page_size": page_size,
                    "sort_by": "last_updated",
                    "sort_order": "desc",
                }

                api_url = urljoin(self.api_base_url, "/api/v1/publications/status")
                response = self.session.get(api_url, params=params)
                response.raise_for_status()

                data = response.json()
                page_publications = data.get("items", [])
                publications.extend(page_publications)

                total = data.get("total", 0)
                fetched = len(publications)
                logger.info(
                    f"Fetched page {page}: {len(page_publications)} records "
                    f"({fetched}/{total} total)"
                )

                if len(page_publications) < page_size or fetched >= total:
                    break

                page += 1

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to fetch page {page}: {e}")
                raise
            except (KeyError, ValueError) as e:
                logger.error(f"Invalid API response format on page {page}: {e}")
                raise

        logger.info(f"Successfully fetched {len(publications)} publication records")
        return publications

    def process_publication_data(
        self, publications: list[dict[str, Any]]
    ) -> pd.DataFrame:
        """Process publication data into a pandas DataFrame for CSV export"""
        logger.info("Processing publication data for CSV export...")

        df = pd.DataFrame(publications)

        if df.empty:
            logger.warning("No publication data to process")
            return df

        if "file_urls" in df.columns:
            df["file_urls"] = df["file_urls"].apply(
                lambda x: "; ".join(x) if isinstance(x, list) and x else ""
            )

        datetime_columns = [
            "discovery_timestamp",
            "last_updated",
            "download_timestamp",
            "processing_timestamp",
            "embedding_timestamp",
            "ingestion_timestamp",
        ]

        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

        column_order = [
            "publication_id",
            "title",
            "authors",
            "year",
            "download_status",
            "processing_status",
            "embedding_status",
            "ingestion_status",
            "last_updated",
            "discovery_timestamp",
            "download_timestamp",
            "processing_timestamp",
            "embedding_timestamp",
            "ingestion_timestamp",
            "download_attempt_count",
            "processing_attempt_count",
            "embedding_attempt_count",
            "ingestion_attempt_count",
            "source_url",
            "file_urls",
            "abstract",
            "error_message",
        ]

        available_columns = [col for col in column_order if col in df.columns]
        remaining_columns = [col for col in df.columns if col not in available_columns]
        final_column_order = available_columns + remaining_columns

        df = df[final_column_order]

        logger.info(f"Processed {len(df)} records with {len(df.columns)} columns")
        return df

    def generate_summary_stats(self, df: pd.DataFrame) -> dict[str, Any]:
        """Generate summary statistics about the publication status data"""
        if df.empty:
            return {"total_publications": 0}

        stats = {
            "total_publications": len(df),
            "export_timestamp": datetime.now().isoformat(),
        }

        status_columns = [
            "download_status",
            "processing_status",
            "embedding_status",
            "ingestion_status",
        ]

        for col in status_columns:
            if col in df.columns:
                status_counts = df[col].value_counts().to_dict()
                stats[f"{col}_distribution"] = status_counts

        if "year" in df.columns:
            year_counts = df["year"].value_counts().sort_index().to_dict()
            stats["year_distribution"] = year_counts

        if "last_updated" in df.columns:
            try:
                df["last_updated_dt"] = pd.to_datetime(df["last_updated"])
                recent_cutoff = datetime.now() - pd.Timedelta(days=7)
                recent_count = len(df[df["last_updated_dt"] > recent_cutoff])
                stats["recent_activity_7_days"] = recent_count
            except Exception:
                stats["recent_activity_7_days"] = "Unable to calculate"

        return stats

    def export_to_csv(self, df: pd.DataFrame, output_path: Path) -> None:
        """Export DataFrame to CSV file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
        logger.info(f"Data exported to: {output_path}")

        file_size = output_path.stat().st_size
        if file_size > 1024 * 1024:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        elif file_size > 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size} bytes"

        logger.info(f"File size: {size_str}")

    def save_summary_stats(self, stats: dict[str, Any], output_path: Path) -> None:
        """Save summary statistics to a JSON file"""
        stats_path = output_path.with_suffix(".json")

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)

        logger.info(f"Summary statistics saved to: {stats_path}")

    def export_publication_status(self, output_path: Path) -> bool:
        """Main export function - orchestrates the entire export process"""
        try:
            if not self.check_api_health():
                return False

            publications = self.fetch_all_publications()

            if not publications:
                logger.warning("No publications found to export")
                return True

            df = self.process_publication_data(publications)
            stats = self.generate_summary_stats(df)

            self.export_to_csv(df, output_path)
            self.save_summary_stats(stats, output_path)

            return True

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False


def main():
    """Main function to handle command line arguments and run the export"""
    parser = argparse.ArgumentParser(
        description="Export publication status data from API to CSV"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output CSV file path (default: auto-generated with timestamp)",
    )

    parser.add_argument(
        "--api-url",
        "-u",
        type=str,
        default=DEFAULT_API_URL,
        help=f"Base URL of the Publication Tracking API (default: {DEFAULT_API_URL})",
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Reduce output verbosity"
    )

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"publication_status_{timestamp}.csv"
        args.output = DEFAULT_OUTPUT_DIR / filename

    exporter = PublicationStatusExporter(args.api_url)
    success = exporter.export_publication_status(args.output)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
