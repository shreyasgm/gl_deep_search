import hashlib
import logging
import mimetypes
from pathlib import Path

import pandas as pd

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def guess_extension(file_url: str) -> str:
    content_type = mimetypes.guess_type(file_url)[0]
    if content_type == "application/pdf":
        return ".pdf"
    elif content_type == "application/msword":
        return ".doc"
    elif (
        content_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        return ".docx"
    elif content_type and content_type.startswith("text/"):
        return ".txt"
    elif content_type and content_type.startswith("image/"):
        return "." + content_type.split("/")[-1]
    return ".bin"


def normalize_file_urls(file_urls_field):
    """Parses file_urls if it's a list-string; otherwise ensures it's a list"""
    if not file_urls_field or file_urls_field == "[]":
        return []
    if isinstance(file_urls_field, str):
        try:
            return (
                eval(file_urls_field)
                if file_urls_field.startswith("[")
                else [file_urls_field]
            )
        except Exception:
            return [file_urls_field]
    return list(file_urls_field)


def expand_publications_to_file_level(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv)
    output_rows = []

    for _, row in df.iterrows():
        pub_id = row.get("paper_id")
        pub_title = row.get("title")
        pub_url = row.get("pub_url")
        file_urls = normalize_file_urls(row.get("file_urls"))

        if not file_urls:
            output_rows.append(
                {
                    "publication_id": pub_id,
                    "publication_title": pub_title,
                    "pub_url": pub_url,
                    "file_url": None,
                    "file_id": None,
                    "file_name": None,
                    "file_path": None,
                    "missing_fileurl": True,
                }
            )
            continue

        for file_url in file_urls:
            file_url = str(file_url).strip()
            url_path = file_url.split("?")[0].split("#")[0]
            file_name_candidate = url_path.split("/")[-1]

            if not file_name_candidate or "." not in file_name_candidate:
                # No extension, guess content type
                url_hash = hashlib.md5(file_url.encode()).hexdigest()[:8]
                file_name = f"{url_hash}{guess_extension(file_url)}"
            else:
                file_name = file_name_candidate

            file_path = f"raw/documents/growthlab/{pub_id}/{file_name}"
            file_id = f"{pub_id}_{hashlib.md5(file_url.encode()).hexdigest()[:8]}"

            output_rows.append(
                {
                    "publication_id": pub_id,
                    "publication_title": pub_title,
                    "pub_url": pub_url,
                    "file_url": file_url,
                    "file_id": file_id,
                    "file_name": file_name,
                    "file_path": file_path,
                    "missing_fileurl": False,
                }
            )

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_csv, index=False)
    logger.info(f"✅ Saved file-level dataset to: {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Expand to file-level.")
    parser.add_argument("--input", required=True, help="Path to publication-level CSV")
    parser.add_argument("--output", required=True, help="Path to output file-level CSV")
    args = parser.parse_args()

    expand_publications_to_file_level(
        input_csv=Path(args.input), output_csv=Path(args.output)
    )
