import hashlib
import logging
import mimetypes
from pathlib import Path

import pandas as pd
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # make language detection deterministic

# Enable logging
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
    elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return ".docx"
    elif content_type and content_type.startswith("text/"):
        return ".txt"
    elif content_type and content_type.startswith("image/"):
        return "." + content_type.split("/")[-1]
    return ".bin"


def parse_file_urls(field):
    """Convert a field into a list"""
    if pd.isna(field) or field == "[]":
        return []
    if isinstance(field, str):
        try:
            parsed = eval(field)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except Exception:
            return [field]
    return list(field)


def detect_language(text: str) -> str:
    try:
        if isinstance(text, str) and len(text.strip()) > 20:
            return detect(text)
        else:
            return "unknown"
    except Exception:
        return "unknown"


def expand_publications_to_file_level(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv)

    # Parse file_urls into actual lists
    df["file_urls"] = df["file_urls"].apply(parse_file_urls)

    # Add num_files before exploding
    df["num_files"] = df["file_urls"].apply(len)

    # Add guessed language from abstract or title
    def guess_language(row):
        if pd.notna(row.get("abstract")):
            return detect_language(row["abstract"])
        elif pd.notna(row.get("title")):
            return detect_language(row["title"])
        else:
            return "unknown"

    df["language_guessed"] = df.apply(guess_language, axis=1)

    # Explode so that each row has one file_url
    df = df.explode("file_urls").rename(columns={"file_urls": "file_url"})

    # Mark missing file URLs
    df["missing_fileurl"] = df["file_url"].isna()

    # Clean file_url
    df["file_url"] = df["file_url"].astype(str).str.strip().replace("nan", None)

    # Generate extra file-level info
    def build_row(row):
        url = row["file_url"]
        pub_id = row["paper_id"]

        if not url or url == "None":
            return pd.Series({"file_id": None, "file_name": None, "file_path": None})

        url_path = url.split("?")[0].split("#")[0]
        file_name_candidate = url_path.split("/")[-1]

        if not file_name_candidate or "." not in file_name_candidate:
            short_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            ext = guess_extension(url)
            file_name = f"{short_hash}{ext}"
        else:
            file_name = file_name_candidate

        file_path = f"raw/documents/growthlab/{pub_id}/{file_name}"
        file_id = f"{pub_id}_{hashlib.md5(url.encode()).hexdigest()[:8]}"

        return pd.Series(
            {"file_id": file_id, "file_name": file_name, "file_path": file_path}
        )

    file_metadata = df.apply(build_row, axis=1)
    df = pd.concat([df, file_metadata], axis=1)

    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"✅ Saved file-level dataset to: {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Explode GrowthLab publications to file-level data."
    )
    parser.add_argument("--input", required=True, help="Path to publication-level CSV")
    parser.add_argument("--output", required=True, help="Path to output file-level CSV")
    args = parser.parse_args()

    expand_publications_to_file_level(
        input_csv=Path(args.input),
        output_csv=Path(args.output)
    )