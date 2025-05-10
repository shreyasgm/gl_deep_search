import hashlib
import logging
import mimetypes
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, detect

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


def parse_file_urls(field):
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


def infer_language_from_filename(file_name: str) -> str:
    if not file_name:
        return "unknown"
    name = file_name.lower()

    lang_map = {
        "en": ["english", "_en", "-en", "report", "policy", "summary"],
        "es": [
            "spanish",
            "_es",
            "-es",
            "esp",
            "informe",
            "resumen",
            "crecimiento",
            "complejidad",
            "diagnostico",
        ],
        "fr": [
            "french",
            "_fr.",
            "-fr.",
            "fr.",
            "execsumfr",
            "-français",
            "-francais",
        ],  # fixed here issue with franchsing being detected as french
        "ar": ["arabic", "-ar", "_ar", "execsumar"],
        "pt": ["portuguese", "_pt", "-pt"],
    }

    for lang, hints in lang_map.items():
        if any(hint in name for hint in hints):
            return lang
    return "unknown"


def guess_lang_from_text(text: str) -> str:
    try:
        if isinstance(text, str) and len(text.strip()) > 20:
            return detect(text)
        return "unknown"
    except Exception:
        return "unknown"


def detect_language(row):
    # Step 1: Try from filename pattern
    file_guess = infer_language_from_filename(row.get("file_name", ""))
    if file_guess != "unknown":
        return file_guess, "filename"

    # Step 2: Try from abstract (most reliable free text)
    abstract = row.get("abstract", "")
    lang_from_abstract = guess_lang_from_text(abstract)
    if lang_from_abstract != "unknown":
        return lang_from_abstract, "abstract"

    # Step 3: Fallback to title
    title = row.get("title", "")
    lang_from_title = guess_lang_from_text(title)
    if lang_from_title != "unknown":
        return lang_from_title, "title"

    # Step 4: If all fail
    return "unknown", "unknown"


def expand_publications_to_file_level(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv)

    df["file_urls"] = df["file_urls"].apply(parse_file_urls)
    df["num_files"] = df["file_urls"].apply(len)

    df = df.explode("file_urls").rename(columns={"file_urls": "file_url"})
    df["missing_fileurl"] = df["file_url"].isna()
    df["file_url"] = df["file_url"].astype(str).str.strip()
    df["file_url"] = df["file_url"].replace(["", "None", "nan"], None)

    def build_row(row):
        url = row["file_url"]
        pub_id = row["paper_id"]

        if not url or str(url).strip().lower() in {"none", "nan", ""}:
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

    # Detect language final pass
    df[["language", "language_source"]] = df.apply(
        detect_language, axis=1, result_type="expand"
    )

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
        input_csv=Path(args.input), output_csv=Path(args.output)
    )
