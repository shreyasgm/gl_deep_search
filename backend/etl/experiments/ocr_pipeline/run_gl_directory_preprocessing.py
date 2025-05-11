import hashlib
import logging
import mimetypes
from pathlib import Path

import pandas as pd
import polars as pl
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
    if field is None or field == "" or field == "[]":
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
    name = str(file_name).lower()

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
            "-francais",
            "-français",
        ],
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


def detect_language(file_name, abstract, title):
    # Step 1: Try from filename pattern
    file_guess = infer_language_from_filename(file_name)
    if file_guess != "unknown":
        return file_guess, "filename"

    # Step 2: Try from abstract (most reliable free text)
    lang_from_abstract = guess_lang_from_text(abstract)
    if lang_from_abstract != "unknown":
        return lang_from_abstract, "abstract"

    # Step 3: Fallback to title
    lang_from_title = guess_lang_from_text(title)
    if lang_from_title != "unknown":
        return lang_from_title, "title"

    # Step 4: If all fail
    return "unknown", "unknown"


def add_ocr_flags(df):
    """
    Add to_ocr and sample flags to the dataframe.

    Args:
        df: A polars DataFrame with columns 'missing_fileurl' and 'is_main_document'

    Returns:
        pandas.DataFrame with additional 'to_ocr' and 'sample' columns
    """
    import numpy as np

    # Convert to pandas for easier processing
    temp_df = df.to_pandas()

    # Add to_ocr flag - documents that have a file and are main documents
    temp_df["to_ocr"] = (~temp_df["missing_fileurl"]) & temp_df["is_main_document"]

    # Set random seed for reproducibility
    np.random.seed(42)

    # Sample approximately 10% of the to_ocr documents
    temp_df["sample"] = False
    to_ocr_indices = temp_df[temp_df["to_ocr"]].index

    if len(to_ocr_indices) > 0:
        sample_size = max(1, int(len(to_ocr_indices) * 0.1))  # 1% sample
        sampled_indices = np.random.choice(
            to_ocr_indices, size=sample_size, replace=False
        )
        temp_df.loc[sampled_indices, "sample"] = True

    return temp_df


def generate_file_metadata(file_url: str, paper_id: str) -> dict:
    if not file_url or str(file_url).strip().lower() in {"none", "nan", ""}:
        return {
            "file_id": f"{paper_id}_nofile",
            "file_name": None,
            "file_path": None,
        }

    hash_prefix = hashlib.md5(file_url.encode()).hexdigest()[:8]
    clean_url = file_url.split("?")[0].split("#")[0]
    file_name_fragment = clean_url.split("/")[-1]

    has_extension = "." in file_name_fragment
    if has_extension:
        file_name = file_name_fragment
    else:
        ext = guess_extension(file_url)
        file_name = f"{hash_prefix}{ext}"

    file_path = f"raw/documents/growthlab/{paper_id}/{file_name}"

    return {
        "file_id": f"{paper_id}_{hash_prefix}",
        "file_name": file_name,
        "file_path": file_path,
    }


def expand_publications_to_file_level(input_csv: Path, output_csv: Path) -> None:
    # Read the data with pandas first (for compatibility with parse_file_urls)
    df_pd = pd.read_csv(input_csv, engine="python")
    df_pd["file_urls"] = df_pd["file_urls"].apply(parse_file_urls)
    df_pd["num_files"] = df_pd["file_urls"].apply(len)

    # Explode file_urls
    df_pd = df_pd.explode("file_urls").rename(columns={"file_urls": "file_url"})

    # Convert to polars
    df = pl.from_pandas(df_pd)

    # Process file URLs
    df = df.with_columns(
        [
            pl.col("file_url").str.strip_chars(),
            pl.col("file_url").is_null().alias("missing_fileurl"),
        ]
    )

    df = df.with_columns(
        [
            pl.when(pl.col("file_url").is_in(["", "None", "nan"]))
            .then(None)
            .otherwise(pl.col("file_url"))
            .alias("file_url")
        ]
    )

    # Generate file metadata using a helper for readability and maintainability
    df = df.with_columns(
        pl.struct(["file_url", "paper_id"])
        .map_elements(
            lambda x: generate_file_metadata(x["file_url"], x["paper_id"]),
            return_dtype=pl.Struct,
        )
        .alias("file_metadata")
    )

    # Unpack the struct column
    df = df.with_columns(
        [
            pl.col("file_metadata").struct.field("file_id").alias("file_id"),
            pl.col("file_metadata").struct.field("file_name").alias("file_name"),
            pl.col("file_metadata").struct.field("file_path").alias("file_path"),
        ]
    )

    # Drop the temporary column
    df = df.drop("file_metadata")

    # Detect language
    df = df.with_columns(
        [
            pl.struct(["file_name", "abstract", "title"])
            .map_elements(
                lambda x: detect_language(x["file_name"], x["abstract"], x["title"]),
                return_dtype=pl.List,
            )
            .alias("language_info")
        ]
    )

    # Extract language info
    df = df.with_columns(
        [
            pl.col("language_info").list.get(0).alias("language"),
            pl.col("language_info").list.get(1).alias("language_source"),
        ]
    )

    # Drop temporary column
    df = df.drop("language_info")

    # Initialize is_main_document column
    df = df.with_columns(pl.lit(False).alias("is_main_document"))

    # Now process by paper_id to select main document
    paper_ids = df.select("paper_id").unique().to_series().to_list()

    # Prepare a list for results
    result_rows = []

    for paper_id in paper_ids:
        paper_df = df.filter(pl.col("paper_id") == paper_id)

        # Initialize selection logic
        selected_row = None

        # First, identify summary/excerpt/brief documents
        is_summary_pattern = (
            "execsum|executive_summary|summary|brief|appendix|toc|excerpt|"
            "policy_brief|policy-brief"
        )

        # Step 1: Look for non-summary documents in English
        non_summary_english = paper_df.filter(
            (pl.col("language") == "en")
            & ~pl.col("file_name").str.contains(is_summary_pattern)
        )

        if non_summary_english.height > 0:
            # If we have non-summary English docs, prefer working papers
            wp_docs = non_summary_english.filter(
                pl.col("file_name").str.contains("wp|working_paper|cidwp")
            )
            if wp_docs.height > 0:
                # Prefer working papers with filename-based language detection
                filename_wp = wp_docs.filter(pl.col("language_source") == "filename")
                selected_row = (
                    filename_wp.row(0) if filename_wp.height > 0 else wp_docs.row(0)
                )
            else:
                # No working papers, prefer filename-based among non-summary English
                filename_non_summary = non_summary_english.filter(
                    pl.col("language_source") == "filename"
                )
                selected_row = (
                    filename_non_summary.row(0)
                    if filename_non_summary.height > 0
                    else non_summary_english.row(0)
                )

        # Step 2: If no non-summary English docs, look for any non-summary document
        elif (
            paper_df.filter(
                ~pl.col("file_name").str.contains(is_summary_pattern)
            ).height
            > 0
        ):
            non_summary_any = paper_df.filter(
                ~pl.col("file_name").str.contains(is_summary_pattern)
            )
            # Prefer English among any non-summary
            any_non_summary_en = non_summary_any.filter(pl.col("language") == "en")
            if any_non_summary_en.height > 0:
                selected_row = any_non_summary_en.row(0)
            else:
                # No English, prefer filename-based detection among non-summary
                filename_non_summary = non_summary_any.filter(
                    pl.col("language_source") == "filename"
                )
                selected_row = (
                    filename_non_summary.row(0)
                    if filename_non_summary.height > 0
                    else non_summary_any.row(0)
                )

        # Step 3: If only summary documents exist, prefer English summaries
        elif paper_df.filter(pl.col("language") == "en").height > 0:
            english_any = paper_df.filter(pl.col("language") == "en")
            # Prefer filename-based among any English
            filename_eng = english_any.filter(pl.col("language_source") == "filename")
            selected_row = (
                filename_eng.row(0) if filename_eng.height > 0 else english_any.row(0)
            )

        # Step 4: Last resort: take any document with filename-based lang
        elif paper_df.filter(pl.col("language_source") == "filename").height > 0:
            selected_row = paper_df.filter(pl.col("language_source") == "filename").row(
                0
            )

        # Step 5: Otherwise take the first document
        else:
            selected_row = paper_df.row(0)

        # Extract all fields from the selected row to build our new dataframe
        paper_df_rows = paper_df.to_dicts()
        for row in paper_df_rows:
            # Mark the selected document as main
            if (
                row["paper_id"] == paper_id
                and row["file_id"] == selected_row[paper_df.columns.index("file_id")]
            ):
                row["is_main_document"] = True
            else:
                row["is_main_document"] = False
            result_rows.append(row)

    # Convert result_rows list back to a dataframe and write to output file
    result_df = pl.DataFrame(result_rows)

    # Add OCR flags and convert to pandas
    result_pd_df = add_ocr_flags(result_df)

    # Write the final result to CSV
    result_pd_df.to_csv(output_csv, index=False)


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
