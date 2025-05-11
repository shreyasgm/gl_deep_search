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


def expand_publications_to_file_level(input_csv: Path, output_csv: Path) -> None:
    # Read the data with pandas first (for compatibility with parse_file_urls)
    df_pd = pd.read_csv(input_csv, engine='python')
    df_pd["file_urls"] = df_pd["file_urls"].apply(parse_file_urls)
    df_pd["num_files"] = df_pd["file_urls"].apply(len)
    
    # Explode file_urls
    df_pd = df_pd.explode("file_urls").rename(columns={"file_urls": "file_url"})
    
    # Convert to polars
    df = pl.from_pandas(df_pd)
    
    # Process file URLs
    df = df.with_columns([
        pl.col("file_url").str.strip_chars(),
        pl.col("file_url").is_null().alias("missing_fileurl")
    ])
    
    df = df.with_columns([
        pl.when(pl.col("file_url").is_in(["", "None", "nan"]))
            .then(None)
            .otherwise(pl.col("file_url"))
            .alias("file_url")
    ])
    
    # Generate file metadata
    df = df.with_columns([
        pl.struct(["file_url", "paper_id"]).map_elements(
            lambda x: {
                "file_id": f"{x['paper_id']}_nofile" if not x['file_url'] or str(x['file_url']).strip().lower() in {"none", "nan", ""} 
                    else f"{x['paper_id']}_{hashlib.md5(str(x['file_url']).encode()).hexdigest()[:8]}",
                
                "file_name": None if not x['file_url'] or str(x['file_url']).strip().lower() in {"none", "nan", ""} 
                    else (x['file_url'].split("?")[0].split("#")[0].split("/")[-1] 
                          if "." in x['file_url'].split("?")[0].split("#")[0].split("/")[-1] 
                          else f"{hashlib.md5(str(x['file_url']).encode()).hexdigest()[:8]}{guess_extension(str(x['file_url']))}"),
                
                "file_path": None if not x['file_url'] or str(x['file_url']).strip().lower() in {"none", "nan", ""} 
                    else f"raw/documents/growthlab/{x['paper_id']}/{x['file_url'].split('?')[0].split('#')[0].split('/')[-1]}" 
                            if "." in x['file_url'].split("?")[0].split("#")[0].split("/")[-1]
                            else f"raw/documents/growthlab/{x['paper_id']}/{hashlib.md5(str(x['file_url']).encode()).hexdigest()[:8]}{guess_extension(str(x['file_url']))}"
            }
        ).alias("file_metadata")
    ])
    
    # Unpack the struct column
    df = df.with_columns([
        pl.col("file_metadata").struct.field("file_id").alias("file_id"),
        pl.col("file_metadata").struct.field("file_name").alias("file_name"),
        pl.col("file_metadata").struct.field("file_path").alias("file_path")
    ])
    
    # Drop the temporary column
    df = df.drop("file_metadata")
    
    # Detect language
    df = df.with_columns([
        pl.struct(["file_name", "abstract", "title"]).map_elements(
            lambda x: detect_language(x["file_name"], x["abstract"], x["title"])
        ).alias("language_info")
    ])
    
    # Extract language info
    df = df.with_columns([
        pl.col("language_info").list.get(0).alias("language"),
        pl.col("language_info").list.get(1).alias("language_source")
    ])
    
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
        
        # STEP 1: Look for English full papers (not summaries/briefs/appendices)
        english_main = paper_df.filter(
            (pl.col("language") == "en") &
            ~pl.col("file_name").str.contains("execsum|executive_summary|summary|brief|appendix")
        )
        
        if english_main.height > 0:
            # If multiple English main documents, prefer working papers
            wp_docs = english_main.filter(
                pl.col("file_name").str.contains("wp|working_paper|cidwp")
            )
            
            if wp_docs.height > 0:
                selected_row = wp_docs.row(0)
            else:
                selected_row = english_main.row(0)
        
        # STEP 2: If no English main docs, try English summaries
        elif paper_df.filter(pl.col("language") == "en").height > 0:
            selected_row = paper_df.filter(pl.col("language") == "en").row(0)
        
        # STEP 3: Otherwise, take the first document of any language
        else:
            selected_row = paper_df.row(0)
        
        # Extract all fields from the selected row to build our new dataframe
        paper_df_rows = paper_df.to_dicts()
        for row in paper_df_rows:
            # Mark the selected document as main
            if (row["paper_id"] == paper_id and 
                row["file_id"] == selected_row[paper_df.columns.index("file_id")]):
                row["is_main_document"] = True
            else:
                row["is_main_document"] = False
            result_rows.append(row)
    
    # Create the final dataframe from our processed rows
    final_df = pl.from_dicts(result_rows)
    
    # Verify we have exactly one main document per paper
    main_doc_counts = final_df.group_by("paper_id").agg(
        pl.col("is_main_document").sum().alias("main_count")
    )
    
    min_count = main_doc_counts["main_count"].min()
    max_count = main_doc_counts["main_count"].max()
    
    print(f"Min main docs per paper: {min_count}")
    print(f"Max main docs per paper: {max_count}")
    
    if min_count != 1 or max_count != 1:
        raise AssertionError("Error: Not exactly one main document per paper!")
    
    # Convert back to pandas for saving with to_csv
    final_df.to_pandas().to_csv(output_csv, index=False)
    print(f"Successfully wrote {final_df.height} rows to {output_csv}")


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