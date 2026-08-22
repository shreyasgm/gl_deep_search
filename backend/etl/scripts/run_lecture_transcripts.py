"""
Creates trascript files from the DEV309 - Development Policy Strategy class
taught by Ricardo Hausmann. The script uses OpenAI API to extract structured
information from the lecture transcripts and saves them as JSON files.
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from backend.etl.utils.atomic_io import atomic_write

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Dated model snapshot: an unpinned alias silently changes the cleaned text
# (and therefore every downstream chunk and embedding) between runs.
OPENAI_MODEL = "gpt-5.6-luna"

# A cleaned transcript below this fraction of the raw length means the model
# summarised instead of cleaning. Filler removal on a lecture transcript
# typically retains well over half the characters.
MIN_CLEANED_RATIO = 0.5

# Raw transcript tokens handed to the model per call. A lecture runs roughly
# 14-16k tokens, so this is 2-3 calls each rather than one call that the model
# answers by summarising.
#
# Measured on 18_fiscal_policy (53k chars), retention and detail preservation:
#   whole transcript, gpt-4.1-nano  -> 22% length, 22% of numbers kept
#   6000-token segments, this model -> 84% length, 83% of numbers kept, 2 calls
#   3000-token segments, this model -> 83% length, 83% of numbers kept, 4 calls
# Halving the segment bought nothing but doubled the call count, so 6000 it is.
CLEANING_SEGMENT_TOKENS = 6000

# Tail of the previously cleaned segment, passed as read-only context so the
# next segment continues mid-thought instead of restarting. Overlapping the
# raw text instead would duplicate content at every seam.
CLEANING_CONTEXT_CHARS = 1200

# Best-effort determinism. The model does not accept temperature=0.
CLEANING_SEED = 20260822


class TranscriptFidelityError(RuntimeError):
    """Raised when transcript cleaning loses too much of the source text."""


# Characters that are not safe in an artifact filename
_SLUG_UNSAFE_PATTERN = re.compile(r"[^A-Za-z0-9]+")


# Define Pydantic model for lecture transcript data
class LectureTranscript(BaseModel):
    """Model representing structured information from a lecture transcript"""

    lecture_number: int
    title: str
    main_topics: list[str]
    summary: str
    transcript: str


_CLEANING_INSTRUCTIONS = """
You are producing a faithful, readable transcript of a lecture from the
'Development Policy Strategy' class taught by Ricardo Hausmann, Director of
Harvard's Growth Lab.

You will be given ONE SEGMENT of a longer raw transcript. Segments arrive in
order. Rewrite the segment you are given as clean lecture prose.

This is a REFORMATTING task, not a summarization task:
- Preserve every substantive point, argument, example, figure, country, and
  name. Nothing may be dropped for brevity.
- Your output should be comparable in length to the segment you were given.
  If it is dramatically shorter, you have summarised, which is wrong.
- Remove only disfluencies: filler words, false starts, stutters, verbatim
  self-repetition, and obvious transcription artifacts.
- Keep the speaker's first-person voice and the order ideas are presented in.
  Do not reorder, editorialise, or add analysis of your own.
- Include only the main speaker. Drop audience questions, but keep the
  speaker's answer and phrase it so it stands on its own.
- Fix clear transcription errors, especially misheard technical terms, place
  names, and people's names.
- Organise into paragraphs. Add no headings, bullet lists, titles, segment
  markers, or commentary.

If a CONTEXT block is supplied, it is the end of the previously cleaned
segment. It exists only so your output continues seamlessly from it. Do not
repeat it, summarise it, or refer to it. Begin exactly where it leaves off,
even if that is mid-argument.

Return only the cleaned prose for the current segment.
"""


def _split_into_segments(text: str, max_tokens: int) -> list[str]:
    """Split a transcript into segments of at most ``max_tokens`` tokens.

    Splits on paragraph breaks where possible and sentence boundaries
    otherwise, so a segment rarely starts mid-sentence.

    Args:
        text: Raw transcript text.
        max_tokens: Maximum tokens per segment.

    Returns:
        Ordered list of segments. A short transcript yields a single segment.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    if len(encoding.encode(text)) <= max_tokens:
        return [text]

    # Prefer paragraph boundaries; fall back to sentence boundaries for any
    # paragraph that is itself too long to fit in a segment.
    blocks: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(encoding.encode(para)) <= max_tokens:
            blocks.append(para)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf = ""
            for sentence in sentences:
                candidate = f"{buf} {sentence}".strip()
                if buf and len(encoding.encode(candidate)) > max_tokens:
                    blocks.append(buf)
                    buf = sentence
                else:
                    buf = candidate
            if buf:
                blocks.append(buf)

    segments: list[str] = []
    current = ""
    for block in blocks:
        candidate = f"{current}\n\n{block}".strip()
        if current and len(encoding.encode(candidate)) > max_tokens:
            segments.append(current)
            current = block
        else:
            current = candidate
    if current:
        segments.append(current)
    return segments


def _clean_segment(segment: str, index: int, total: int, context: str | None) -> str:
    """Clean a single transcript segment.

    Args:
        segment: Raw text of this segment.
        index: Zero-based position of this segment.
        total: Total number of segments.
        context: Tail of the previously cleaned segment, or None for the first.

    Returns:
        Cleaned prose for this segment.

    Raises:
        TranscriptFidelityError: If the model truncated its own output.
    """
    parts: list[str] = []
    if total > 1:
        parts.append(f"This is segment {index + 1} of {total}.")
    if context:
        parts.append(
            "CONTEXT (end of the previous cleaned segment - do not repeat "
            f"any of it):\n{context}"
        )
    parts.append(f"SEGMENT TO CLEAN:\n{segment}")

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _CLEANING_INSTRUCTIONS},
            {"role": "user", "content": "\n\n".join(parts)},
        ],
        # gpt-5.6-luna rejects temperature=0 ("only the default (1) is
        # supported"), so pin a seed instead. OpenAI treats seed as
        # best-effort rather than a guarantee, but cleaned transcripts are
        # cached on disk and reused, so a corpus stays stable once built.
        seed=CLEANING_SEED,
    )

    choice = response.choices[0]
    if getattr(choice, "finish_reason", None) == "length":
        raise TranscriptFidelityError(
            f"Segment {index + 1}/{total} hit the model output limit "
            f"(finish_reason='length'); reduce CLEANING_SEGMENT_TOKENS"
        )
    return (choice.message.content or "").strip()


def clean_transcript(transcript_text: str) -> str:
    """Clean a raw lecture transcript, segment by segment.

    A whole 60KB transcript handed to the model in one call comes back
    summarised - on 2026-08-22 every lecture returned at roughly 18% of its
    raw length, so the index held summaries rather than lectures. Splitting
    the transcript keeps each call small enough that the model reformats
    instead of compressing, and passing the tail of the previous cleaned
    segment keeps the seams continuous without duplicating text.

    Args:
        transcript_text: The raw transcript text to be processed.

    Returns:
        A cleaned and reformatted version of the transcript.

    Raises:
        TranscriptFidelityError: If the output is truncated or so much
            shorter than the input that content was clearly dropped.
    """
    segments = _split_into_segments(transcript_text, CLEANING_SEGMENT_TOKENS)
    logger.info(
        f"Cleaning transcript in {len(segments)} segment(s) "
        f"({len(transcript_text)} chars)"
    )

    cleaned_parts: list[str] = []
    try:
        for index, segment in enumerate(segments):
            context = (
                cleaned_parts[-1][-CLEANING_CONTEXT_CHARS:] if cleaned_parts else None
            )
            cleaned_parts.append(_clean_segment(segment, index, len(segments), context))
    except TranscriptFidelityError:
        raise
    except Exception as e:
        logger.error(f"Error cleaning transcript: {str(e)}")
        raise

    cleaned = "\n\n".join(part for part in cleaned_parts if part)

    ratio = len(cleaned) / len(transcript_text) if transcript_text else 1.0
    if ratio < MIN_CLEANED_RATIO:
        raise TranscriptFidelityError(
            f"Cleaned transcript is {ratio:.0%} of the raw length "
            f"({len(cleaned)} vs {len(transcript_text)} chars), below the "
            f"{MIN_CLEANED_RATIO:.0%} floor. The model summarised rather than "
            f"reformatted. Lower CLEANING_SEGMENT_TOKENS or revisit the prompt."
        )
    logger.info(f"Cleaned transcript retains {ratio:.0%} of raw length")
    return cleaned


def extract_lecture_metadata(
    clean_transcript: str, lecture_number: int
) -> LectureTranscript:
    """
    Extract structured metadata from a cleaned lecture transcript
    using OpenAI API and Pydantic.

    Args:
        clean_transcript (str): The already cleaned transcript text.
        lecture_number (int): The lecture number for identification.

    Returns:
        LectureTranscript: A Pydantic model instance.
    """

    # Define paramenters to pass to messages
    instructions = """
    You are an expert assistant analyzing lecture transcripts from the
    'Development Policy Strategy' class taught by Ricardo Hausmann
    (Director of Harvard's Growth Lab). Extract the essential information
    from the transcript.
    """
    prompt = f"""
    Here is a cleaned lecture transcript #{lecture_number} from the
    Development Policy Strategy class. Please extract the key information:
    :\n\n{clean_transcript}
    """

    try:
        # Using the parse method with the Pydantic model
        completion = client.beta.chat.completions.parse(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": instructions,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format=LectureTranscript,
            seed=CLEANING_SEED,
        )

        # Get the parsed data directly as a LectureTranscript object
        lecture_data = completion.choices[0].message.parsed

        # Ensure lecture number is set
        if lecture_data.lecture_number != lecture_number:
            lecture_data.lecture_number = lecture_number

        return lecture_data

    except Exception as e:
        logger.error(
            f"Error extracting metadata for lecture {lecture_number}: {str(e)}"
        )
        raise


def derive_lecture_identifiers(stem: str) -> tuple[str, int]:
    """Derive the artifact identifier and lecture number from a filename stem.

    Artifacts are named ``lecture_{slug}_cleaned.txt`` and
    ``lecture_{slug}_processed.json``. The slug must be unique per raw
    transcript: mashing every digit of the filename together used to map
    ``01_lecture`` and ``1_lecture`` onto the same artifacts, and every
    digit-less filename onto ``lecture_00_*``, so all but one transcript were
    silently reported as processed without anything being written.

    Stems that begin with a plain, unpadded lecture number keep the legacy
    zero-padded slug (``0_intro`` -> ``00``) so the transcripts already
    cleaned on disk are reused rather than re-cleaned. Every other stem falls
    back to a sanitized copy of itself, which is unique by construction.

    Args:
        stem: Filename stem of the raw transcript, e.g. ``"7_growth_cities"``.

    Returns:
        Tuple of (artifact slug, lecture number). The lecture number is 0 when
        the filename carries none; it is metadata only and never a filename.
    """
    prefix = stem.split("_", 1)[0]
    try:
        lecture_number = int(prefix)
    except ValueError:
        logger.warning(
            f"Could not extract a lecture number from filename '{stem}'. "
            f"Using the filename itself as the artifact identifier."
        )
        return _slugify(stem), 0

    if prefix != str(lecture_number):
        # Zero-padded or otherwise non-canonical ("01", "007"): folding it
        # onto the legacy name would collide with the canonical spelling.
        return _slugify(stem), lecture_number

    return f"{lecture_number:02d}", lecture_number


def _slugify(stem: str) -> str:
    """Convert a filename stem into a filename-safe artifact slug.

    Args:
        stem: Filename stem of the raw transcript.

    Returns:
        The stem with runs of non-alphanumeric characters replaced by
        underscores. Never returns digits only, so a slug can never collide
        with the zero-padded numeric identifiers.
    """
    slug = _SLUG_UNSAFE_PATTERN.sub("_", stem).strip("_")
    if not slug:
        return "unnamed"
    return f"n{slug}" if slug.isdigit() else slug


def process_single_transcript(
    file_path: Path,
    output_dir: str,
    intermediate_dir: str | None = None,
    max_tokens: int | None = None,
) -> bool:
    """
    Process a single transcript file and save the structured result to output directory.
    Also optionally saves the cleaned transcript to an intermediate directory.

    Args:
        file_path (Path): Path to the raw transcript file.
        output_dir (str): Directory to save processed transcript file.
        intermediate_dir (str): Directory to save cleaned transcripts.
        max_tokens (int | None): Limit transcript to first N tokens (for testing).

    """
    try:
        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)

        # Derive a per-transcript identifier for the artifact filenames
        lecture_slug, lecture_num = derive_lecture_identifiers(file_path.stem)

        # Check if the output file already exists
        output_file = Path(output_dir) / f"lecture_{lecture_slug}_processed.json"
        if output_file.exists():
            logger.info(
                f"Output file already exists: {output_file}. Skipping processing."
            )
            return True

        logger.info(f"Processing file: {file_path.name} (Lecture #{lecture_num})")

        # Handle intermediate directory for cleaned transcripts
        cleaned_transcript = None
        if intermediate_dir:
            os.makedirs(intermediate_dir, exist_ok=True)
            clean_file_path = (
                Path(intermediate_dir) / f"lecture_{lecture_slug}_cleaned.txt"
            )

            # Check if cleaned transcript already exists
            if clean_file_path.exists():
                logger.info(f"Found existing cleaned transcript: {clean_file_path}")
                with open(clean_file_path, encoding="utf-8") as f:
                    cleaned_transcript = f.read()

        # If no cleaned transcript exists yet, generate one
        if cleaned_transcript is None:
            # Read raw transcript content
            with open(file_path, encoding="utf-8") as f:
                transcript_text = f.read()

            # Apply max_tokens limit if specified
            if max_tokens and max_tokens > 0:
                # Use tiktoken for accurate token counting
                encoding = tiktoken.encoding_for_model("gpt-4")
                tokens = encoding.encode(transcript_text)
                if len(tokens) > max_tokens:
                    logger.info(
                        f"Limiting transcript from {len(tokens)} to {max_tokens} tokens"
                    )
                    transcript_text = encoding.decode(tokens[:max_tokens])

            logger.info("Cleaning transcript...")
            cleaned_transcript = clean_transcript(transcript_text)

            # Save cleaned transcript if intermediate directory is provided
            if intermediate_dir:
                clean_file_path = (
                    Path(intermediate_dir) / f"lecture_{lecture_slug}_cleaned.txt"
                )
                # Atomic: the resume check above reuses any existing cleaned
                # transcript, so a truncated one would be reused forever.
                atomic_write(clean_file_path, cleaned_transcript)
                logger.info(f"Saved cleaned transcript to: {clean_file_path}")

        # Extract metadata using the cleaned transcript
        logger.info("Extracting metadata...")
        structured_data = extract_lecture_metadata(cleaned_transcript, lecture_num)

        # Save as JSON
        output_file = Path(output_dir) / f"lecture_{lecture_slug}_processed.json"
        # Convert Pydantic model to dict first, then serialize the dict to JSON
        model_dict = structured_data.model_dump()
        atomic_write(output_file, json.dumps(model_dict, indent=2, ensure_ascii=False))

        logger.info(f"Successfully processed and saved: {output_file}")

        return True

    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {str(e)}")
        return False


def get_project_root():
    """Get the project root directory to allow running from any location."""
    script_path = Path(__file__).resolve()
    return str(script_path.parent.parent.parent.parent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process lecture transcripts using OpenAI API."
    )

    # Get project root for default paths
    project_root = get_project_root()

    parser.add_argument(
        "--input",
        "-i",
        default=str(Path(project_root) / "data" / "raw" / "lecture_transcripts"),
        help="Directory containing raw transcript files",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(Path(project_root) / "data" / "processed" / "lecture_transcripts"),
        help="Directory to save processed transcript files",
    )
    parser.add_argument(
        "--intermediate",
        "-m",
        default=str(
            Path(project_root) / "data" / "intermediate" / "lecture_transcripts"
        ),
        help="Directory to save cleaned transcript files",
    )
    parser.add_argument(
        "--single",
        "-s",
        help="Process a single file (provide filename only, not full path)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        help="Limit transcript processing to first N tokens (for testing)",
    )

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    intermediate_dir = args.intermediate

    if args.single:
        # Process just one specific file
        file_path = Path(input_dir) / args.single
        if file_path.exists():
            process_single_transcript(
                file_path, output_dir, intermediate_dir, max_tokens=args.max_tokens
            )
        else:
            logger.error(f"File {file_path} not found")
    else:
        # Process all transcript files in the directory
        transcript_files = sorted(Path(input_dir).glob("*.txt"))

        if not transcript_files:
            logger.error(f"No transcript files found in {input_dir}")
            exit(1)

        logger.info(f"Found {len(transcript_files)} transcript files to process")

        successful = 0
        for i, file_path in enumerate(transcript_files, 1):
            logger.info(f"\nProcessing file {i}/{len(transcript_files)}")
            result = process_single_transcript(
                file_path, output_dir, intermediate_dir, max_tokens=args.max_tokens
            )
            if result:
                successful += 1

            # Add delay to avoid hitting API rate limits
            if i < len(transcript_files):
                logger.info("Waiting before processing next transcript...")
                time.sleep(2)
