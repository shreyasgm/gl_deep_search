import json
import os
from pathlib import Path

import pytest

from backend.etl.scripts.run_lecture_transcripts import (
    LectureTranscript,
    clean_transcript,
    extract_lecture_metadata,
    process_single_transcript,
)

# Check if we're running in CI (GitHub Actions)
in_github_actions = os.environ.get("GITHUB_ACTIONS") == "true"

# Skip all data-dependent tests in GitHub Actions
skip_data_tests = pytest.mark.skipif(
    in_github_actions, reason="Skipping tests that require data files in GitHub Actions"
)

# Sample text for testing when real data is not available
SAMPLE_TEXT = """
Welcome to Development Policy Strategy. I'm Ricardo Hausmann, the instructor.
Today we'll discuss economic growth theories and their policy implications.
First, we'll cover the basics of growth accounting.
Then we'll move on to structural transformation and productivity.
Finally, we'll discuss the importance of economic complexity in development.
"""


# Fixture to load a sample raw transcript
@pytest.fixture
def sample_raw_transcript():
    # In GitHub Actions, use the sample text
    if in_github_actions:
        return SAMPLE_TEXT

    # In local environment, use real data file
    raw_transcript_path = Path("data/raw/lecture_transcripts/0_intro.txt")
    if not raw_transcript_path.exists():
        return SAMPLE_TEXT

    with open(raw_transcript_path, encoding="utf-8") as f:
        return f.read()


# Fixture to load a sample cleaned transcript
@pytest.fixture
def sample_cleaned_transcript():
    # In GitHub Actions, use the sample text
    if in_github_actions:
        return SAMPLE_TEXT

    # In local environment, use real data file
    cleaned_transcript_path = Path(
        "data/intermediate/lecture_transcripts/lecture_00_cleaned.txt"
    )
    if not cleaned_transcript_path.exists():
        return SAMPLE_TEXT

    with open(cleaned_transcript_path, encoding="utf-8") as f:
        return f.read()


# Unit test for clean_transcript
def test_clean_transcript(sample_raw_transcript):
    # Limit the raw transcript to the first 500 characters for testing
    limited_raw_transcript = sample_raw_transcript[:500]

    # Clean the transcript
    cleaned = clean_transcript(limited_raw_transcript)

    # Ensure the cleaned transcript is not empty
    assert len(cleaned) > 0, "Cleaned transcript should not be empty."

    # Ensure the cleaned transcript starts with meaningful content
    assert not cleaned.startswith("um") and not cleaned.startswith("uh"), (
        "Cleaned transcript should not start with filler words."
    )

    # Ensure the cleaned transcript is shorter than the raw transcript
    assert len(cleaned) < len(limited_raw_transcript), (
        "Cleaned transcript should be shorter than the raw transcript."
    )


# Unit test for extract_lecture_metadata
def test_extract_lecture_metadata(sample_cleaned_transcript):
    lecture_number = 0
    metadata = extract_lecture_metadata(sample_cleaned_transcript, lecture_number)
    assert isinstance(metadata, LectureTranscript)
    assert metadata.lecture_number == lecture_number
    assert len(metadata.title) > 0  # Ensure the title is extracted


# Integration test for process_single_transcript using real data
@skip_data_tests
def test_process_single_transcript(tmp_path):
    # Use a real raw transcript file
    raw_transcript_path = Path(
        "data/raw/lecture_transcripts/1_malthusian_economics.txt"
    )

    if not raw_transcript_path.exists():
        pytest.skip("Test transcript file does not exist")

    # Define output and intermediate directories
    output_dir = tmp_path / "processed"
    intermediate_dir = tmp_path / "intermediate"

    # Process the transcript
    result = process_single_transcript(
        raw_transcript_path, str(output_dir), str(intermediate_dir)
    )

    # Check that processing was successful
    assert result is True

    # Verify output file exists
    output_file = output_dir / "lecture_01_processed.json"
    assert output_file.exists()

    # Verify the content of the output file
    with open(output_file, encoding="utf-8") as f:
        data = json.load(f)
        assert "lecture_number" in data
        assert data["lecture_number"] == 1
