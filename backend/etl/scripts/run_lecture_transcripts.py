"""
Creates trascript files from the DEV309 - Development Policy Strategy class taught by Ricardo Hausmann.
The script uses OpenAI API to extract structured information from the lecture transcripts and saves them as JSON files.
"""

import os
import json
import time
import argparse

from pathlib import Path
from typing import List
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Define Pydantic model for lecture transcript data
class LectureTranscript(BaseModel):
    """ Model representing structured information from a lecture transcript """
    lecture_number: int
    title: str
    main_topics: list[str]
    summary: str
    transcript: str
    
def clean_transcript(transcript_text: str) -> str:
    """
    Clean and improve the raw lecture transcript using OpenAI API.
    
    Args:
        transcript_text (str): The raw transcript text to be processed.
    
    Returns:
        str: A cleaned and improved version of the transcript.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert assistant helping to clean up lecture transcripts from the 'Development Policy Strategy' class taught by Ricardo Hausmann (Director of Harvard's Growth Lab).
                    
                    Your task is to create a cleaned lecture transcript with the following improvements:
                    - Remove filler words, false starts, and repetitions
                    - Only preserve the content of the main speaker, excluding questions from the audience
                    - Organize content into logical paragraphs
                    - Fix any obvious transcription errors
                    - Maintain the full content and detail of the lecture
                    
                    Return only the cleaned transcript text with no additional commentary.
                    """
                },
                {
                    "role": "user",
                    "content": f"Here is a raw lecture transcript from the Development Policy Strategy class. Please clean it according to the instructions:\n\n{transcript_text}"
                }
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error cleaning transcript: {str(e)}")
        raise

def extract_lecture_metadata(clean_transcript: str, lecture_number: int) -> LectureTranscript:
    """
    Extract structured metadata from a cleaned lecture transcript using OpenAI API and Pydantic.
    
    Args:
        clean_transcript (str): The already cleaned transcript text.
        lecture_number (int): The lecture number for identification.
    
    Returns:
        LectureTranscript: A Pydantic model instance containing structured information about the lecture.
    """
    try:
        # Using the parse method with the Pydantic model
        completion = client.beta.chat.completions.parse(
            model="gpt-4.1-nano-2025-04-14",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert assistant analyzing lecture transcripts from the 'Development Policy Strategy' class 
                    taught by Ricardo Hausmann (Director of Harvard's Growth Lab). Extract the essential information from the transcript."""
                },
                {
                    "role": "user",
                    "content": f"Here is a cleaned lecture transcript #{lecture_number} from the Development Policy Strategy class. Please extract the key information:\n\n{clean_transcript}"
                }
            ],
            response_format=LectureTranscript,
            temperature=0.3
        )
        
        # Get the parsed data directly as a LectureTranscript object
        lecture_data = completion.choices[0].message.parsed
        
        # Ensure lecture number is set
        if lecture_data.lecture_number != lecture_number:
            lecture_data.lecture_number = lecture_number
            
        return lecture_data
        
    except Exception as e:
        print(f"Error extracting metadata for lecture {lecture_number}: {str(e)}")
        raise

def process_single_transcript(file_path: Path, output_dir: str, intermediate_dir: str = None) -> None:
    """
    Process a single transcript file and save the structured result to output directory.
    Also optionally saves the cleaned transcript to an intermediate directory.

    Args:
        file_path (Path): Path to the raw transcript file.
        output_dir (str): Directory to save processed transcript file.
        intermediate_dir (str, optional): Directory to save cleaned transcripts.
    """
    try:
        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract lecture number from filename or use index if not possible
        try:
            # Filenames start with lecture number like "01_lecture.txt"
            lecture_num = int(''.join(filter(str.isdigit, file_path.stem)))
        except:
            lecture_num = 0  # Default if unable to extract number
            print(f"Warning: Could not extract lecture number from filename {file_path.name}. Using 0 as default.")
        
        print(f"Processing file: {file_path.name} (Lecture #{lecture_num})")
        
        # Handle intermediate directory for cleaned transcripts
        cleaned_transcript = None
        if intermediate_dir:
            os.makedirs(intermediate_dir, exist_ok=True)
            clean_file_path = Path(intermediate_dir) / f"lecture_{lecture_num:02d}_cleaned.txt"
            
            # Check if cleaned transcript already exists
            if clean_file_path.exists():
                print(f"Found existing cleaned transcript: {clean_file_path}")
                with open(clean_file_path, 'r', encoding='utf-8') as f:
                    cleaned_transcript = f.read()
        
        # If no cleaned transcript exists yet, generate one
        if cleaned_transcript is None:
            # Read raw transcript content
            with open(file_path, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
                
            print(f"Cleaning transcript...")
            cleaned_transcript = clean_transcript(transcript_text)
            
            # Save cleaned transcript if intermediate directory is provided
            if intermediate_dir:
                clean_file_path = Path(intermediate_dir) / f"lecture_{lecture_num:02d}_cleaned.txt"
                with open(clean_file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_transcript)
                print(f"Saved cleaned transcript to: {clean_file_path}")
        
        # Extract metadata using the cleaned transcript
        print(f"Extracting metadata...")
        structured_data = extract_lecture_metadata(cleaned_transcript, lecture_num)
        
        # Save as JSON
        output_file = Path(output_dir) / f"lecture_{lecture_num:02d}_processed.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            # Convert Pydantic model to dict first
            model_dict = structured_data.model_dump()

            # Serialize the dict to JSON
            f.write(json.dumps(model_dict, indent=2, ensure_ascii=False))
        
        print(f"Successfully processed and saved: {output_file}")
        
        return True
            
    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")
        return False
    
def get_project_root():
    """Get the project root directory to allow running from any location."""
    script_path = Path(__file__).resolve()
    return str(script_path.parent.parent.parent.parent)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process lecture transcripts using OpenAI API.")
    
    # Get project root for default paths
    project_root = get_project_root()
    
    parser.add_argument(
        "--input", "-i",
        default=str(Path(project_root) / "backend" / "etl" / "data" / "raw" / "transcripts"),
        help="Directory containing raw transcript files"
    )
    parser.add_argument(
        "--output", "-o",
        default=str(Path(project_root) / "backend" / "etl" / "data" / "processed" / "transcripts"),
        help="Directory to save processed transcript files"
    )
    parser.add_argument(
        "--intermediate", "-m",
        default=str(Path(project_root) / "backend" / "etl" / "data" / "intermediate" / "transcripts"),
        help="Directory to save cleaned transcript files"
    )
    parser.add_argument(
        "--single", "-s",
        help="Process a single file (provide filename only, not full path)"
    )
    
    args = parser.parse_args()
    
    input_dir = args.input
    output_dir = args.output
    intermediate_dir = args.intermediate
    
    if args.single:
        # Process just one specific file
        file_path = Path(input_dir) / args.single
        if file_path.exists():
            process_single_transcript(file_path, output_dir, intermediate_dir)
        else:
            print(f"Error: File {file_path} not found")
    else:
        # Process all transcript files in the directory
        transcript_files = sorted(Path(input_dir).glob('*.txt'))
        
        if not transcript_files:
            print(f"No transcript files found in {input_dir}")
            exit(1)
        
        print(f"Found {len(transcript_files)} transcript files to process")
        
        successful = 0
        for i, file_path in enumerate(transcript_files, 1):
            print(f"\nProcessing file {i}/{len(transcript_files)}")
            result = process_single_transcript(file_path, output_dir, intermediate_dir)
            if result:
                successful += 1
            
            # Add delay to avoid hitting API rate limits
            if i < len(transcript_files):
                print("Waiting before processing next transcript...")
                time.sleep(2)
        
        print(f"\nProcessing complete! {successful}/{len(transcript_files)} files successfully processed and saved to {output_dir}")