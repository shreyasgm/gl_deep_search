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
    main_topics: List[str] = Field(..., min_items=1)
    summary: str
    transcript: str = Field(..., min_length=100)

    # Add custom field validators    
    @field_validator('summary')
    def validate_summary_length(cls, summary, values):
        """Ensure summary is appropriate length relative to transcript."""
        # This validator receives both the field value and a dict of previously validated values
        if 'transcript' in values:
            transcript = values['transcript']
            
            # Calculate appropriate summary length
            min_length = 200  # Minimum 200 characters
            max_length = len(transcript) * 0.15  # No more than 15% of transcript length
            
            if len(summary) < min_length:
                raise ValueError(f"Summary is too short. Expected at least {min_length:.0f} characters.")
            
            if len(summary) > max_length:
                raise ValueError(f"Summary is too long. Expected no more than {max_length:.0f} characters.")
                
        return summary.strip()
    
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
            model="gpt-4o-mini",
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
            model="gpt-4o-mini",
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

def process_single_transcript(file_path: Path, output_dir: str) -> None:
    """
    Process a single transcript file and save the structured result to output directory.

    Args:
        file_path (Path): Path to the raw transcript file.
        output_dir (str): Directory to save processed transcript file.
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing file: {file_path.name}")
        
        # Read transcript content
        with open(file_path, 'r', encoding='utf-8') as f:
            transcript_text = f.read()
        
        # Extract lecture number from filename or use index if not possible
        try:
            # Filenames start with lecture number like "01_lecture.txt"
            lecture_num = int(''.join(filter(str.isdigit, file_path.stem)))
        except:
            lecture_num = 0  # Default if unable to extract number
            print(f"Warning: Could not extract lecture number from filename {file_path.name}. Using 0 as default.")
        
        # Step 1: Clean the transcript
        print(f"Cleaning transcript...")
        cleaned_transcript = clean_transcript(transcript_text)
        
        # Step 2: Extract metadata using the cleaned transcript
        print(f"Extracting metadata...")
        structured_data = extract_lecture_metadata(cleaned_transcript, lecture_num)
        
        # Save as JSON
        output_file = Path(output_dir) / f"lecture_{lecture_num:02d}_processed.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(structured_data.model_dump_json(indent=2, ensure_ascii=False))
        
        print(f"Successfully processed and saved: {output_file}")
        
        return True
            
    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")
        return False