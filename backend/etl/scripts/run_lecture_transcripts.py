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