#!/usr/bin/env python
"""
Generate a dummy SQLite database for publication tracking with well-distributed data.

This script creates a SQLite database with the schema defined in etl_metadata_schema.sql
and populates it with sample data covering different scenarios for testing purposes.
"""

import datetime
import json
import os
import random
import sqlite3
import string
from pathlib import Path

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SCHEMA_PATH = PROJECT_ROOT / "backend" / "storage" / "etl_metadata_schema.sql"
DB_DIR = PROJECT_ROOT / "data" / "processed"
DB_PATH = DB_DIR / "publication_tracking.db"

# Ensure output directory exists
DB_DIR.mkdir(parents=True, exist_ok=True)

# Sample data for generation
TITLES = [
    "Economic Growth and Productivity in Developing Countries",
    "The Impact of Climate Change on Global Food Security",
    "Artificial Intelligence: Applications in Healthcare",
    "Innovation and Technology Transfer in Rural Communities",
    "Urban Planning and Sustainability: Case Studies from Asia",
    "Renewable Energy Adoption: Barriers and Opportunities",
    "Digital Transformation in the Public Sector",
    "International Trade Policies and Economic Development",
    "Microfinance and Poverty Reduction Strategies",
    "Supply Chain Resilience in Global Crises",
    "Gender Equality and Economic Empowerment",
    "Water Resource Management in Arid Regions",
    "Blockchain Applications in Supply Chain Management",
    "Public Health Infrastructure in Developing Nations",
    "Education Technology and Learning Outcomes",
    "Migration Patterns and Economic Implications",
    "Financial Inclusion and Digital Banking",
    "Sustainable Agriculture Practices in Changing Climates",
    "Entrepreneurship Ecosystems in Emerging Markets",
    "Circular Economy Implementation: Global Best Practices",
]

AUTHORS = [
    "John Smith, Maria Garcia",
    "David Johnson, Ahmed Hassan, Lisa Wong",
    "Emma Wilson, Carlos Rodriguez",
    "James Brown, Priya Patel, Olga Ivanova",
    "Michael Chen, Sarah Lee",
    "Robert Taylor, Fatima Ahmed",
    "Daniel Martinez, Aisha Nkosi",
    "Thomas Wilson, Jing Zhang, Eva Müller",
    "Christopher Davis, Sofia Perez",
    "Matthew Anderson, Leila Osman",
]

SOURCES = [
    "https://growthlab.hks.harvard.edu/publications/",
    "https://www.nber.org/papers/",
    "https://www.worldbank.org/en/research/",
    "https://www.imf.org/en/Publications/",
    "https://academic.oup.com/journals/",
    "https://www.sciencedirect.com/journal/",
    "https://link.springer.com/article/",
    "https://www.tandfonline.com/doi/abs/",
    "https://www.jstor.org/stable/",
    "https://onlinelibrary.wiley.com/doi/",
]

# Status options for each stage
DOWNLOAD_STATUS = ['Pending', 'In Progress', 'Downloaded', 'Failed']
PROCESSING_STATUS = ['Pending', 'In Progress', 'Processed', 'OCR_Failed', 'Chunking_Failed', 'Failed']
EMBEDDING_STATUS = ['Pending', 'In Progress', 'Embedded', 'Failed']
INGESTION_STATUS = ['Pending', 'In Progress', 'Ingested', 'Failed']

# Generate a random hash
def generate_hash(length=16):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

# Generate random timestamp within the past year
def generate_timestamp(days_ago_max=365):
    days_ago = random.randint(0, days_ago_max)
    return (datetime.datetime.now() - datetime.timedelta(days=days_ago)).strftime('%Y-%m-%d %H:%M:%S')

def generate_publication_v2(idx):
    """Generate publication with more evenly distributed stages."""
    pub_id = f"gl_{generate_hash(8)}"
    title = random.choice(TITLES)
    authors = random.choice(AUTHORS)
    year = random.randint(2010, 2025)
    abstract = f"This is a sample abstract for the publication titled '{title}'."
    source_url = f"{random.choice(SOURCES)}{generate_hash(8)}"
    
    file_urls = []
    num_files = random.randint(1, 3)
    for _ in range(num_files):
        file_type = random.choice(['pdf', 'docx', 'xlsx'])
        file_urls.append(f"https://example.com/files/{generate_hash(8)}.{file_type}")
    
    # Generate timestamps based on status
    discovery_timestamp = generate_timestamp(365)
    
    # Group records into progression scenarios
    scenario = idx % 10  # 0-9 different scenarios
    
    # Scenario 0: Just discovered, still pending download (10% of records)
    if scenario == 0:
        download_status = 'Pending'
        download_timestamp = None
        download_attempt_count = 0
        
        processing_status = 'Pending'
        processing_timestamp = None
        processing_attempt_count = 0
        
        embedding_status = 'Pending'
        embedding_timestamp = None
        embedding_attempt_count = 0
        
        ingestion_status = 'Pending'
        ingestion_timestamp = None
        ingestion_attempt_count = 0
    
    # Scenario 1: Download in progress (10% of records)
    elif scenario == 1:
        download_status = 'In Progress'
        download_timestamp = generate_timestamp(30)
        download_attempt_count = random.randint(1, 3)
        
        processing_status = 'Pending'
        processing_timestamp = None
        processing_attempt_count = 0
        
        embedding_status = 'Pending'
        embedding_timestamp = None
        embedding_attempt_count = 0
        
        ingestion_status = 'Pending'
        ingestion_timestamp = None
        ingestion_attempt_count = 0
    
    # Scenario 2: Download failed (10% of records)
    elif scenario == 2:
        download_status = 'Failed'
        download_timestamp = generate_timestamp(60)
        download_attempt_count = random.randint(1, 5)
        
        processing_status = 'Pending'
        processing_timestamp = None
        processing_attempt_count = 0
        
        embedding_status = 'Pending'
        embedding_timestamp = None
        embedding_attempt_count = 0
        
        ingestion_status = 'Pending'
        ingestion_timestamp = None
        ingestion_attempt_count = 0
    
    # Scenario 3: Downloaded, pending processing (10% of records)
    elif scenario == 3:
        download_status = 'Downloaded'
        download_timestamp = generate_timestamp(90)
        download_attempt_count = random.randint(1, 3)
        
        processing_status = 'Pending'
        processing_timestamp = None
        processing_attempt_count = 0
        
        embedding_status = 'Pending'
        embedding_timestamp = None
        embedding_attempt_count = 0
        
        ingestion_status = 'Pending'
        ingestion_timestamp = None
        ingestion_attempt_count = 0
    
    # Scenario 4: Processing in progress (10% of records)
    elif scenario == 4:
        download_status = 'Downloaded'
        download_timestamp = generate_timestamp(120)
        download_attempt_count = random.randint(1, 2)
        
        processing_status = 'In Progress'
        processing_timestamp = generate_timestamp(90)
        processing_attempt_count = random.randint(1, 2)
        
        embedding_status = 'Pending'
        embedding_timestamp = None
        embedding_attempt_count = 0
        
        ingestion_status = 'Pending'
        ingestion_timestamp = None
        ingestion_attempt_count = 0
    
    # Scenario 5: Processing failed (OCR or chunking) (10% of records)
    elif scenario == 5:
        download_status = 'Downloaded'
        download_timestamp = generate_timestamp(150)
        download_attempt_count = random.randint(1, 2)
        
        processing_status = random.choice(['OCR_Failed', 'Chunking_Failed', 'Failed'])
        processing_timestamp = generate_timestamp(120)
        processing_attempt_count = random.randint(1, 3)
        
        embedding_status = 'Pending'
        embedding_timestamp = None
        embedding_attempt_count = 0
        
        ingestion_status = 'Pending'
        ingestion_timestamp = None
        ingestion_attempt_count = 0
    
    # Scenario 6: Processed, pending embedding (10% of records)
    elif scenario == 6:
        download_status = 'Downloaded'
        download_timestamp = generate_timestamp(180)
        download_attempt_count = random.randint(1, 2)
        
        processing_status = 'Processed'
        processing_timestamp = generate_timestamp(150)
        processing_attempt_count = random.randint(1, 2)
        
        embedding_status = 'Pending'
        embedding_timestamp = None
        embedding_attempt_count = 0
        
        ingestion_status = 'Pending'
        ingestion_timestamp = None
        ingestion_attempt_count = 0
    
    # Scenario 7: Embedding in progress or failed (10% of records)
    elif scenario == 7:
        download_status = 'Downloaded'
        download_timestamp = generate_timestamp(210)
        download_attempt_count = random.randint(1, 2)
        
        processing_status = 'Processed'
        processing_timestamp = generate_timestamp(180)
        processing_attempt_count = random.randint(1, 2)
        
        embedding_status = random.choice(['In Progress', 'Failed'])
        embedding_timestamp = generate_timestamp(160)
        embedding_attempt_count = random.randint(1, 3)
        
        ingestion_status = 'Pending'
        ingestion_timestamp = None
        ingestion_attempt_count = 0
    
    # Scenario 8: Embedded, pending ingestion (10% of records)
    elif scenario == 8:
        download_status = 'Downloaded'
        download_timestamp = generate_timestamp(240)
        download_attempt_count = random.randint(1, 2)
        
        processing_status = 'Processed'
        processing_timestamp = generate_timestamp(210)
        processing_attempt_count = random.randint(1, 2)
        
        embedding_status = 'Embedded'
        embedding_timestamp = generate_timestamp(180)
        embedding_attempt_count = random.randint(1, 2)
        
        ingestion_status = 'Pending'
        ingestion_timestamp = None
        ingestion_attempt_count = 0
    
    # Scenario 9: Complete pipeline (successfully ingested or failed ingestion) (10% of records)
    else:
        download_status = 'Downloaded'
        download_timestamp = generate_timestamp(300)
        download_attempt_count = random.randint(1, 2)
        
        processing_status = 'Processed'
        processing_timestamp = generate_timestamp(270)
        processing_attempt_count = random.randint(1, 2)
        
        embedding_status = 'Embedded'
        embedding_timestamp = generate_timestamp(240)
        embedding_attempt_count = random.randint(1, 2)
        
        ingestion_status = random.choice(['Ingested', 'In Progress', 'Failed'])
        ingestion_timestamp = generate_timestamp(210)
        ingestion_attempt_count = random.randint(1, 3)
    
    # Last updated is the most recent timestamp among all stages
    timestamps = [t for t in [discovery_timestamp, download_timestamp, processing_timestamp, embedding_timestamp, ingestion_timestamp] if t]
    last_updated = max(timestamps) if timestamps else discovery_timestamp
    
    # Set error message if any stage failed
    error_message = None
    if 'Failed' in [download_status, processing_status, embedding_status, ingestion_status]:
        failed_stage = 'download' if download_status == 'Failed' else \
                      'processing' if processing_status == 'Failed' else \
                      'embedding' if embedding_status == 'Failed' else 'ingestion'
        errors = [
            f"Connection timeout during {failed_stage}",
            f"HTTP Error 404: Not Found during {failed_stage}",
            f"File is corrupted or unsupported format during {failed_stage}",
            f"Insufficient memory for operation during {failed_stage}",
            f"API rate limit exceeded during {failed_stage}",
            f"Invalid response format from server during {failed_stage}"
        ]
        error_message = random.choice(errors)
    
    content_hash = generate_hash(64)
    
    return (
        pub_id, source_url, title, authors, year, abstract, 
        json.dumps(file_urls), content_hash, discovery_timestamp, 
        download_status, download_timestamp, download_attempt_count,
        processing_status, processing_timestamp, processing_attempt_count,
        embedding_status, embedding_timestamp, embedding_attempt_count,
        ingestion_status, ingestion_timestamp, ingestion_attempt_count,
        last_updated, error_message
    )

def create_and_populate_database():
    # Remove existing database if it exists
    if DB_PATH.exists():
        os.remove(DB_PATH)
    
    # Create connection to SQLite database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Read schema SQL from file and execute
    with open(SCHEMA_PATH, 'r') as f:
        schema_sql = f.read()
    
    # Execute the schema SQL
    cursor.executescript(schema_sql)
    
    # Generate dummy data
    num_records = 50  # Number of dummy publications to create
    publications = []
    
    for i in range(num_records):
        publications.append(generate_publication_v2(i))
    
    # Insert the data into the database
    cursor.executemany('''
        INSERT INTO publication_tracking (
            publication_id, source_url, title, authors, year, abstract, 
            file_urls, content_hash, discovery_timestamp, 
            download_status, download_timestamp, download_attempt_count,
            processing_status, processing_timestamp, processing_attempt_count,
            embedding_status, embedding_timestamp, embedding_attempt_count,
            ingestion_status, ingestion_timestamp, ingestion_attempt_count,
            last_updated, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', publications)
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"Created dummy database at {DB_PATH}")
    print(f"Generated {num_records} publication records")
    
    # Print some statistics
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("\nStatus Distribution:")
    for status_field in ['download_status', 'processing_status', 'embedding_status', 'ingestion_status']:
        cursor.execute(f"SELECT {status_field}, COUNT(*) FROM publication_tracking GROUP BY {status_field}")
        results = cursor.fetchall()
        print(f"\n{status_field.replace('_', ' ').title()}:")
        for status, count in results:
            print(f"  {status}: {count}")
    
    conn.close()

if __name__ == "__main__":
    create_and_populate_database()
