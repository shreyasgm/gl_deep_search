#!/usr/bin/env python
"""
Verify the dummy database contents.
"""

import sqlite3
from pathlib import Path

# Define database path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "publication_tracking.db"

def check_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get table schema
    cursor.execute("PRAGMA table_info(publication_tracking)")
    columns = cursor.fetchall()
    print(f"Database columns ({len(columns)}):")
    for i, col in enumerate(columns, 1):
        print(f"{i}. {col[1]} ({col[2]})")
    
    # Get record count
    cursor.execute("SELECT COUNT(*) FROM publication_tracking")
    count = cursor.fetchone()[0]
    print(f"\nTotal records: {count}")
    
    # Sample a few records
    cursor.execute("""
        SELECT publication_id, title, year, download_status, 
               processing_status, embedding_status, ingestion_status
        FROM publication_tracking
        LIMIT 5
    """)
    sample = cursor.fetchall()
    
    print("\nSample records:")
    for rec in sample:
        print(f"ID: {rec[0]}")
        print(f"  Title: {rec[1]}")
        print(f"  Year: {rec[2]}")
        print(f"  Status: Download={rec[3]}, Processing={rec[4]}, Embedding={rec[5]}, Ingestion={rec[6]}")
        print()
    
    # Check status distribution
    print("Status distribution:")
    for stage in ['download', 'processing', 'embedding', 'ingestion']:
        cursor.execute(f"""
            SELECT {stage}_status, COUNT(*)
            FROM publication_tracking
            GROUP BY {stage}_status
        """)
        statuses = cursor.fetchall()
        print(f"\n{stage.title()} status:")
        for status, count in statuses:
            print(f"  {status}: {count}")
    
    conn.close()

if __name__ == "__main__":
    check_database()
