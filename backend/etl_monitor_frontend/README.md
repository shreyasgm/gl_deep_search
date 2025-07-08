# ETL Monitor Frontend

This directory contains tools for monitoring and exporting publication status data from the ETL tracking database.

## Overview

The ETL Monitor Frontend provides a simple way to export publication status data to CSV format without requiring a full web interface. This is useful for quick data analysis, reporting, and monitoring the health of the publication processing pipeline.

## Files

- `export_publication_status.py` - Main script for exporting publication data to CSV
- `README.md` - This documentation file
- `code_explanation.md` - Technical explanation of the implementation

## Usage

### Basic Export

To export all publication status data with default settings:

```bash
uv run backend/etl_monitor_frontend/export_publication_status.py
```

This will:
1. Connect to the API at `http://localhost:8000` (default)
2. Fetch all publication records using pagination
3. Export to a timestamped CSV file in the current directory
4. Generate a JSON file with summary statistics

### Custom Options

```bash
# Custom output file
uv run backend/etl_monitor_frontend/export_publication_status.py --output my_export.csv

# Different API URL
uv run backend/etl_monitor_frontend/export_publication_status.py --api-url http://production-server:8000

# Quiet mode (less verbose logging)
uv run backend/etl_monitor_frontend/export_publication_status.py --quiet
```

## Prerequisites

1. **API Service Running**: The Publication Tracking API must be running
   ```bash
   python -m backend.service.main
   ```

2. **Dependencies**: Required packages (included in project dependencies)
   - `requests` - For API communication
   - `pandas` - For data processing
   - `python >= 3.12`

## Output Files

### CSV File
Contains all publication records with the following columns (in order):
- **Basic Metadata**: `publication_id`, `title`, `authors`, `year`
- **Status Fields**: `download_status`, `processing_status`, `embedding_status`, `ingestion_status`
- **Timestamps**: `last_updated`, `discovery_timestamp`, individual status timestamps
- **Attempt Counts**: `download_attempt_count`, `processing_attempt_count`, etc.
- **Additional Data**: `source_url`, `file_urls`, `abstract`, `error_message`

### JSON Statistics File
Contains summary information:
- Total publication count
- Status distribution for each pipeline stage
- Year distribution
- Recent activity metrics (7-day window)
- Export timestamp

## Use Cases

1. **Pipeline Monitoring**: Get an overview of ETL pipeline health and progress
2. **Data Analysis**: Import CSV into Excel, R, Python, or other analysis tools
3. **Reporting**: Generate status reports for stakeholders
4. **Debugging**: Identify failed publications and error patterns
5. **Historical Tracking**: Compare exports over time to track progress

## Troubleshooting

### Common Issues

1. **Connection Error**: Ensure the API service is running
   ```bash
   uv run -m backend.service.main
   ```

2. **Module Import Error**: Make sure you're in the project root directory and dependencies are installed

3. **Empty Results**: Check if the ETL tracking database exists and contains data

4. **Permission Error**: Ensure the output directory is writable

### Checking API Status

You can manually verify the API is running by visiting:
- Health endpoint: `http://localhost:8000/api/health`
- API docs: `http://localhost:8000/docs`

## Integration

The exported CSV files can be easily integrated with:
- **Excel/Google Sheets**: Direct import for manual analysis
- **Business Intelligence Tools**: Power BI, Tableau, etc.
- **Data Analysis**: Python pandas, R, statistical software
- **Databases**: Import into PostgreSQL, MySQL for further querying
- **Automation**: Include in scheduled reports or monitoring dashboards

## File Location

By default, files are exported to the `backend/etl_monitor_frontend/` directory with timestamped filenames to prevent overwrites and maintain historical records.
