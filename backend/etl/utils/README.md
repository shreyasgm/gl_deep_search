# File Downloaders Documentation

This document provides detailed explanations of how the GrowthLab and OpenAlex file downloaders work.

## Table of Contents
- [File Downloaders Documentation](#file-downloaders-documentation)
  - [Table of Contents](#table-of-contents)
  - [GrowthLab File Downloader ](#growthlab-file-downloader-)
    - [Overview ](#overview-)
    - [Key Features ](#key-features-)
    - [Core Components ](#core-components-)
    - [Download Process ](#download-process-)
    - [Configuration ](#configuration-)
    - [Usage Example ](#usage-example-)
  - [OpenAlex File Downloader ](#openalex-file-downloader-)
    - [Overview ](#overview--1)
    - [Key Features ](#key-features--1)
    - [Core Components ](#core-components--1)
    - [Download Process ](#download-process--1)
    - [Configuration ](#configuration--1)
    - [Usage Example ](#usage-example--1)

## GrowthLab File Downloader <a name="growthlab-file-downloader"></a>

### Overview <a name="gl-overview"></a>

The GrowthLab file downloader is an asynchronous tool designed to download files from Growth Lab publication URLs. It handles the complexities of concurrent downloads, rate limiting, retries, and file validation.

### Key Features <a name="gl-key-features"></a>

- **Asynchronous downloads** with configurable concurrency limits
- **Intelligent retry logic** with exponential backoff
- **Resume support** for partially downloaded files
- **File validation** to ensure downloaded files are correct and complete
- **Caching** to avoid re-downloading existing files
- **Progress tracking** with detailed statistics
- **Configurable rate limiting** to avoid overwhelming servers

### Core Components <a name="gl-core-components"></a>

1. **FileDownloader class**: Main class responsible for downloading files
2. **DownloadResult dataclass**: Represents the result of a download operation
3. **download_growthlab_files function**: High-level function to download files for multiple publications

### Download Process <a name="gl-download-process"></a>

The file downloader follows this process:

1. **Initialization**:
   - Creates a configurable storage backend
   - Sets up concurrency controls with semaphores
   - Configures retry parameters and file validation thresholds
   - Initializes download statistics

2. **For each publication**:
   - Extracts file URLs from publication metadata
   - Determines appropriate file paths to save each file
   - Creates download tasks for each URL

3. **For each file download**:
   - Checks if the file already exists (caching)
   - Creates an HTTP session with proper headers (including random user agents)
   - Uses a semaphore to limit concurrent downloads
   - Attempts to download the file with retry logic
   - Adds random delays between requests to avoid rate limiting

4. **Download implementation**:
   - Supports resuming partial downloads using HTTP range requests
   - Handles various HTTP status codes (200, 206, 416)
   - Downloads file content in chunks
   - Tracks download progress

5. **File validation**:
   - Checks file size (minimum and maximum thresholds)
   - Verifies file format based on content type and file signatures
   - Validates PDF files by checking for "%PDF-" signature
   - Validates Word documents (.doc, .docx) by checking appropriate signatures

6. **Reporting**:
   - Tracks statistics on successful, failed, and cached downloads
   - Logs a detailed summary with success rates and total data downloaded

### Configuration <a name="gl-configuration"></a>

The downloader can be configured with:

- **concurrency_limit**: Maximum number of concurrent downloads
- **download_delay**: Delay between downloads to avoid overwhelming servers
- **retry settings**: max_retries, base_delay, and max_delay
- **file validation thresholds**: min_file_size and max_file_size
- **user_agent_list**: List of user agents to rotate through for requests

Configuration is loaded from a YAML file with fallback to default values.

### Usage Example <a name="gl-usage-example"></a>

The file downloader is typically run using the `run_gl_file_downloader.py` script:

```python
# Run with default settings
python -m backend.etl.scripts.run_gl_file_downloader

# Run with custom settings
python -m backend.etl.scripts.run_gl_file_downloader \
    --publication-data path/to/publications.csv \
    --overwrite \
    --limit 10 \
    --concurrency 5
```

Alternatively, the downloader can be used programmatically:

```python
from backend.etl.utils.gl_file_downloader import download_growthlab_files

# Download files asynchronously
results = await download_growthlab_files(
    publication_data_path=Path("path/to/data.csv"),
    overwrite=False,
    limit=10,
    concurrency=3
)
```

## OpenAlex File Downloader <a name="openalex-file-downloader"></a>

### Overview <a name="oa-overview"></a>

The OpenAlex file downloader is designed to download academic papers from DOIs, with special handling for open access publications. It integrates with open access APIs and includes fallback mechanisms for accessing papers that aren't freely available.

### Key Features <a name="oa-key-features"></a>

- **Open access verification** to find freely available versions of papers
- **Fallback to scidownl** for closed-access papers
- **Asynchronous downloads** with configurable concurrency
- **Intelligent retry logic** with exponential backoff
- **DOI-specific handling** for academic papers
- **File validation** to ensure downloaded files are valid
- **Progress tracking** with detailed statistics

### Core Components <a name="oa-core-components"></a>

1. **OpenAlexFileDownloader class**: Main class responsible for downloading files
2. **DownloadResult dataclass**: Represents the result of a download operation
3. **download_openalex_files function**: High-level function to download files for multiple publications

### Download Process <a name="oa-download-process"></a>

The OpenAlex file downloader follows a more complex process than the GrowthLab downloader:

1. **Initialization**:
   - Creates a configurable storage backend
   - Sets up concurrency controls with semaphores
   - Configures retry parameters and file validation thresholds
   - Initializes download statistics including open access and scidownl counters

2. **For each publication**:
   - Extracts file URLs (typically DOIs) from publication metadata
   - Determines appropriate file paths to save each file
   - Creates download tasks for each URL

3. **For each DOI/URL**:
   - Checks if the file already exists (caching)
   - For DOIs, follows a two-step process:
     a. **Check for open access versions** using services like Unpaywall and CORE
     b. **Fall back to scidownl** if no open access version is found or download fails

4. **Open Access Check**:
   - Queries the Unpaywall API to find open access versions of papers
   - If available, retrieves the best download URL (preferring direct PDF links)
   - Falls back to CORE API if Unpaywall doesn't find a result

5. **Download methods**:
   - **HTTP download** for open access papers using aiohttp
   - **scidownl download** for papers not available through open access
     - Uses a subprocess to run the scidownl command-line tool
     - Creates a temporary directory for the download
     - Moves the downloaded file to the destination path

6. **File validation**:
   - Checks file size (minimum and maximum thresholds)
   - Verifies file format based on content type and file signatures
   - Specifically validates PDF files by checking for "%PDF-" signature
   - Deletes invalid files to avoid caching broken downloads

7. **Reporting**:
   - Tracks statistics on successful, failed, and cached downloads
   - Additionally tracks open access vs scidownl downloads
   - Logs a detailed summary with success rates, methods used, and total data downloaded

### Configuration <a name="oa-configuration"></a>

In addition to the standard download configuration, the OpenAlex downloader includes:

- **unpaywall_email**: Email to use with the Unpaywall API
- **open_access_apis**: List of APIs to try for open access papers
- Other common settings like concurrency_limit, download_delay, and retry parameters

### Usage Example <a name="oa-usage-example"></a>

The OpenAlex file downloader is typically run using the `run_openalex_file_downloader.py` script:

```python
# Run with default settings
python -m backend.etl.scripts.run_openalex_file_downloader

# Run with custom settings
python -m backend.etl.scripts.run_openalex_file_downloader \
    --input path/to/publications.csv \
    --limit 10 \
    --concurrency 3 \
    --overwrite \
    --verbose
```

Alternatively, the downloader can be used programmatically:

```python
from backend.etl.utils.oa_file_downloader import download_openalex_files

# Download files asynchronously
results = await download_openalex_files(
    publication_data_path=Path("path/to/data.csv"),
    overwrite=False,
    limit=10,
    concurrency=3
)
```
