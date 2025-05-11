import asyncio
import logging
import os

import aiofiles
import aiohttp
import pandas as pd
import tqdm.asyncio

logger = logging.getLogger(__name__)


async def download_file(session, url, destination_path):
    """Download a single file"""
    # Ensure directory exists
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    try:
        async with session.get(url) as response:
            if response.status == 200:
                async with aiofiles.open(destination_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(64 * 1024):
                        await f.write(chunk)
                return True, None
            else:
                return False, f"HTTP error {response.status}"
    except Exception as e:
        return False, str(e)


async def download_all_files(csv_path, output_dir, concurrency=3):
    """Download all files marked as samples"""
    # Read the CSV
    df = pd.read_csv(csv_path)

    # Filter for sample files with valid file URLs
    sample_files = df[df["sample"] & (~df["file_url"].isna())]
    logger.info(f"Found {len(sample_files)} files to download")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for _, row in sample_files.iterrows():
            # Get filename from URL
            file_url = row["file_url"]
            paper_id = row["paper_id"]
            file_name = file_url.split("/")[-1]

            # Create destination path - using the paper_id to organize files
            dest_path = os.path.join(output_dir, paper_id, file_name)

            # Create download task
            async def bounded_download(url=file_url, path=dest_path):
                async with semaphore:
                    return await download_file(session, url, path)

            tasks.append(bounded_download())

        # Execute downloads with progress bar
        results = []
        with tqdm.asyncio.tqdm(total=len(tasks), desc="Downloading files") as pbar:
            for task in asyncio.as_completed(tasks):
                success, error = await task
                results.append((success, error))
                pbar.update(1)
                if success:
                    pbar.set_postfix_str("Last: success")
                else:
                    pbar.set_postfix_str(f"Last: failed - {error}")

    # Log summary
    successful = sum(1 for success, _ in results if success)
    failed = len(results) - successful
    logger.info(f"Download summary: {successful} successful, {failed} failed")

    return results


# Run the download function
if __name__ == "__main__":
    CSV_PATH = "data/publevel.csv"  # Path to your CSV file
    OUTPUT_DIR = "downloaded_papers"  # Where to save downloaded files

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    asyncio.run(download_all_files(CSV_PATH, OUTPUT_DIR, concurrency=3))
