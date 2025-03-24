"""
Minimal test for Growth Lab scraper functionality
"""
import asyncio
import logging
from pathlib import Path

import aiohttp
import yaml
from bs4 import BeautifulSoup

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_scraper_functionality():
    """Test if we can load the config and access the Growth Lab website"""
    
    # 1. Test loading config - FIXED PATH for running from scrapers directory
    config_path = Path(__file__).parent.parent / "config.yaml"
    
    # Debug the path to verify it's correct
    logger.info(f"Looking for config at: {config_path.absolute()}")
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            base_url = config["sources"]["growth_lab"]["base_url"]
            logger.info(f"Successfully loaded config, base_url: {base_url}")
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        # Fall back to hardcoded URL if config loading fails
        base_url = "https://growthlab.hks.harvard.edu/publications"
        logger.info(f"Using fallback base_url: {base_url}")
    
    # 2. Test connecting to website
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url) as response:
                if response.status == 200:
                    logger.info(f"Successfully connected to {base_url}")
                    
                    # 3. Test parsing HTML
                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")
                    
                    # Find first publication title
                    first_pub = soup.find("span", {"class": "biblio-title"})
                    if first_pub:
                        logger.info(f"Found publication: {first_pub.text.strip()}")
                    else:
                        logger.warning("No publications found on page")
                else:
                    logger.error(f"Failed to connect: Status {response.status}")
    except Exception as e:
        logger.error(f"Error during website test: {e}")

if __name__ == "__main__":
    asyncio.run(test_scraper_functionality())