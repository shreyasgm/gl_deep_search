"""
Minimal test for scraper functionality - yet to learn some of the packages
"""

#async programming - maybe I just use dask?
import asyncio

#stuff
import logging
from pathlib import Path

#async http
import aiohttp

#read config
import yaml

#usual web scraping
from bs4 import BeautifulSoup

#setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_scraper_functionality():    
    #unsure if this is the way to do it
    config_path = Path(__file__).parent.parent / "config.yaml"
    
    #check once - had to skip one directory, maybe we can move this script itself down to dev
    logger.info(f"Looking for config at: {config_path.absolute()}")
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            #use the inputs for this script
            base_url = config["sources"]["growth_lab"]["base_url"]
            logger.info(f"Successfully loaded config, base_url: {base_url}")
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        # Fall back to hardcoded URL if config loading fails
        base_url = "https://growthlab.hks.harvard.edu/publications"
        logger.info(f"Using fallback base_url: {base_url}")
    
    #test connection -0 used to get a chrome headless machine error
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url) as response:
                if response.status == 200:
                    logger.info(f"Successfully connected to {base_url}")
                    
                    #wait for response and parse the HTML
                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")
                    
                    #first pub
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