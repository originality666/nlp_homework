"""
scraper.py:
    Collect Chinese and English sample texts from the internet using a web crawler.
    Please prepare the seed_urls.json file in advance, which contains a list of URLs to be crawled.
"""

import requests  # Used to send HTTP requests to obtain web content
from bs4 import BeautifulSoup  # Used to parse HTML and extract text
import os, json, time, re  # Respectively used for file operations, JSON processing, delay control, and regular expressions
import argparse  # For parsing command-line arguments

def clean_text(html: str) -> str:
    """Clean HTML to plain text""" 
    soup = BeautifulSoup(html, "lxml")  # Parse HTML with lxml parser   
    # Remove unnecessary tags (scripts, styles, noscript, headers, footers, forms, etc.)
    for s in soup(["script", "style", "noscript", "header", "footer", "form"]):
        s.decompose()  # Delete these tags from HTML    
    # Extract text, separate content from different tags with newlines
    text = soup.get_text(separator="\n")   
    # Use regular expressions to replace consecutive whitespace characters (spaces, newlines, etc.) with a single space
    text = re.sub(r"\s+", " ", text)   
    return text.strip()  # Remove leading and trailing spaces and return

OUT_DIR = "collected_samples"  
os.makedirs(OUT_DIR, exist_ok=True)  # Create the folder, no error if it already exists

def scrape(type = "english"):
    """Crawl and save sample texts from the internet"""
    if type == "chinese":
        url_file = "chinese_seed_urls.json"
    else:
        url_file = "english_seed_urls.json"
    # Read URL list
    with open(url_file, "r", encoding="utf-8") as f:
        urls = json.load(f)  
    # Set request headers to simulate browser behavior and avoid being identified as a crawler by the website
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; CASIA-EntropyBot/1.0)"
    }  
    for i, url in enumerate(urls):  # Traverse the URL list, i is the index, url is the current URL
        try:
            print(f"[{i+1}/{len(urls)}] Fetching: {url}")  # Print progress information
            # Send a GET request to obtain web content, set timeout to 15 seconds
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()  # Throw an exception if the request fails (e.g., 404, 500)
            # Clean HTML text
            text = clean_text(r.text)
            # Save the cleaned text to a local file
            with open(os.path.join(OUT_DIR, f"{type}_sample_{i+1}.txt"), "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Saved {len(text)} characters")  # Print the number of saved characters
            time.sleep(1)  # Sleep for 1 second to avoid excessive request frequency
        
        except Exception as e:  # Catch all exceptions
            print(f"Error on {url}: {e}")  # Print error information and continue processing the next URL

if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="collect Chinese and English sample texts from the internet")
    parser.add_argument(
        "--type", 
        default="chinese", 
        help="Type of text to scrape: 'chinese' or 'english' (default: 'chinese')"
    )
    # Parse arguments
    args = parser.parse_args()
    # Start crawling
    scrape(args.type)

