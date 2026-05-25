import os
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import time

def scrape_and_save(url, subject="astronomy"):
    """
    Scrapes a single article, extracts its text, and saves it to a text file
    along with a metadata JSON file.
    
    This version finds all paragraphs with more than 30 characters.
    """
    try:
        headers = {
            "User-Agent": "AstronomyRAGBot/1.0 (YourContactInfo; +http://your-project-url.com)"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Raises an HTTPError for bad responses

        soup = BeautifulSoup(response.content, 'html.parser')

        title = soup.find('h1').get_text(strip=True) if soup.find('h1') else "untitled"
        
        # 1. Find all paragraph tags in the entire document.
        all_paragraphs = soup.find_all('p')
        
        # 2. Filter them based on character length.
        filtered_content = [
            p.get_text(strip=True) 
            for p in all_paragraphs 
            if len(p.get_text(strip=True)) > 30
        ]
        
        # 3. Join the filtered paragraphs into a single string.
        content = "\n".join(filtered_content)
        
        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        if not content:
            print(f"Warning: No paragraphs with > 30 chars found on {url}. Skipping.")
            return

        # --- File and Metadata Handling (unchanged) ---
        safe_filename = re.sub(r'[^a-zA-Z0-9]', '_', url.split('/')[-1] or url.split('/')[-2])
        txt_path = os.path.join("docs", f"{safe_filename}.txt")
        meta_path = os.path.join("docs", f"{safe_filename}.meta.json")

        # Save content
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Save metadata
        metadata = {
            "source": url,
            "subject": subject,
            "title": title,
            "timestamp": time.strftime("%Y-m-%dT%H:%M:%SZ", time.gmtime())
        }
        # with open(meta_path, 'w', encoding='utf-8') as f:
        #     json.dump(metadata, f, indent=4)
            
        print(f"Successfully scraped and saved: {title}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred for {url}: {e}")

if __name__ == "__main__":
    if not os.path.exists("docs"):
        os.makedirs("docs")


    urls_to_scrape = [
        "https://science.nasa.gov/universe/stars/",
        "https://science.nasa.gov/universe/galaxies/",
        "https://science.nasa.gov/universe/black-holes/",
        "https://science.nasa.gov/universe/overview/",
        "https://science.nasa.gov/universe/overview/building-blocks/",
        "https://science.nasa.gov/universe/overview/forces/",
        "https://science.nasa.gov/universe/galaxies/types/",
        "https://science.nasa.gov/universe/galaxies/evolution/",
        "https://science.nasa.gov/universe/galaxies/large-scale-structures/",
        "https://science.nasa.gov/universe/black-holes/types/",
        "https://science.nasa.gov/universe/black-holes/anatomy/",
        "https://science.nasa.gov/universe/stars/types/",
        "https://science.nasa.gov/universe/stars/multiple-star-systems/",
        "https://science.nasa.gov/universe/stars/planetary-system/",
        "https://science.nasa.gov/exoplanets/",
        "https://science.nasa.gov/universe/sensing-the-universe/",
        "https://science.nasa.gov/universe/telescopes-101/",
        "https://science.nasa.gov/universe/observatories/",
        "https://science.nasa.gov/universe/whats-a-nova-inside-the-chaos-of-erupting-and-exploding-stars/",
        "https://science.nasa.gov/dark-energy/",
        "https://science.nasa.gov/universe/gamma-ray-bursts-harvesting-knowledge-from-the-universes-most-powerful-explosions/",
        "https://science.nasa.gov/universe/what-is-betelgeuse-inside-the-strange-volatile-star/",
        "https://science.nasa.gov/universe/what-happens-when-something-gets-too-close-to-a-black-hole/"
    ]

    for url in urls_to_scrape:
        scrape_and_save(url)
        time.sleep(1) # Be polite and wait 1 second between requests