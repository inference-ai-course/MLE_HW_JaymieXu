import os
import time, random
import re
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import feedparser
import requests
import trafilatura

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from PIL import Image
import pytesseract  # pip install pytesseract first

options = Options()
options.headless = True  # Run in background (no window)
options.add_argument("--headless")
options.add_argument("--headless=new")

output_dir = "cleaned_htmls"
os.makedirs(output_dir, exist_ok=True)

# arXiv API endpoint and query parameters
base_url = "https://export.arxiv.org/api/query?"
search_query = "cat:cs.CL"
start = 0
max_results = 200
sortBy = "submittedDate"
sortOrder = "descending"

# Construct the full query URL
query = (
    f"search_query={search_query}"
    f"&start={start}"
    f"&max_results={max_results}"
    f"&sortBy={sortBy}"
    f"&sortOrder={sortOrder}"
)
url = base_url + query

@dataclass
class Record:
    url: str
    title: str
    abstract: str
    authors: str
    date: str

records = []

def extract_from_ocr(ocr_text):
    def extract_abs():
        abs_start_marker = "Abstract:"
        end_marker = "References & Citations"

        abs_start = ocr_text.find(abs_start_marker)
        abs_end = ocr_text.find(end_marker)

        if abs_start == -1 or abs_end == -1 or abs_end <= abs_start:
            abstract = ""
        else:
            abstract = ocr_text[abs_start + len(abs_start_marker):abs_end].strip()

        return abstract

    # Extract date using regex to find e.g. [Submitted on 28 Jul 2025]
    date_match = re.search(r"\[Submitted on ([0-9]{1,2} [A-Za-z]{3} [0-9]{4})\]", ocr_text)
    date = date_match.group(1) if date_match else ""

    return extract_abs(), date

# Parse the Atom feed from arXiv
feed = feedparser.parse(url)

driver = webdriver.Chrome(options=options)

# Print titles and URLs for each paper
for entry in feed.entries:
    abs_url = entry.link  # This is the /abs/ page
    print(f"Fetching: {abs_url}")

    html = requests.get(abs_url).text

    cleaned = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False
    )

    metadata = trafilatura.metadata.extract_metadata(html)

    if metadata.title:
        print(f"Title: {metadata.title}")

    if cleaned:
        # Save as HTML
        paper_id = abs_url.split('/')[-1]
        html_path = os.path.join(output_dir, f"{paper_id}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            # html and css nightmare so that the text wraps and screenshot can be taken
            f.write(
                "<html><head><style>"
                "body { font-family: monospace; background: #fff; color: #222; } "
                ".content { white-space: pre-wrap; word-break: break-word; }"
                "</style></head><body>"
                f"<div class='content'>{cleaned}</div>"
                "</body></html>"
            )

        html_abs_path = Path(html_path).resolve()
        file_url = f"file:///{html_abs_path.as_posix()}"

        print(f"Saved cleaned HTML to {html_abs_path}")

        screenshotpath = os.path.join(output_dir, f"{paper_id}.png")
        driver.get(file_url)
        driver.save_screenshot(screenshotpath)
        print(f"Saved screenshot to {screenshotpath}")

        # Load an image using Pillow (PIL)
        image = Image.open(screenshotpath)
        text = pytesseract.image_to_string(image)

        #print(text)
        abstract, date = extract_from_ocr(text)
        #print(abstract)

        this_record = Record(url=abs_url, title=metadata.title, abstract=abstract, authors=metadata.author, date=date)
        records.append(this_record)
    else:
        print(f"No clean content extracted for {abs_url}")

    time.sleep(random.uniform(0.1, 0.3))

with open("arxiv_clean.json", "w", encoding="utf-8") as f:
    json.dump([asdict(r) for r in records], f, ensure_ascii=False, indent=2)

driver.quit()