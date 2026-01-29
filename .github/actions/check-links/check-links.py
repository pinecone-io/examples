#! /usr/bin/env python

# Check links in a notebook

import re
import os
import sys
import json
import nbformat
import requests

# Get the notebook filename from the command line
filename = "../../../" + sys.argv[1]
print(f"Processing notebook: {filename}")
nb_source_path = os.path.join(os.path.dirname(__file__), filename)

known_good = [
    "https://www.pinecone.io",
    "https://app.pinecone.io",
]

ignore_links = [
    'platform.openai.com', # cloudflare blocks requests sometimes
    'colab.research.google.com', # cloudflare blocks requests sometimes
    'quora.com', # cloudflare blocks requests sometimes
    'nbviewer.org' # nbviewer has a pretty strict rate limit, so we don't want to waste requests
    'app.pinecone.io', # we don't need to spam our own homepage
]

known_good_links = set(known_good)
for link in known_good:
    known_good_links.add(f"{link}/")

# Read the notebook
with open(nb_source_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

    try:
        good_links = set()
        failed_links = set()
        links = set()  # Use set to avoid duplicates
        
        # URL regex pattern - updated to handle markdown links better
        url_pattern = r'https?://[^\s<>"\)]+|www\.[^\s<>"\)]+'
        
        # Search through all cells
        for cell in nb['cells']:
            if 'source' in cell:
                # Join multi-line source into single string
                content = ''.join(cell['source'])
                # Find all URLs
                found_links = re.findall(url_pattern, content)
                links.update(found_links)
        
        if links:
            print(f"\nFile: {filename}")
            for link in sorted(links):
                if any(ignore_link in link for ignore_link in ignore_links):
                    print(f"  ⏭️ {link}")
                    continue
                if link in known_good_links:
                    good_links.add(link)
                    print(f"  ✅ {link}")
                    continue
                elif link in good_links:
                    continue
                elif link in failed_links:
                    continue
                else:
                    try:
                        response = requests.head(link, timeout=10)
                        if response.status_code == 405:
                            # Not all links can be checked with HEAD, so we fall back to GET
                            response = requests.get(link, timeout=10)
                        
                        if 200 <= response.status_code < 400:
                            good_links.add(link)
                            print(f"  ✅ {link}")
                        else:
                            failed_links.add(link)
                            print(f"  ❌ {response.status_code} {link}")
                    except Exception as e:
                        failed_links.add(link)
                        print(f"  ❌ {link}")

        print(f"Found {len(links)} links")
        print(f"Good links: {len(good_links)}")
        
        if len(failed_links) > 0:
            print("Failed links:")
            for link in sorted(failed_links):
                print(f"  ❌ {link}")
            sys.exit(1)
        else:
            print("No bad links found")
            sys.exit(0)

    except json.JSONDecodeError:
        print(f"Error: Could not parse {filename} as JSON")
    except KeyError:
        print(f"Error: Unexpected notebook format in {filename}")