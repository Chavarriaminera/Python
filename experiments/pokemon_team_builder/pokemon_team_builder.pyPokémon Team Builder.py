import requests
from bs4 import BeautifulSoup
import os
import csv
import re
import time 
import pandas as pd
# Inteleon Picks: AI-Powered Pokémon Team Builder

"""

 What this Current Script Does:
Checks if Smogon is reachable.

Finds all available months for 2025.

Downloads VGC usage files (for standard and Bo3 across all skill ratings).

Also downloads relevant VGC files from each category (leads, metagame, moveset, monotype, chaos).

Organizes files by month, then by category (like folders: 2025-03/vgc/, 2025-03/leads/, etc.).

Goals:
1. Identify Pokémon legal in the current regulation that maximize winning potential, and suggest improvements to a given partial team (e.g., suggest alternatives or better teammates).
2. Generate full 6-Pokémon teams that have the highest possible win probability across meta matchups.
3. Analyze the meta to find weaknesses and suggest underused or low-usage Pokémon that could exploit those weaknesses.
4. Suggest a "trump card" or surprise pick to catch opponents off guard.

Potential ML Components:
- Supervised learning for win prediction based on team composition
- Clustering/meta analysis based on common teams in tournament data
- Recommendation system for team building (e.g., based on synergy, role coverage, and meta threats)

Data Needed:
- Smogon/Showdown or tournament battle logs (with outcomes)
- Legal Pokémon for current regulation
- Meta usage stats (from sources like Pikalytics)

Planned Workflow:
1. Scrape or load data from VGC tournaments and Pikalytics
2. Create a Pokémon feature dataset (type, base stats, common items, roles, usage)
3. Build team composition vectors
4. Train models to predict win probability
5. Suggest optimal teammates for a given team
6. Explore surprising picks based on meta gaps
"""
""

""

"""Smogon VGC Data Scraper and Cleaner
This script downloads monthly VGC usage data from Smogon, including leads, moveset,
metagame, chaos, and monotype categories, and saves them as cleaned CSV files.
"""





"""Smogon VGC Data Scraper and Cleaner
This script downloads monthly VGC usage data from Smogon, including leads, moveset,
metagame, chaos, and monotype categories, and saves them as cleaned CSV files.
"""

"""Smogon VGC Data Scraper and Cleaner
This script downloads monthly VGC usage data from Smogon, including leads, moveset,
metagame, chaos, and monotype categories, and saves them as cleaned CSV files.
"""

import requests
from bs4 import BeautifulSoup
import os
import csv
import re
import time

# === CONFIGURATION SECTION ===
VGC_FILENAMES = [
    "gen9vgc2025regg-0.txt",
    "gen9vgc2025regg-1500.txt",
    "gen9vgc2025regg-1630.txt",
    "gen9vgc2025regg-1760.txt",
    "gen9vgc2025reggbo3-0.txt",
    "gen9vgc2025reggbo3-1500.txt",
    "gen9vgc2025reggbo3-1630.txt",
    "gen9vgc2025reggbo3-1760.txt"
]

BASE_DIR = r"C:\Users\chava\OneDrive\Documents\Pokemon\List of Pokemon"
BASE_DOWNLOAD_DIR = os.path.join(BASE_DIR, "csv_files")
ADDITIONAL_FOLDERS = ["leads", "metagame", "moveset", "monotype", "chaos", "vgc"]
os.makedirs(BASE_DOWNLOAD_DIR, exist_ok=True)

# === STEP 0: Check if Smogon is reachable ===
print("\U0001F50D Checking if Smogon is reachable...")
try:
    test_response = requests.get("https://www.smogon.com/stats/", timeout=10)
    if test_response.status_code == 200:
        print("✅ Smogon site is reachable! Proceeding with scraping.\n")
    else:
        print(f"❌ Site returned status code: {test_response.status_code}")
        exit()
except requests.exceptions.RequestException as e:
    print(f"❌ Could not reach Smogon: {e}")
    exit()

# === STEP 1: Get available 2025 months ===
print("\U0001F4C5 Scraping available months from Smogon...")
BASE_URL = "https://www.smogon.com/stats/"
response = requests.get(BASE_URL)
soup = BeautifulSoup(response.text, "html.parser")
month_links = [link.get("href") for link in soup.find_all("a") if link.get("href").startswith("2025-")]

# === STEP 2: Download and convert TXT to CSV ===
for month in month_links:
    month_folder = os.path.join(BASE_DOWNLOAD_DIR, month.strip("/"))
    os.makedirs(month_folder, exist_ok=True)

    # A. Download VGC Usage Files
    vgc_folder = os.path.join(month_folder, "vgc")
    os.makedirs(vgc_folder, exist_ok=True)
    for filename in VGC_FILENAMES:
        file_url = f"{BASE_URL}{month}{filename}"
        save_path = os.path.join(vgc_folder, filename.replace(".txt", ".csv"))
        if os.path.exists(save_path):
            print(f"📁 Already downloaded: {filename}")
            continue
        print(f"⬇️ Downloading: {file_url}")
        try:
            resp = requests.get(file_url)
            if resp.status_code == 200:
                lines = resp.text.splitlines()
                with open(save_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for line in lines:
                        parts = [p.strip() for p in line.split("|") if p.strip()]
                        if parts:
                            writer.writerow(parts)
                print(f"✅ Saved to: {save_path}")
            else:
                print(f"❌ Not found or not available yet: {file_url}")
        except Exception as e:
            print(f"⚠️ Error downloading {file_url}: {e}")
        time.sleep(1)

    # B. Download Additional Folder Files
    for folder in ADDITIONAL_FOLDERS:
        if folder == "vgc":
            continue  # Skip, already handled above
        sub_url = f"{BASE_URL}{month}{folder}/"
        sub_folder = os.path.join(month_folder, folder)
        os.makedirs(sub_folder, exist_ok=True)
        try:
            sub_resp = requests.get(sub_url)
            sub_soup = BeautifulSoup(sub_resp.text, "html.parser")
            sub_links = [link.get("href") for link in sub_soup.find_all("a") if any(name in link.get("href") for name in VGC_FILENAMES)]
            for subfile in sub_links:
                file_url = f"{sub_url}{subfile}"
                save_path = os.path.join(sub_folder, subfile.replace(".txt", ".csv"))
                if os.path.exists(save_path):
                    print(f"📁 Already downloaded: {subfile}")
                    continue
                print(f"⬇️ Downloading: {file_url}")
                try:
                    resp = requests.get(file_url)
                    if resp.status_code == 200:
                        lines = resp.text.splitlines()
                        with open(save_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            for line in lines:
                                parts = [p.strip() for p in line.split("|") if p.strip()]
                                if parts:
                                    writer.writerow(parts)
                        print(f"✅ Saved to: {save_path}")
                    else:
                        print(f"❌ Not found or not available yet: {file_url}")
                except Exception as e:
                    print(f"⚠️ Error downloading {file_url}: {e}")
                time.sleep(1)
        except Exception as e:
            print(f"⚠️ Error accessing folder {folder}: {e}")

print("\n🧼 Cleaning all leads/metagame/moveset/monotype/chaos/vgc CSV files...")

# === STEP 3: Clean all additional folders CSVs ===
CLEAN_HEADERS = {
    "vgc": ["Rank", "Pokemon", "Usage", "%", "Real", "%"],
    "leads": ["Rank", "Pokemon", "Usage %", "Raw", "%"],
    "metagame": ["Format", "Battle Count", "Avg. Weight Team"],
    "moveset": ["Pokemon", "Move", "Usage %"],
    "monotype": ["Rank", "Pokemon", "Usage %", "Raw", "Raw %"],
    "chaos": None  # we won’t clean chaos for now
}

for month in os.listdir(BASE_DOWNLOAD_DIR):
    month_path = os.path.join(BASE_DOWNLOAD_DIR, month)
    if not os.path.isdir(month_path):
        continue
    for folder in ADDITIONAL_FOLDERS:
        folder_path = os.path.join(month_path, folder)
        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if not filename.endswith(".csv"):
                continue
            file_path = os.path.join(folder_path, filename)
            cleaned_rows = []

            with open(file_path, "r", encoding="utf-8") as infile:
                reader = csv.reader(infile)
                for row in reader:
                    if not row or any(x in row[0] for x in ["----", "Total", "Rank", "Format", "Avg. weight/team"]):
                        continue
                    cleaned_rows.append(row)

            # Skip chaos since its format is unpredictable
            if folder == "chaos":
                continue

            with open(file_path, "w", newline="", encoding="utf-8") as outfile:
                writer = csv.writer(outfile)
                header = CLEAN_HEADERS.get(folder)
                if header:
                    writer.writerow(header)
                writer.writerows(cleaned_rows)

print("✅ All data downloaded and cleaned!")
