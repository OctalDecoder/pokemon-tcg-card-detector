import os
import csv
import sqlite3
import requests
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

CSV_DIR = "scripts/data/"
DB_PATH = "models/cards.db"

FIELDS_TO_USE = [
    "SeriesID", "ID", "Name", "Rarity", "EX", "Shiny", "Type", "HP", "Card Type", "Card Sub Type",
    "Evolves From", "Ability Name", "Ability Info", "Move 1 Name", "Move 1 Cost", "Move 1 Power",
    "Move 1 Info", "Move 2 Name", "Move 2 Cost", "Move 2 Power", "Move 2 Info",
    "Weakness", "Retreat", "Illustrator", "Flavor Text", "Pack", "Series"
]

def create_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS cards (
        series_id TEXT,
        id TEXT,
        name TEXT,
        rarity TEXT,
        ex TEXT,
        shiny TEXT,
        type TEXT,
        hp TEXT,
        card_type TEXT,
        card_sub_type TEXT,
        evolves_from TEXT,
        ability_name TEXT,
        ability_info TEXT,
        move1_name TEXT,
        move1_cost TEXT,
        move1_power TEXT,
        move1_info TEXT,
        move2_name TEXT,
        move2_cost TEXT,
        move2_power TEXT,
        move2_info TEXT,
        weakness TEXT,
        retreat TEXT,
        illustrator TEXT,
        flavor_text TEXT,
        pack TEXT,
        series TEXT,
        image_blob BLOB,
        PRIMARY KEY (series_id, id)
    )
    ''')
    conn.commit()
    conn.close()

def download_and_pack(row):
    try:
        image_blob = None
        if row.get("Image URL"):
            r = requests.get(row["Image URL"], timeout=10)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content))
            buf = BytesIO()
            img.save(buf, format='png')
            image_blob = buf.getvalue()
        # Tuple must match number of columns in DB table
        data = tuple(row.get(field, "") for field in FIELDS_TO_USE) + (image_blob,)
        if len(data) != 28:
            print(f"Skipping row {row.get('Name')} due to wrong column count: {len(data)}")
            return None
        return data
    except Exception as e:
        print(f"Failed to process row {row.get('Name','?')}: {e}")
        return None

def import_csvs():
    # Collect all rows from all csv files
    rows = []
    for fname in os.listdir(CSV_DIR):
        if not fname.endswith(".csv"):
            continue
        with open(os.path.join(CSV_DIR, fname), encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Series") == "Promo-A":
                    row["SeriesID"] = "P-A"
                rows.append(row)
    # Multithreaded image download and data packaging
    data_list = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_and_pack, row) for row in rows]
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                data_list.append(result)
    # Insert into DB in one go
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executemany('''
        INSERT OR REPLACE INTO cards VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        )
    ''', data_list)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_db()
    import_csvs()
    print("Import complete!")
