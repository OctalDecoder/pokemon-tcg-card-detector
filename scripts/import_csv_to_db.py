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

FIELD_TYPE_MAP = {
    "SeriesID": str,
    "ID": int,
    "Name": str,
    "Rarity": str,
    "EX": bool,
    "Shiny": bool,
    "Type": str,
    "HP": int,
    "Card Type": str,
    "Card Sub Type": str,
    "Evolves From": str,
    "Ability Name": str,
    "Ability Info": str,
    "Move 1 Name": str,
    "Move 1 Cost": str,
    "Move 1 Power": int,
    "Move 1 Info": str,
    "Move 2 Name": str,
    "Move 2 Cost": str,
    "Move 2 Power": int,
    "Move 2 Info": str,
    "Weakness": str,
    "Retreat": int,
    "Illustrator": str,
    "Flavor Text": str,
    "Pack": str,
    "Series": str,
}

def create_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS cards (
        series_id TEXT,
        id INT,
        name TEXT,
        rarity TEXT,
        ex BOOL,
        shiny BOOL,
        type TEXT,
        hp INT,
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
        retreat INT,
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
    
def parse_value(val, target_type):
    # Treat empty string and "-" as NULL/None
    if val in ("", "-", None):
        return None
    # Convert to proper type
    if target_type is int:
        try:
            return int(val)
        except ValueError:
            return None
    elif target_type is bool:
        # Accept "True", "true", "1", "yes" etc as True; "False" etc as False
        if str(val).strip().lower() in ("1", "true", "yes", "y"):
            return True
        elif str(val).strip().lower() in ("0", "false", "no", "n"):
            return False
        return None
    elif target_type is str:
        return str(val)
    return val

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
        data = tuple(
            parse_value(row.get(field, ""), FIELD_TYPE_MAP[field])
            for field in FIELDS_TO_USE
        ) + (image_blob,)
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
