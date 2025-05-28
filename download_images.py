import os
import csv
import requests
from io import BytesIO
from PIL import Image

def download_and_convert_images():
    # Ensure the images directory exists
    os.makedirs('images', exist_ok=True)

    # Loop through all files in the current directory
    for filename in os.listdir('.'):
        if filename.lower().endswith('.csv'):
            with open(filename, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                headers = reader.fieldnames or []

                # Skip files without an ImageURL column
                if 'ImageURL' not in headers:
                    continue

                # The first header is the ID column
                id_header = headers[0]
                base_name = os.path.splitext(filename)[0]

                # Process each row in the CSV
                for row in reader:
                    image_url = row['ImageURL']
                    id_val = row[id_header]

                    try:
                        # Download the image
                        response = requests.get(image_url)
                        response.raise_for_status()

                        # Convert from WebP to PNG
                        img = Image.open(BytesIO(response.content)).convert('RGBA')

                        # Construct the output filename
                        output_filename = f"{base_name} {id_val}.png"
                        output_path = os.path.join('images', output_filename)

                        # Save as PNG
                        img.save(output_path, format='PNG')
                        print(f"Saved: {output_path}")

                    except Exception as e:
                        print(f"Error processing {image_url}: {e}")

if __name__ == '__main__':
    download_and_convert_images()
