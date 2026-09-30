import os
import requests
from zipfile import ZipFile
from tqdm import tqdm


def download_and_extract(url, filename="FloodNet.zip"):
    print("Starting download")
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))

        with open(filename, "wb") as file:
            with tqdm(total=total, unit="B", unit_scale=True) as bar:
                for chunk in response.iter_content(1024 * 1024):
                    file.write(chunk)
                    bar.update(len(chunk))

    with ZipFile(filename) as archive:
        files = archive.infolist()

        for file in tqdm(files, desc="Extracting", unit="file"):
            archive.extract(file, "FloodNet")

    print(f"Downloading and Extraction of {filename} is complete")

if __name__ == "__main__":
    url = os.environ.get("FLOODSIGHT_DATASET_URL")
    if not url:
        raise SystemExit("Set FLOODSIGHT_DATASET_URL to the dataset archive URL before running this script.")
    download_and_extract(url)
