import re
import constants
import os
import requests
import pandas as pd
import multiprocessing
import time
from time import time as timer
from tqdm import tqdm
import numpy as np
from pathlib import Path
from functools import partial
import requests
import urllib
from PIL import Image, ImageOps
from io import BytesIO
import cv2

class ImageConverter:
    def __init__(self, csv_path, image_folder, image_size=(500, 500)):
        self.csv_path = csv_path
        self.image_folder = image_folder
        self.image_size = image_size

        # Create the image folder if it doesn't exist
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

    def download_image(self, url, timeout=10):
        """Downloads an image from a URL with a specified timeout."""
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            return Image.open(BytesIO(response.content))
        except requests.exceptions.RequestException as e:
            print(f"Failed to download image from {url}: {e}")
        return None

    def preprocess_image(self, image):
        """Preprocess the image by converting to grayscale, resizing, and adding padding."""
        # Convert to grayscale
        grayscale_image = ImageOps.grayscale(image)

        # Resize while keeping the aspect ratio
        grayscale_image.thumbnail(self.image_size, resample=Image.LANCZOS)

        # Add padding to match the target size
        padded_image = ImageOps.pad(grayscale_image, self.image_size, color='black')

        return padded_image

    def process_images_from_csv(self):
        """Processes all images from the CSV."""
        df = pd.read_csv(self.csv_path)
        df = df.head(200)

        for index, row in df.iterrows():
            image_url = row['image_link']
            img = self.download_image(image_url)

            if img:
                processed_img = self.preprocess_image(img)

                # Convert to OpenCV format (optional, if you're working with OpenCV functions later)
                open_cv_image = np.array(processed_img)
                open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

                # Save the preprocessed image
                image_path = os.path.join(self.image_folder, f'processed_image_{index}.jpg')
                cv2.imwrite(image_path, open_cv_image)
                print(f"Saved processed image to {image_path}")


def common_mistake(unit):
    if unit in constants.allowed_units:
        return unit
    if unit.replace('ter', 'tre') in constants.allowed_units:
        return unit.replace('ter', 'tre')
    if unit.replace('feet', 'foot') in constants.allowed_units:
        return unit.replace('feet', 'foot')
    return unit

def parse_string(s):
    s_stripped = "" if s==None or str(s)=='nan' else s.strip()
    if s_stripped == "":
        return None, None
    pattern = re.compile(r'^-?\d+(\.\d+)?\s+[a-zA-Z\s]+$')
    if not pattern.match(s_stripped):
        raise ValueError("Invalid format in {}".format(s))
    parts = s_stripped.split(maxsplit=1)
    number = float(parts[0])
    unit = common_mistake(parts[1])
    if unit not in constants.allowed_units:
        raise ValueError("Invalid unit [{}] found in {}. Allowed units: {}".format(
            unit, s, constants.allowed_units))
    return number, unit


def create_placeholder_image(image_save_path):
    try:
        placeholder_image = Image.new('RGB', (100, 100), color='black')
        placeholder_image.save(image_save_path)
    except Exception as e:
        return

def download_image(image_link, save_folder, retries=3, delay=3):
    if not isinstance(image_link, str):
        return

    filename = Path(image_link).name
    image_save_path = os.path.join(save_folder, filename)

    if os.path.exists(image_save_path):
        return

    for _ in range(retries):
        try:
            urllib.request.urlretrieve(image_link, image_save_path)
            return
        except:
            time.sleep(delay)
    
    create_placeholder_image(image_save_path) #Create a black placeholder image for invalid links/images

def download_images(image_links, download_folder, allow_multiprocessing=True):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    if allow_multiprocessing:
        download_image_partial = partial(
            download_image, save_folder=download_folder, retries=3, delay=3)

        with multiprocessing.Pool(64) as pool:
            list(tqdm(pool.imap(download_image_partial, image_links), total=len(image_links)))
            pool.close()
            pool.join()
    else:
        for image_link in tqdm(image_links, total=len(image_links)):
            download_image(image_link, save_folder=download_folder, retries=3, delay=3)
        