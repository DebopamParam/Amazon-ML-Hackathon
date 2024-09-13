import os
import numpy as np
import requests
from PIL import Image, ImageOps
from io import BytesIO
import cv2

class ImageConverter:
    def __init__(self, csv_obj, image_folder, image_size=(300, 300)):
        self.csv_obj = csv_obj
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
        try:
            # Convert to grayscale
            grayscale_image = ImageOps.grayscale(image)

            # # Resize while keeping the aspect ratio
            # grayscale_image.thumbnail(self.image_size, resample=Image.LANCZOS)

            # # Add padding to match the target size
            # padded_image = ImageOps.pad(grayscale_image, self.image_size, color="black")

            return grayscale_image
        except Exception as e:
            print(f"Failed to preprocess image: {e}")
        return None

    def process_images_from_csv(self):
        """Processes all images from the CSV object."""
        df = self.csv_obj

        for index, row in df.iterrows():
            image_url = row["image_link"]
            img = self.download_image(image_url)

            if img:
                processed_img = self.preprocess_image(img)
                # processed_img = img

                if processed_img:
                    # Convert to OpenCV format (optional, if you're working with OpenCV functions later)
                    open_cv_image = np.array(processed_img)
                    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

                    # Save the preprocessed image
                    image_path = os.path.join(
                        self.image_folder, f"processed_image_{index}.jpg"
                    )
                    cv2.imwrite(image_path, open_cv_image)
                    print(f"Saved processed image to {image_path}")