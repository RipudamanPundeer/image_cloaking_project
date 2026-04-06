import os
import random
import shutil


def copy_random_images(source_dir, dest_dir, num_images):
    # Ensure destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created destination directory: {dest_dir}")

    # List all files and filter for common image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    all_images = [f for f in os.listdir(source_dir)
                  if f.lower().endswith(valid_extensions)]

    # Check if there are enough images to copy
    if not all_images:
        print("No images found in the source directory.")
        return

    # Adjust number to copy if requested amount exceeds available images
    num_to_copy = min(len(all_images), num_images)

    # Select random images
    selected_images = random.sample(all_images, num_to_copy)

    # Copy selected images
    for image_name in selected_images:
        src_path = os.path.join(source_dir, image_name)
        dst_path = os.path.join(dest_dir, image_name)
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {image_name}")

    print(f"Successfully copied {num_to_copy} images.")


# Configuration
SOURCE = r".\benchmark"
DESTINATION = r".\subset"
COUNT = 500  # Number of images to copy

copy_random_images(SOURCE, DESTINATION, COUNT)
