import os
import cv2
import shutil

from src.utils.config_loader import ConfigLoader
from src.data_loader.data_loader import DataLoader
from src.thumbnail_selection.thumbnail_selector import ThumbnailSelector
from src.injection.injector import Injector
from src.utils.output_handler import OutputHandler

STORE_THUMBNAILS_LOCALLY = 0
STORE_FULL_IMAGES_LOCALLY = 0
FULL_IMAGES_NUMBER = 10

def main():
    # Load config and create output directory structure
    config_path = 'config/config.json'
    config = ConfigLoader(config_path)
    output_manager = OutputHandler(config)
    output_manager.create_output_directory()

    # Set CSV
    csv_path = config.get('csv_file')
    MAX_INJECTIONS_PER_IMAGE = config.get('MAX_INJECTIONS_PER_IMAGE')

    # Init dataloader
    dataLoader = DataLoader(config)

    # Load full images
    full_images = dataLoader.load_full_images()
    print(f"Amount of loaded full_images: {len(full_images)}")
    
    # Load thumbnails
    thumbnails_by_class, load_thumbnail = dataLoader.load_thumbnails()
    thumbnail_selector = ThumbnailSelector(config, thumbnails_by_class, load_thumbnail)
    selected_thumbnails = thumbnail_selector.select_thumbnails(['1CHEG'])
    print(f"Amount of loaded thumbnails: {len(selected_thumbnails['1CHEG'])}")

    # Select thumbnails to inject
    thumbdata = selected_thumbnails["1CHEG"]

    # Inject selected thumbnails into loaded full images
    output_injected_images_dir = output_manager.get_output_path('output_injected_images_dir')
    injector = Injector(config, full_images, thumbdata, csv_path)
    injected_images = injector.inject_thumbnails_into_n_full_images(FULL_IMAGES_NUMBER, MAX_INJECTIONS_PER_IMAGE, output_injected_images_dir)

    # Store injected images locally
    for (injected_image, image_data) in injected_images:
        dataLoader.store_image(injected_image, f"{output_injected_images_dir}/{os.path.splitext(image_data.filename)[0]}.png")

if __name__ == '__main__':
    main()