import os
import cv2
import shutil
import time

from src.utils.config_loader import ConfigLoader
from src.data_loader.data_loader import DataLoader
from src.thumbnail_selection.thumbnail_selector import ThumbnailSelector
from src.injection.injector import Injector
from src.utils.output_handler import OutputHandler
from src.utils.eppo_to_plant_id_translator import EppoToPlantIdTranslator

STORE_THUMBNAILS_LOCALLY = 0
STORE_FULL_IMAGES_LOCALLY = 0
FULL_IMAGES_NUMBER = 5000
IGIS = 1

def main():
    # Load config and create output directory structure
    config_path = 'config/config.json'

    if IGIS:
        config_path = 'config/igis_config.json'
    
    config = ConfigLoader(config_path)
    output_manager = OutputHandler(config)
    output_manager.create_output_directory()

    # Set CSV
    csv_path = config.get('csv_file')
    MAX_INJECTIONS_PER_IMAGE = config.get('MAX_INJECTIONS_PER_IMAGE')

    # Init dataloader
    dataLoader = DataLoader(config)

    # Load thumbnails for multiple classes
    thumbnail_classes = ['CIRAR', 'PIBSA', 'SPQOL', 'SOLTU', 'VICFX']
    #thumbnail_classes = ['PIBSA']
    thumbnails_by_class, load_thumbnail = dataLoader.load_thumbnails(thumbnail_classes)
    thumbnail_selector = ThumbnailSelector(config, thumbnails_by_class, load_thumbnail)
    selected_thumbnails = thumbnail_selector.select_thumbnails(thumbnail_classes)
    
    # Combine all selected thumbnails into one list
    combined_thumbnails = []
    for key in selected_thumbnails:
        combined_thumbnails.extend(selected_thumbnails[key])
    
    print(f"Total thumbnails loaded: {len(combined_thumbnails)}")

    # Initialize the EppoToPlantIdTranslator
    eppo_to_plant_id_translator = EppoToPlantIdTranslator('preprocessing/plant_info_IGIS.csv')

    start_time = time.time()

    # Inject selected thumbnails into loaded full images
    output_injected_images_dir = output_manager.get_output_path('output_injected_images_dir')
    injector = Injector(config, dataLoader, combined_thumbnails, csv_path, eppo_to_plant_id_translator)
    injector.inject_thumbnails_into_n_full_images(FULL_IMAGES_NUMBER, MAX_INJECTIONS_PER_IMAGE, output_injected_images_dir)
    #injected_images = injector.inject_thumbnails_into_n_full_images(FULL_IMAGES_NUMBER, MAX_INJECTIONS_PER_IMAGE, output_injected_images_dir)

    # Store injected images locally
    #for (injected_image, image_data) in injected_images:
        #dataLoader.store_image(injected_image, f"{output_injected_images_dir}/{os.path.splitext(image_data.filename)[0]}.jpg")

    end_time = time.time()
    print(f"Injection tool took: {(end_time-start_time)} seconds.")

if __name__ == '__main__':
    main()
