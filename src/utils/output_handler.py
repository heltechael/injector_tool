import os

class OutputHandler:
    def __init__(self, config):
        self.config = config
        self.output_dir = config.get('output_dir')

    def create_output_directory(self):
        injection_num = 1
        while os.path.exists(f"{self.output_dir}/injection{injection_num}"):
            injection_num += 1
        injection_dir = f"{self.output_dir}/injection{injection_num}"
        os.makedirs(injection_dir)

        output_bounding_boxes_dir = f"{injection_dir}/bounding_boxes"
        output_injected_images_dir = f"{injection_dir}/injected_images"
        output_csv_file = f"{injection_dir}/filtered_annotations.csv"

        os.makedirs(output_bounding_boxes_dir)
        os.makedirs(output_injected_images_dir)

        self.config.config['output_bounding_boxes_dir'] = output_bounding_boxes_dir
        self.config.config['output_injected_images_dir'] = output_injected_images_dir
        self.config.config['output_csv_file'] = output_csv_file

    def get_output_path(self, key):
        return self.config.get(key)