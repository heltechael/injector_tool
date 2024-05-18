import csv

class EppoToPlantIdTranslator:
    def __init__(self, mapping_csv_path):
        self.eppo_to_plant_id = self._load_mapping(mapping_csv_path)

    def _load_mapping(self, csv_path):
        mapping = {}
        with open(csv_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                eppo_code = row['EPPOCode'].strip()  # Strip spaces
                plant_id = row['Id']
                mapping[eppo_code] = plant_id
        return mapping

    def translate(self, eppo_code):
        eppo_code = eppo_code.strip()  # Ensure the input EPPO code is stripped as well
        return self.eppo_to_plant_id.get(eppo_code, None)
