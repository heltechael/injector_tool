import csv

def process_csv(input_file, output_file):
    with open(input_file, mode='r', newline='') as infile, open(output_file, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = ['EPPOcode', 'uploadId', 'imageId', 'annotationId', 'pred_label_eppo', 'confidence_score']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            # del unnecessary part of image path
            filename = row['image_path'].split('/')[-1]
            
            parts = filename.split('_')
            eppo_code = parts[0]
            upload_id = parts[1]
            image_id = parts[2]
            annotation_id = parts[3].split('.')[0]  # Remove the .png part
            
            writer.writerow({
                'EPPOcode': eppo_code,
                'uploadId': upload_id,
                'imageId': image_id,
                'annotationId': annotation_id,
                'pred_label_eppo': row['pred_label_eppo'],
                'confidence_score': row['confidence_score']
            })

input_file = 'PREDICTED_LABELS_20FEB_200epoch_model_TRAINING_DATA_80000.csv'
output_file = 'Predicted_labels_rwm_approved.csv'
process_csv(input_file, output_file)
