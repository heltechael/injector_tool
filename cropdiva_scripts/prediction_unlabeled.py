import os
import json
import glob
import pandas as pd
import numpy as np
import skimage.io
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.efficientnet import preprocess_input
import tqdm

def load_model_and_config(network_folder_path):
    # Load config
    with open(os.path.join(network_folder_path, 'configuration.json'),'r') as fob:
        train_config = json.load(fob)
    model = load_model(os.path.join(network_folder_path, 'best_model_checkpoint'))
    return model, train_config

def create_dataframe_from_images(image_root_folder):
    image_paths = []
    labels = []
    for root, dirs, files in os.walk(image_root_folder):
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg')):
                image_paths.append(os.path.join(root, file))
                labels.append(os.path.basename(root))  # Assuming the folder name is the label
    df = pd.DataFrame({'image_path': image_paths, 'label': labels})
    return df

def preprocess_images(df, image_size=(224, 224)):
    images = []
    for img_path in tqdm.tqdm(df['image_path'], desc='Preprocessing images'):
        img = load_img(img_path, target_size=image_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        images.append(img_array)
    images = np.array(images)
    return images

def make_predictions(model, images):
    predictions = model.predict(images, batch_size=32)
    predicted_labels = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)
    return predicted_labels, confidence_scores

def save_predictions_to_csv(df, predicted_labels, confidence_scores, labels_dict, output_csv_path):
    # Map numerical labels back to EPPo
    inv_labels_dict = {v: k for k, v in labels_dict.items()}
    pred_label_eppo = [inv_labels_dict[label] for label in predicted_labels]
    df['pred_label'] = pred_label_eppo
    df['confidence_score'] = confidence_scores
    df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

def main(network_folder_path, image_root_folder, output_csv_path):
    model, train_config = load_model_and_config(network_folder_path)
    dataset_folder = 'Datasets'
    labels_dict_file = os.path.join(dataset_folder, train_config['dataset_id'], 'labels_dict_' + train_config['dataset_id'] + '.json')
    with open(labels_dict_file, 'r') as fob:
        labels_dict = json.load(fob)
    
    df = create_dataframe_from_images(image_root_folder)
    images = preprocess_images(df, image_size=(224, 224))
    predicted_labels, confidence_scores = make_predictions(model, images)
    save_predictions_to_csv(df, predicted_labels, confidence_scores, labels_dict, output_csv_path)

if __name__ == "__main__":
    network_folder_path = 'networks/EfficientNetV2S_imagenet_224x224_avg_adam_lr0.0001_bs32_22APR_Approved_MIMAD_TEST__7880845c'
    image_root_folder = '../TrainingData/PREDICTION_TEST'
    output_csv_path = os.path.join('predictions_on_new_images.csv')
    
    main(network_folder_path, image_root_folder, output_csv_path)
