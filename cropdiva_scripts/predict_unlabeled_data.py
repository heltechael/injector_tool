import os
import json
import numpy as np
import pandas as pd
import gc
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.efficientnet import preprocess_input
from tqdm import tqdm
from scipy.special import softmax
import tensorflow as tf

# Load model function
def load_model(filepath):
    return tf.keras.models.load_model(filepath)

# Prepare image function
def prepare_image(image_path, image_size):
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Predict on images function
def predict_on_images(model, image_paths, image_size):
    predictions = []
    confidence_scores = []
    processed_paths = []
    
    for image_path in tqdm(image_paths, desc='Predicting on images'):
        image = prepare_image(image_path, image_size)
        pred = model.predict(image)
        
        # Apply softmax to get probabilities
        pred_prob = softmax(pred, axis=1)

        predictions.append(pred_prob)
        
        confidence = np.max(pred_prob, axis=1)
        confidence_scores.append(confidence)
        processed_paths.append(image_path)
        
    predictions = np.concatenate(predictions, axis=0)
    confidence_scores = np.concatenate(confidence_scores, axis=0)
    pred_labels = np.argmax(predictions, axis=1)
    
    return processed_paths, pred_labels, confidence_scores

# Get all image paths function
def get_all_image_paths(root_folder):
    image_paths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(dirpath, filename))
    return image_paths

# Generator to get batches of paths
def batch_generator(image_paths, batch_size):
    for i in range(0, len(image_paths), batch_size):
        yield image_paths[i:i + batch_size]

if __name__ == "__main__":
    network_folder_path = 'networks/EfficientNetV2S_imagenet_224x224_avg_adam_lr0.0001_bs32_21MAY_Approved_MIMAD_TEST_200Epoch__76ba2866'
    dataset_folder = 'Datasets'
    #subdataset = 'Test'
    trainingdata_folder = '../TrainingData/thumbnails_output_24MAY'
    output_csv = 'PREDICTED_LABELS_24MAY_200epoch_model_TRAINING_DATA_80000.csv'
    image_size = (224, 224)
    batch_size = 1000

    model_path = os.path.join(network_folder_path, 'best_model_checkpoint')
    model = load_model(model_path)
    
    # Load labels dict from CORRECT set
    train_config_file = os.path.join(network_folder_path, 'configuration.json')
    with open(train_config_file, 'r') as fob:
        train_config = json.load(fob)
    labels_dict_file = os.path.join(dataset_folder, train_config['dataset_id'], 'labels_dict_' + train_config['dataset_id'] + '.json')
    with open(labels_dict_file, 'r') as fob:
        labels_dict = json.load(fob)
    
    # Get all image paths from subfolders
    image_paths = get_all_image_paths(trainingdata_folder)
    
    if os.path.exists(output_csv):
        os.remove(output_csv)
    
    # Process and save predictions in batches
    for batch_paths in batch_generator(image_paths, batch_size):
        processed_paths, pred_labels, confidence_scores = predict_on_images(model, batch_paths, image_size)
        
        # Convert predicted labels to EPPO
        pred_labels_eppo = []
        for label in pred_labels:
            if label in labels_dict.values():
                pred_labels_eppo.append(list(labels_dict.keys())[list(labels_dict.values()).index(label)])
            else:
                pred_labels_eppo.append("Unknown")  # Handle unknown labels gracefully
        
        df_results = pd.DataFrame({
            'image_path': processed_paths,
            'pred_label_eppo': pred_labels_eppo,
            'confidence_score': confidence_scores
        })
        
        if not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0:
            df_results.to_csv(output_csv, index=False, mode='w')  # Write header on first batch
        else:
            df_results.to_csv(output_csv, index=False, mode='a', header=False)  # Append without header
        
        del processed_paths, pred_labels, confidence_scores, pred_labels_eppo, df_results
        gc.collect()
    
    print(f"Predictions saved to '{output_csv}'")
