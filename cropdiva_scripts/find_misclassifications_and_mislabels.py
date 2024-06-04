import os
import json
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.efficientnet import preprocess_input
import cnn_model

network_folder_path = 'networks/EfficientNetV2S_imagenet_224x224_avg_adam_lr0.0001_bs32_08DEC_Approved_MIMAD_TEST_100Epoch__10084c1a' # mimad
dataset_folder = 'Datasets'
subdataset = 'Test'

network_name = os.path.split(network_folder_path)[-1]
unique_identifier = network_name.split('__')[-1]

# Load configuration file 
with open(os.path.join(network_folder_path, 'configuration.json'),'r') as fob:
    train_config = json.load(fob)

print('\nTraining configuration: ', train_config)

# Load labels dict
labels_dict_file = os.path.join(dataset_folder,train_config['dataset_id'],'labels_dict_' + train_config['dataset_id'] + '.json')
with open(labels_dict_file,'r') as fob:
    labels_dict = json.load(fob)
N_classes = len(labels_dict)
print('Classes: ', labels_dict)

# Read dataset
dataframe_filepath = glob.glob(os.path.join(dataset_folder, train_config['dataset_id'],'*_'+ train_config['dataset_id'] + '_' + subdataset.title() + '.pkl'))[0]
df = pd.read_pickle(dataframe_filepath)

# Load dataframe
df_filename = os.path.split(dataframe_filepath)[-1]
df_filename_parts = df_filename.split(sep='.')
df_filename_predictions = '.'.join(df_filename_parts[:-1]) + '_w_pred_'  + unique_identifier + '.pkl'
df = pd.read_pickle(os.path.join(network_folder_path, df_filename_predictions))

# Filter misclassified
misclassified = df[df['label_no'] != df['pred_label_no']]

# Sort misclassified by confidence score in descending order
sorted_misclassified = misclassified.sort_values(by='confidence_score', ascending=False)

# Convert label numbers to label names
sorted_misclassified['true_label_name'] = sorted_misclassified['label_no'].apply(lambda x: {value: key for key, value in labels_dict.items()}[x])
sorted_misclassified['predicted_label_name'] = sorted_misclassified['pred_label_no'].apply(lambda x: {value: key for key, value in labels_dict.items()}[x])

# Select relevant columns to display
columns_to_display = ['true_label_name', 'predicted_label_name', 'confidence_score']
sorted_misclassified_display = sorted_misclassified[columns_to_display]

print(sorted_misclassified_display)

def create_and_save_misclassified_collages(df, trainingdata_folder, labels_dict, base_save_path):
    n_rows, n_cols = 5, 6  # Specify rows and columns for the collage
    images_per_collage = n_rows * n_cols  # Total images per collage
    misclassified = df[df['label_no'] != df['pred_label_no']].sort_values(by='confidence_score', ascending=False)
    total_misclassifications = len(misclassified)
    collage_count = 0  # Keep track of how many collages have been created

    for start_idx in range(0, total_misclassifications, images_per_collage):
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), dpi=100)
        fig.subplots_adjust(hspace=0.6, wspace=0.4)
        end_idx = min(start_idx + images_per_collage, total_misclassifications)
        subset = misclassified.iloc[start_idx:end_idx]

        ax_flat = axs.flatten()
        for ax_idx, (_, row) in enumerate(subset.iterrows()):
            ax = ax_flat[ax_idx]
            img_path = os.path.join(trainingdata_folder, row['folder'], row['image'])
            try:
                img = skimage.io.imread(img_path)
            except FileNotFoundError:
                print(f"File not found: {img_path}")
                continue
            ax.imshow(img)
            
            actual_label = [label for label, label_no in labels_dict.items() if label_no == row['label_no']][0]
            predicted_label = [label for label, label_no in labels_dict.items() if label_no == row['pred_label_no']][0]
            confidence = row['confidence_score']

            image_file_id = row['image'].split('_')
            species_and_id = '_'.join(image_file_id[:3])
            ax.set_title(f"Filename: {species_and_id}\nLabel: {actual_label} | Predicted: {predicted_label} | Conf: {confidence:.2f}", fontsize=10)
            ax.axis('off')

        # Turn off any unused subplots
        for ax in ax_flat[ax_idx + 1:]:
            ax.axis('off')

        collage_count += 1
        collage_save_path = os.path.join(base_save_path, f"mislabeled_sorted_by_confidence_collage_{collage_count}.png")
        plt.tight_layout()
        plt.savefig(collage_save_path, dpi=300)
        plt.close()
        print(f"Saved collage {collage_count} at {collage_save_path}")

eppo_to_danish = {
    "1CHEG": "Chenopodium",
    "1CRUF": "Korsblomstrede",
    "1LUPG": "Lupinus",
    "1URTF": "Urticaceae",
    "ATXPA": "Mælde, svine",
    "FAGES": "Fagopyrum esculentum",
    "POLAV": "Pileurt, vej",
    "POLLA": "Pileurt, bleg",
    "PPPMM": "1-Kimbladet",
    "VERPE": "Ærenpris, storkronet",
    "VICFX": "Hestebønne",
    "VIOAR": "Ager-Stedmoderblomst",
}

def find_and_save_likely_mislabeled_collages(df, trainingdata_folder, labels_dict, base_save_path, eppo_to_danish):
    # Calculate the difference between predicted confidence and actual label confidence
    df['confidence_difference'] = df['confidence_score'] - df['actual_label_confidence']
    
    # Filter for misclassifications with high confidence in prediction and low in actual label
    likely_mislabeled = df[(df['label_no'] != df['pred_label_no']) & (df['confidence_difference'] > 0)].copy()
    
    # Sort by 'confidence_difference'
    likely_mislabeled_sorted = likely_mislabeled.sort_values(by='confidence_difference', ascending=False)

    n_rows, n_cols = 5, 6
    images_per_collage = n_rows * n_cols
    total_misclassifications = len(likely_mislabeled_sorted)
    collage_count = 0

    for start_idx in range(0, total_misclassifications, images_per_collage):
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), dpi=100)
        fig.subplots_adjust(hspace=0.6, wspace=0.4)
        end_idx = min(start_idx + images_per_collage, total_misclassifications)
        subset = likely_mislabeled_sorted.iloc[start_idx:end_idx]

        ax_flat = axs.flatten()
        for ax_idx, (_, row) in enumerate(subset.iterrows()):
            ax = ax_flat[ax_idx]
            img_path = os.path.join(trainingdata_folder, row['folder'], row['image'])
            try:
                img = skimage.io.imread(img_path)
            except FileNotFoundError:
                print(f"File not found: {img_path}")
                continue
            ax.imshow(img)
            
            actual_label = [label for label, label_no in labels_dict.items() if label_no == row['label_no']][0]
            predicted_label = [label for label, label_no in labels_dict.items() if label_no == row['pred_label_no']][0]
            confidence_difference = row['confidence_difference']

            actual_label_in_danish_name = eppo_to_danish.get(actual_label)
            predicted_label_in_danish_name = eppo_to_danish.get(predicted_label)

            image_file_id = row['image'].split('_')
            species_and_id = '_'.join(image_file_id[:3])
            ax.set_title(f"Filename: {species_and_id}\nTrue: {actual_label_in_danish_name} | Pred: {predicted_label_in_danish_name}\nConf diff: {confidence_difference:.2f}", fontsize=10)
            ax.axis('off')

        
        for ax in ax_flat[ax_idx + 1:]:
            ax.axis('off')

        collage_count += 1
        collage_save_path = os.path.join(base_save_path, f"likely_mislabeled_collage_{collage_count}.png")
        plt.tight_layout()
        plt.savefig(collage_save_path, dpi=300)
        plt.close()
        print(f"Saved collage {collage_count} at {collage_save_path}")

network_folder_path = 'networks/EfficientNetV2S_imagenet_224x224_avg_adam_lr0.0001_bs32_08DEC_Approved_MIMAD_TEST_100Epoch__10084c1a' # mimad
dataset_folder = 'Datasets'
subdataset = 'Test'
trainingdata_folder = '../TrainingData/08DEC24_Approved_scale224px_min50px_padding' 
misclassified_sorted_by_confidence_save_path = f"{network_folder_path}/collages/misclassifications_by_confidence"
misclassified_likely_wrong_labels_save_path = f"{network_folder_path}/collages/likely_wrong_labels"
#create_and_save_misclassified_collages(df, trainingdata_folder, labels_dict, misclassified_sorted_by_confidence_save_path)
#find_and_save_likely_mislabeled_collages(df, trainingdata_folder, labels_dict, misclassified_likely_wrong_labels_save_path, eppo_to_danish)

def find_and_save_likely_mislabeled_collages_to_excel(df, trainingdata_folder, labels_dict, base_save_path, eppo_to_danish, excel_save_path):
    df['confidence_difference'] = df['confidence_score'] - df['actual_label_confidence']
    likely_mislabeled = df[(df['label_no'] != df['pred_label_no']) & (df['confidence_difference'] > 0)].copy()
    likely_mislabeled_sorted = likely_mislabeled.sort_values(by='confidence_difference', ascending=False)

    # List to store each row's data
    rows_list = []
    count = 0
    
    for _, row in likely_mislabeled_sorted.iterrows():
        img_path = os.path.join(trainingdata_folder, row['folder'], row['image'])
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            continue

        actual_label = [label for label, label_no in labels_dict.items() if label_no == row['label_no']][0]
        predicted_label = [label for label, label_no in labels_dict.items() if label_no == row['pred_label_no']][0]
        confidence_difference = row['confidence_difference']
        actual_label_in_danish = eppo_to_danish.get(actual_label, "Unknown")
        predicted_label_in_danish = eppo_to_danish.get(predicted_label, "Unknown")

        # Append each row's data to the list
        rows_list.append({
            'Filename': row['image'],
            'Actual Label': actual_label,
            'Actual Label (Danish)': actual_label_in_danish,
            'Predicted Label': predicted_label,
            'Predicted Label (Danish)': predicted_label_in_danish,
            'Confidence Difference': confidence_difference,
            'Appears on Collage': (count // 30)+1
        })

        count += 1

    # Convert dictionaries list to dataframe
    excel_data = pd.DataFrame(rows_list, columns=['Filename', 'Actual Label', 'Actual Label (Danish)', 'Predicted Label', 'Predicted Label (Danish)', 'Confidence Difference', 'Appears on Collage'])

    excel_data.to_excel(excel_save_path, index=False)
    print(f"Excel file saved at {excel_save_path}")

excel_save_path  = f"{network_folder_path}/collages/likely_wrong_labels/likely_mislabeled.xlsx"
find_and_save_likely_mislabeled_collages_to_excel(df, trainingdata_folder, labels_dict, misclassified_likely_wrong_labels_save_path, eppo_to_danish, excel_save_path)
