import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.special import softmax
from scipy.stats import norm
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import utils
import matplotlib.pyplot as plt
import skimage.io

network_folder_path = 'networks/EfficientNetV2S_imagenet_224x224_avg_adam_lr0.0001_bs32_08DEC_Approved_MIMAD_TEST_100Epoch__10084c1a' # mimad
dataset_folder = 'Datasets'
subdataset = 'Test'

network_name = os.path.split(network_folder_path)[-1]
unique_identifier = network_name.split('__')[-1]

# Load configuration file from trained network
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

print(f"Filename for df: {df_filename_predictions}")
print(f"Categories (columns) in the dataframe: {df.columns.tolist()}")

if 'confidence_score' in df.columns:
    print("'confidence_score' successfully added to the dataframe")
else:
    print("'confidence_score' not found in the dataframe")

# Visualize misclassifications with lowest confidence scores
dataset_folder = "../TrainingData/08DEC24_Approved_scale224px_min50px_padding/"

misclassified = df[df['label_no'] != df['pred_label_no']]
misclassified_sorted_by_least_confidence = misclassified.sort_values(by='confidence_score', ascending=True)
print(misclassified_sorted_by_least_confidence)

df['normalized_confidence_score'] = (df['confidence_score'] - df['confidence_score'].min()) / \
                                    (df['confidence_score'].max() - df['confidence_score'].min())

quartiles = df['normalized_confidence_score'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
print(f"The normalized confidence score quartiles are computed as:\n{quartiles}")

def get_images_closest_to_quartiles(df, quartiles, labels_dict):
    images_by_quartile = {qt: {} for qt in quartiles.index}
    
    for label, label_no in labels_dict.items():
        class_df = df[df['label_no']==label_no]

        if class_df.empty:
            print(f"No data for class {label_no}.")
            continue
        
        for qt in quartiles.index:
            closest_row_to_quartile_value = class_df.iloc[(class_df['normalized_confidence_score'] - quartiles[qt]).abs().argsort()[:1]]
            
            if closest_row_to_quartile_value.empty:
                print(f"No closest row found for class {label} at quartile {qt}.")
                continue
            
            image_path = closest_row_to_quartile_value.iloc[0]['folder'] + '/' + closest_row_to_quartile_value.iloc[0]['image']
            images_by_quartile[qt][label] = image_path
    
    return images_by_quartile

images_by_quartile = get_images_closest_to_quartiles(df, quartiles, labels_dict)

print(images_by_quartile)

def create_collage_and_save_with_confidence(images_by_quartile, dataset_folder, labels_dict, save_path):
    n_rows = len(labels_dict) + 1 
    n_cols = len(images_by_quartile)
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    fig.subplots_adjust(hspace=0.6, wspace=0.4)

    # Set quartile labels at the top of each column
    for qt_idx, qt in enumerate(images_by_quartile.keys()):
        axs[0, qt_idx].text(0.5, 0.5, f'Quartile {qt}', ha='center', va='bottom', transform=axs[0, qt_idx].transAxes, fontsize=26)
        axs[0, qt_idx].axis('off')
    
    for row_idx, (label, label_no) in enumerate(labels_dict.items(), start=1):
        fig.text(0.01, (n_rows-row_idx)/n_rows, label, va='center', ha='left', fontsize=12)
        
        for col_idx, (qt, images) in enumerate(images_by_quartile.items()):
            ax = axs[row_idx, col_idx]

            if label in images:
                image_path = images[label]
                img = skimage.io.imread(os.path.join(dataset_folder, image_path))
                ax.imshow(img)
                
                # Find the row in df corresponding to this image
                folder_name, image_name = image_path.split('/')[-2], image_path.split('/')[-1]
                df_row = df[(df['folder'] == folder_name) & (df['image'] == image_name)].iloc[0]
                
                # Correctly identify actual and predicted labels and include confidence score
                actual_label = label  
                pred_label_no = df_row['pred_label_no']
                pred_label = [key for key, value in labels_dict.items() if value == pred_label_no][0]
                confidence_score = df_row['confidence_score']
                
                ax.set_title(f"Actual: {actual_label}\nPredicted: {pred_label}\nConfidence: {confidence_score:.2f}", fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No image', ha='center', va='center', fontsize=8)
            
            ax.axis('off') 
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  
    plt.close()

save_path = f"{network_folder_path}/collage_corrected.png" 
create_collage_and_save_with_confidence(images_by_quartile, dataset_folder, labels_dict, save_path)

def create_misclassified_collage_and_save(images_by_quartile, dataset_folder, labels_dict, save_path):
    n_rows = len(labels_dict)
    n_cols = 10

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), dpi=100)
    fig.subplots_adjust(hspace=0.6, wspace=0.4)

    for row_idx, (label, label_no) in enumerate(labels_dict.items()):
        fig.text(0.01, (n_rows-row_idx)/n_rows, label, va='center', ha='left', fontsize=12)
        misclassified = df[(df['label_no']==label_no) & (df['pred_label_no'] != label_no)]
        misclassified_sorted = misclassified.sort_values(by='confidence_score', ascending=True).head(n_cols)
        print(misclassified_sorted)

        for col_idx, (_, row) in enumerate(misclassified_sorted.iterrows()):
            ax = axs[row_idx, col_idx]
            img_path = os.path.join(dataset_folder, row['folder'], row['image'])
            img = skimage.io.imread(img_path)
            ax.imshow(img)
            
            actual_label = label  
            pred_label_no = row['pred_label_no']
            pred_label = [key for key, value in labels_dict.items() if value == pred_label_no][0]
            confidence = row['confidence_score']
            
            ax.set_title(f"Actual: {actual_label}\nPredicted: {pred_label}\nConf: {confidence:.2f}", fontsize=10)
            ax.axis('off')

    for ax in axs.flat[misclassified_sorted.shape[0]:]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  
    plt.close()  

save_path = f"{network_folder_path}/misclassifications_collage.png" 
create_misclassified_collage_and_save(df, dataset_folder, labels_dict, save_path)

