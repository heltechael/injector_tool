import argparse
import glob
import GPUtil
import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cnn_model
import confusionmatrix as CM

def main():
    
    # Setup input argument parser
    parser = argparse.ArgumentParser()

    parser_group_data = parser.add_argument_group('GPU operations')
    parser_group_data.add_argument('--gpu_cluster', action='store', default=0, type=int, help='Use GPU cluster for training (default: %(default)s).')

    parser.add_argument('--network', action='store', default='', type=str, help='Path to folder with trained network (default: %(default)s).')
    parser.add_argument('--checkpoint', action='store', default='best', choices=['best', 'last'], type=str, help="Select which chcekpoint to load. 'best' (best_model_checkpoint) or 'last' (model_after_training) (default: %(default)s)")
    parser.add_argument('--dataset_folder', action='store', default='datasets', type=str, help='Path to main folder containing all datasets (patch must not including dataset ID) (default: %(default)s).')
    parser.add_argument('--dataset_id', action='store', default='', type=str, help='ID of dataset used for evaluation. If not specified, use the same as used for training.')
    parser.add_argument('--subdataset', action='store', default='validation', choices=['train', 'validation', 'test'],type=str, help='Which subset of the dataset to evaluate (default: %(default)s).')
    parser.add_argument('--image_folder', action='store', default='images', type=str, help='Path to main folder containing the images in subfolders (default: %(default)s).')

    args = vars(parser.parse_args())
    print('\nArguments: ', args)
    gpu_cluster = args['gpu_cluster']
    network_folder_path = args['network']
    network_name = os.path.split(network_folder_path)[-1]
    checkpoint = args['checkpoint']
    if (checkpoint == 'best'):
        checkpoint_to_load = 'best_model_checkpoint'
    else:
        checkpoint_to_load = 'model_after_training'
    dataset_folder = args['dataset_folder']
    dataset_id = args['dataset_id']
    subdataset = args['subdataset']
    image_folder = args['image_folder']
    unique_identifier = network_name.split('__')[-1]

    # Load configuration file from trained network
    with open(os.path.join(network_folder_path, 'configuration.json'),'r') as fob:
        train_config = json.load(fob)

    print('\nTraining configuration: ', train_config)

    # Allocate a GPU
    if gpu_cluster:
        # Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # Get the first available GPU
        DEVICE_ID_LIST = GPUtil.getFirstAvailable(maxLoad=0.01, maxMemory=0.01, attempts=999999, interval=60, verbose=True)
        DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list
        # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
    else:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]="0"

    if not dataset_id:
        dataset_id = train_config['dataset_id']
    # Read dataset
    dataframe_filepath = glob.glob(os.path.join(dataset_folder, dataset_id,'*_'+ dataset_id + '_' + subdataset.title() + '.pkl'))[0]
    df = pd.read_pickle(dataframe_filepath)

    # Load labels dict
    labels_dict_file = os.path.join(dataset_folder, dataset_id,'labels_dict_' + dataset_id + '.json')
    with open(labels_dict_file,'r') as fob:
        labels_dict = json.load(fob)
    N_classes = len(labels_dict)
    print('Classes: ', labels_dict)
    print('Number of classes:', N_classes)

    # Load model
    print('Loading model...')
    model = cnn_model.load_model(os.path.join(network_folder_path, checkpoint_to_load))

    print('Predicting on dataset (' + subdataset + ')...')
    df_w_predictions = cnn_model.prediction_on_df(model, df, image_folder=image_folder)
    # Save/dump dataframe
    df_filename = os.path.split(dataframe_filepath)[-1]
    df_filename_parts = df_filename.split(sep='.')
    df_filename_out = '.'.join(df_filename_parts[:-1]) + '_w_pred_'  + unique_identifier + '.pkl'
    print('Dumping predictions to file: ', df_filename_out)
    df_w_predictions.to_pickle(os.path.join(network_folder_path, df_filename_out))

    # Create, display and save confusion matrix of dataset
    labels = [key for key in labels_dict.keys()]
    print('\nConfusion matrix from dataset (' + subdataset + '):')
    CM_subdataset = CM.confusionmatrix(N_classes, labels)
    CM_subdataset.Append(df['label_no'], df['pred_label_no'])
    CM_subdataset.Save(os.path.join(network_folder_path, 'ConfMat_' + checkpoint + '_' + subdataset + '.csv'), fileFormat='csv')
    print(CM_subdataset)
    print(labels_dict)

if __name__ == '__main__':
    main()
