import argparse
import glob
import json
import numpy as np
import os
import pandas as pd
import tqdm

import utils

def main():
    # Setup input argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_df', action='store', default='dataframe_annotations__all.pkl', type=str, help='Filename of pickle file containing dataframe to be filtered (default: %(default)s).')
    parser.add_argument('--output_df', action='store', default='', type=str, help='Filename of filtered dataframe. Leave empty to simply append "__filtered" to the filename (default: %(default)s).')
    parser.add_argument('--min_sample_count', action='store', default=100, type=int, help='Minimum number of samples in a given class to be included. Classes with fewer samples are removed. Overruled by <discard_labels_file> and <labels_dict> (default: %(default)s).')
    parser.add_argument('--discard_labels_file', action='store', default='', type=str, help='Filename of json file with list of classes to discard. Ignoring <min_sample_count>, if it exists. Else removes classes with less than <min_sample_count> samples and saves discarded labels to file (default: %(default)s).')
    parser.add_argument('--labels_dict', action='store', default='', type=str, help='Filename of json file with dictionany of classes to keep. Ignoring <min_sample_count>. Leave empty to remove classes with less than <min_sample_count> samples (default: %(default)s).')

    # Parse input arguments and handle un
    args = vars(parser.parse_known_args()[0])
    print('\nArguments: ', args)

    input_dataframe_filename = args['input_df']
    print('Input: ', input_dataframe_filename)
    output_dataframe_filename = args['output_df']
    basename, ext = os.path.splitext(os.path.basename(input_dataframe_filename))
    if not output_dataframe_filename:
        output_dataframe_filename = os.path.join(os.path.dirname(input_dataframe_filename), basename + '__filtered' + ext)
    print('Output: ', output_dataframe_filename)
    labels_discard_file = args['discard_labels_file']
    if not labels_discard_file:
        labels_discard_file = os.path.join(os.path.dirname(input_dataframe_filename), basename + '__labels_discard.json')
    labels_dict_file = args['labels_dict']
    min_sample_count = args['min_sample_count']
    
    # Read dataframe
    df_all = pd.read_pickle(input_dataframe_filename)

    print('\n\n### Dataframe stats BEFORE filtering ###')
    utils.print_annotation_stats(df_all)

    ## Clean-up
    print('\n\n### Cleaning up dataset ###')

    print('\n# Filtering classes #')

    if labels_dict_file:
        # If label dict file is specified, keep only classes which match one of the classes from the label dict. Ignoring the number of samples in each class.
        print('\nRemoving classes not in labels dict file')
        with open(labels_dict_file,'r') as fob:
                labels_dict = json.load(fob)
        print(labels_dict)
        df_filt_list = []
        for label in labels_dict:
            df_filt_list.append(utils.dataframe_filtering(df_all, df_all['label'] == label)[0])
        df_filt = pd.concat(df_filt_list)
    else:
        # Remove images from classes with few samples
        if os.path.exists(labels_discard_file):
            print('\nRemoving classes from discard file')
            # Load discarded labels and use those instead of number of samples
            with open(labels_discard_file,'r') as fob:
                labels_discard = json.load(fob)
        else:
            print('\nRemoving classes with too few samples')
            df_label_count = df_all.groupby(['label'])['image'].count()
            
            labels = df_label_count.index.to_list()
            label_count = df_label_count.to_list()
            labels_discard = [l for l,c in zip(labels, label_count) if c < min_sample_count]
            # Save discarded labels for later use.
            with open(labels_discard_file,'w') as fob:
                json.dump(labels_discard, fob)
        # Do the actual filtering
        df_filt = df_all
        for label in tqdm.tqdm(labels_discard, desc='Removing classes'):
            df_filt, _ = utils.dataframe_filtering(df_filt, df_filt['label'] != label)

    print('\n### End of Cleaning up dataset ###\n\n')
    ## End of Clean-up

    print('\n\n### Dataframe stats AFTER filtering ###')
    utils.print_annotation_stats(df_filt)

    df_filt.to_pickle(output_dataframe_filename)
    print('Filtered dataframe saved to: ', output_dataframe_filename)

    print('done')

if __name__ == '__main__':
    main()