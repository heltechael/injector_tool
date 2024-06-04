import pandas as pd

def make_filenames_unique(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    
    current_filename = None
    filename_counter = {}
    
    for index, row in df.iterrows():
        filename = row['FileName']
        if filename != current_filename:
            if filename in filename_counter:
                filename_counter[filename] += 1
            else:
                filename_counter[filename] = 0
            current_filename = filename
        
        if filename_counter[filename] > 0:
            new_filename = f"{filename.split('.')[0]}_FIXED{filename_counter[filename]}.jpg"
            df.at[index, 'FileName'] = new_filename
    
    df.to_csv(output_csv, index=False)

input_csv = 'filtered_annotations_upload1632.csv' 
output_csv = 'FIXED_annotations_data_1632.csv'

make_filenames_unique(input_csv, output_csv)
