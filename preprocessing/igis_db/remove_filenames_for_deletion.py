import pandas as pd

file_names_df = pd.read_csv('deleted_filenames_1633.csv')
annotation_df = pd.read_csv('updated_data_1633_deleted.csv')
file_names_set = set(file_names_df['FileName'])
filtered_annotation_df = annotation_df[~annotation_df['FileName'].isin(file_names_set)]
filtered_annotation_df.to_csv('updated_data_1633_deleted_filename.csv', index=False)

print("Filtered annotation data saved")
