import pandas as pd

input_file = 'updated_data_1632_deleted_filename.csv'
df = pd.read_csv(input_file)

df['UseForTraining'] = True
df['Approved'] = True

output_file = 'updated_data_1632_deleted_filename_true.csv'
df.to_csv(output_file, index=False)

print(f"Modified CSV file saved as {output_file}")
