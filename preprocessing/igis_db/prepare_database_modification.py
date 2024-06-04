import pandas as pd

fetched_csv_path = 'fetched_1633.csv'
injected_csv_path = 'DELETED_annotations_data_1633.csv'
output_csv_path = 'updated_data_1633_deleted.csv'

fetched_df = pd.read_csv(fetched_csv_path)
injected_df = pd.read_csv(injected_csv_path)

updated_rows = []

for filename in fetched_df['FileName'].unique():
    fetched_rows = fetched_df[fetched_df['FileName'] == filename]
    injected_rows = injected_df[injected_df['FileName'] == filename]
    
    for _, injected_row in injected_rows.iterrows():
        new_row = fetched_rows.iloc[0].copy()  # Use the first row of fetched_rows as a template
        new_row['MinX'] = injected_row['MinX']
        new_row['MaxX'] = injected_row['MaxX']
        new_row['MinY'] = injected_row['MinY']
        new_row['MaxY'] = injected_row['MaxY']
        new_row['PlantId'] = injected_row['PlantId']
        new_row['Approved'] = injected_row['Approved']
        
        updated_rows.append(new_row)
    
    for _, fetched_row in fetched_rows.iterrows():
        if not injected_rows.empty:
            continue
        updated_rows.append(fetched_row)

updated_df = pd.DataFrame(updated_rows)
updated_df.to_csv(output_csv_path, index=False)
print(f"Updated data saved to {output_csv_path}")
