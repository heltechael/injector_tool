import pandas as pd

df = pd.read_csv('updated_data_1632_without_duplicates.csv')
df['UseForTraining'] = True
df.to_csv('your_file.csv', index=False)

