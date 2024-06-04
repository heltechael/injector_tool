import pyodbc
import pandas as pd
from tqdm import tqdm

server = 'REDACTED'
database = 'REDACTED'
username = 'REDACTED'
password = 'REDACTED'

csv_file = 'updated_annotations_unapproved.csv'

def update_annotations_in_db(csv_file):
    try:
        conn_str = f'DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password};Encrypt=no'
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        df = pd.read_csv(csv_file)
        
        for index, row in tqdm(df.iterrows(), total=df.shape[0], unit='rows'):
            upload_id = row['UploadId']
            image_id = row['ImageId']
            annotation_id = row['ID']
            plant_id = row['PlantId']
            classification_score = row['ClassificationScore']
            approved = row['Approved']
            
            # Update
            update_query = '''
            UPDATE ad
            SET ad.PlantId = ?, ad.ClassificationScore = ?, ad.Approved = ?
            FROM RoboWeedMaps.[data].Images i
            JOIN RoboWeedMaps.[data].Annotations a ON i.Id = a.ImageId
            JOIN RoboWeedMaps.[data].AnnotationData ad ON i.Id = ad.AnnotationId
            WHERE i.UploadId = ? AND i.Id = ? AND ad.ID = ? AND i.UploadId IN (1372,1373,1378,1379,1380,1381,1384,1385,1386,1387,1394,1395)
            '''
            
            cursor.execute(update_query, (plant_id, classification_score, approved, upload_id, image_id, annotation_id))
        
        conn.commit()
        
        print("Database updated successfully.")
        
    except pyodbc.Error as e:
        print(f"Error connecting to the database: {e}")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()

update_annotations_in_db(csv_file)
