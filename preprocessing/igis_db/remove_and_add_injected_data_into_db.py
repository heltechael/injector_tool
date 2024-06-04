import pyodbc
import csv
from tqdm import tqdm

server = 'REDACTED'
database = 'REDACTED'
username = 'REDACTED'
password = 'REDACTED'

csv_file = 'updated_data_1633_deleted_filename_true.csv'
UPLOAD_ID = '1633'

# DELETE
delete_annotation_data_query = '''
DELETE ad
FROM RoboWeedMaps.[data].AnnotationData ad
JOIN RoboWeedMaps.[data].Annotations a ON ad.AnnotationId = a.ImageId
JOIN RoboWeedMaps.[data].Images i ON a.ImageId = i.Id
WHERE i.UploadId = 1633
'''

delete_annotations_query = '''
DELETE a
FROM RoboWeedMaps.[data].Annotations a
JOIN RoboWeedMaps.[data].Images i ON a.ImageId = i.Id
WHERE i.UploadId = 1633
'''

# INSERT
insert_annotations_query = '''
INSERT INTO RoboWeedMaps.[data].Annotations (ImageId, UseForTraining)
VALUES (?, ?)
'''

insert_annotation_data_query = '''
INSERT INTO RoboWeedMaps.[data].AnnotationData (AnnotationId, PlantId, MinX, MaxX, MinY, MaxY, Approved)
VALUES (?, ?, ?, ?, ?, ?, ?)
'''

def delete_old_data(delete_annotation_data_query, delete_annotations_query):
    try:
        conn_str = f'DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password};Encrypt=no'
        conn = pyodbc.connect(conn_str)
        
        cursor = conn.cursor()
    
        cursor.execute(delete_annotation_data_query)
        cursor.execute(delete_annotations_query)
        
        conn.commit()
        print("Old annotation data deleted successfully.")
            
    except pyodbc.Error as e:
        print(f"Error connecting to the database: {e}")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()

def insert_annotation_data(csv_file, insert_annotations_query, insert_annotation_data_query):
    try:
        conn_str = f'DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password};Encrypt=no'
        conn = pyodbc.connect(conn_str)
        
        cursor = conn.cursor()
        
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip the header row
            
            total_rows = sum(1 for row in reader)
            file.seek(0)  # Reset the reader to the beginning of the file
            next(reader)  # Skip the header row again
            pbar = tqdm(total=total_rows, unit='rows')
            
            inserted_annotations = set()
            
            for row in reader:
                image_id = row[0]
                upload_id = row[1]
                use_for_training = row[4]
                annotation_id = row[3]
                plant_id = row[5]
                min_x = row[6]
                max_x = row[7]
                min_y = row[8]
                max_y = row[9]
                approved = row[10]
                
                # IMPORTANT - ONLY UPLOADID SPECIFIED!!!
                if upload_id == UPLOAD_ID:
                    if (image_id, use_for_training) not in inserted_annotations:
                        cursor.execute(insert_annotations_query, (image_id, use_for_training))
                        inserted_annotations.add((image_id, use_for_training))
                    
                    cursor.execute(insert_annotation_data_query, (image_id, plant_id, min_x, max_x, min_y, max_y, approved))
                pbar.update(1)
            pbar.close()
        
        conn.commit()
        
        print("Annotation data inserted successfully.")
            
    except pyodbc.Error as e:
        print(f"Error connecting to the database: {e}")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()

delete_old_data(delete_annotation_data_query, delete_annotations_query)
insert_annotation_data(csv_file, insert_annotations_query, insert_annotation_data_query)
