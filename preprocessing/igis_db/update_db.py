import pyodbc
import csv
from tqdm import tqdm

server = 'REDACTED'
database = 'REDACTED'
username = 'REDACTED'
password = 'REDACTED'
input_file = 'updated_data.csv'

# SET UPLOADID 
specified_upload_id = 1629

def update_database_from_csv(input_file, specified_upload_id):
    try:
        conn_str = f'DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password};Encrypt=no'
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        with open(input_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            total_rows = sum(1 for row in reader)
            csvfile.seek(0)
            reader = csv.DictReader(csvfile)

            pbar = tqdm(total=total_rows, unit='rows')
            for row in reader:
                annotation_id = row['ID']  # This is the unique annotation ID
                plant_id = row['PlantId']
                min_x = row['MinX']
                max_x = row['MaxX']
                min_y = row['MinY']
                max_y = row['MaxY']
                approved = row['Approved']
                
                update_query = '''
                UPDATE RoboWeedMaps.[data].AnnotationData
                SET PlantId = ?, MinX = ?, MaxX = ?, MinY = ?, MaxY = ?, Approved = ?
                WHERE ID = ? AND ID IN (
                    SELECT ad.ID
                    FROM RoboWeedMaps.[data].Images i
                    JOIN RoboWeedMaps.[data].Annotations a ON i.Id = a.ImageId
                    JOIN RoboWeedMaps.[data].AnnotationData ad ON i.Id = ad.AnnotationId
                    WHERE i.UploadId = ?
                )
                '''

                cursor.execute(update_query, (plant_id, min_x, max_x, min_y, max_y, approved, annotation_id, specified_upload_id))
                pbar.update(1)
            
            pbar.close()
            conn.commit()
            print(f"Database updated successfully from {input_file}")
    
    except pyodbc.Error as e:
        print(f"Error connecting to the database: {e}")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()

update_database_from_csv(input_file, specified_upload_id)
