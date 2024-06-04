import pyodbc
import csv
from tqdm import tqdm

# Database connection details
server = '10.0.50.28'
database = 'RoboWeedMaps'
username = 'RoboWeedUser'
password = 'pAW87JgxFbiRS5JT'

new_upload_id = '''
SELECT i.Id, i.UploadId, i.FileName, a.UseForTraining, ad.PlantId, ad.MinX, ad.MaxX, ad.MinY, ad.MaxY, ad.Approved
FROM RoboWeedMaps.[data].Images i
JOIN RoboWeedMaps.[data].Annotations a ON i.Id = a.ImageId
JOIN RoboWeedMaps.[data].AnnotationData ad ON i.Id = ad.AnnotationId
WHERE i.UploadId = 1629
'''

new_upload_id_count = '''
SELECT 
    COUNT(*) AS total_rows,
    COUNT(DISTINCT i.FileName) AS distinct_filenames
FROM RoboWeedMaps.[data].Images i
JOIN RoboWeedMaps.[data].Annotations a ON i.Id = a.ImageId
JOIN RoboWeedMaps.[data].AnnotationData ad ON i.Id = ad.AnnotationId
WHERE i.UploadId = 1629
'''

# Output file path
output_file = 'new_upload_id.csv'

def fetch_and_store_from_db(fetch_query, count_query, output_file):
    try:
        # Connect to the database
        conn_str = f'DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password};Encrypt=no'
        conn = pyodbc.connect(conn_str)
        
        # Create a cursor object
        cursor = conn.cursor()
        
        # Execute the count query
        cursor.execute(count_query)
        
        # Fetch the count results
        count_results = cursor.fetchone()
        total_rows = count_results[0]
        distinct_filenames = count_results[1]
        
        print(f"Total rows: {total_rows}")
        print(f"Distinct filenames: {distinct_filenames}")
        
        # Execute the main query if needed
        if total_rows > 0:
            cursor.execute(fetch_query)
            
            # Get the column names
            column_names = [column[0] for column in cursor.description]
            
            # Open the CSV file for writing
            with open(output_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(column_names)
                
                # Fetch rows in batches and update the progress bar
                batch_size = 50
                pbar = tqdm(total=total_rows, unit='rows')
                while True:
                    rows = cursor.fetchmany(batch_size)
                    if not rows:
                        break
                    writer.writerows(rows)
                    pbar.update(len(rows))
                pbar.close()
            
            print(f"Query result saved to {output_file}")
        else:
            print("No rows found matching the query conditions.")
            
    except pyodbc.Error as e:
        print(f"Error connecting to the database: {e}")
    finally:
        # Close the cursor and the connection
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()

def get_count_query(count_query):
    try:
        # Connect to the database
        conn_str = f'DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password};Encrypt=no'
        conn = pyodbc.connect(conn_str)
        
        # Create a cursor object
        cursor = conn.cursor()
        
        # Execute the count query
        cursor.execute(count_query)
        
        # Fetch the count results
        count_results = cursor.fetchone()
        total_rows = count_results[0]
        distinct_filenames = count_results[1]
        
        print(f"Total rows: {total_rows}")
        print(f"Distinct filenames: {distinct_filenames}")
            
    except pyodbc.Error as e:
        print(f"Error connecting to the database: {e}")
    finally:
        # Close the cursor and the connection
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()
# Execute query

#get_count_query(all_training_thumbnails_count)
fetch_and_store_from_db(new_upload_id, new_upload_id_count, output_file)