import pyodbc
import csv
from tqdm import tqdm

# Database connection details
server = '10.0.50.28'
database = 'RoboWeedMaps'
username = 'RoboWeedUser'
password = 'pAW87JgxFbiRS5JT'

# Query to fetch all images without annotation data
fetch_images_without_annotation_data = '''
SELECT i.UploadId, i.FileName
FROM RoboWeedMaps.[data].Images i
JOIN RoboWeedMaps.[data].Annotations a ON i.Id = a.ImageId
JOIN RoboWeedMaps.[data].AnnotationData ad ON i.Id = ad.AnnotationId
WHERE ad.AnnotationId IS NULL
'''

# Count query for the images without annotation data
count_images_without_annotation_data = '''
SELECT 
    COUNT(*) AS total_rows,
    COUNT(DISTINCT i.FileName) AS distinct_filenames
FROM RoboWeedMaps.[data].Images i
JOIN RoboWeedMaps.[data].Annotations a ON i.Id = a.ImageId
JOIN RoboWeedMaps.[data].AnnotationData ad ON i.Id = ad.AnnotationId
WHERE ad.AnnotationId IS NULL
'''

# Output file path
output_file_without_annotation_data = 'images_without_annotation_data.csv'

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

# Execute the function with the new query
fetch_and_store_from_db(fetch_images_without_annotation_data, count_images_without_annotation_data, output_file_without_annotation_data)
