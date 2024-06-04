import pyodbc

server = 'REDACTED'
database = 'REDACTED'
username = 'REDACTED'
password = 'REDACTED'

delete_query = '''
DELETE ad
FROM RoboWeedMaps.[data].AnnotationData ad
JOIN RoboWeedMaps.[data].Annotations a ON ad.AnnotationId = a.ImageId
JOIN RoboWeedMaps.[data].Images i ON a.ImageId = i.Id
WHERE i.UploadId = 1631
'''

# Purge uploadid from rwm
def delete_annotation_data(delete_query):
    try:
        conn_str = f'DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password};Encrypt=no'
        conn = pyodbc.connect(conn_str)
        
        cursor = conn.cursor()
        
        cursor.execute(delete_query)
        
        conn.commit()
        
        print("Annotation data deleted successfully.")
            
    except pyodbc.Error as e:
        print(f"Error connecting to the database: {e}")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()

delete_annotation_data(delete_query)
