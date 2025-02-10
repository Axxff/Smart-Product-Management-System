import sqlite3
import os
import csv

# Ensure database is saved in the correct directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "products.db")

def init_db():
    """Initialize the database and create table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            barcode TEXT PRIMARY KEY,
            serial_number TEXT,
            product_image BLOB
        )
    """)

    conn.commit()
    conn.close()
    print("Database initialized successfully!")

def store_csv(file_path):
    """Read a .csv file and store its data into the database."""
    if not os.path.exists(file_path):
        print("Error: File not found!")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip the header if present

            for row in reader:
                if len(row) < 2:
                    print(f"Skipping invalid row: {row}")
                    continue
                
                barcode, serial_number = row[:2]  # Only first 2 columns
                
                try:
                    cursor.execute("INSERT INTO products (barcode, serial_number) VALUES (?, ?)", 
                                   (barcode, serial_number))
                except sqlite3.IntegrityError:
                    print(f"Skipping duplicate barcode: {barcode}")

        conn.commit()
        print(f"CSV data from {file_path} stored successfully!")
    except Exception as e:
        print(f"Error processing CSV: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
    csv_file = "path/to/your/file.csv"  # Update with your CSV file path
    store_csv(csv_file)
