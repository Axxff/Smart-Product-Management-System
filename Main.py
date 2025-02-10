import cv2
import pandas as pd
import numpy as np
from pyzbar.pyzbar import decode
from PIL import Image, ImageTk
import sqlite3
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import io
import uuid
import re
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional, Tuple, Any
from dataclasses import dataclass

@dataclass
class Product:
    barcode: str
    serial_number: str
    image_data: bytes
    created_at: datetime = None

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        try:
            with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
                conn.execute("""
                   CREATE TABLE IF NOT EXISTS products (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       barcode TEXT UNIQUE,
                       serial_number TEXT,
                       product_image BLOB,
                       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Database initialization failed: {e}")
            raise
    
    def execute_query(self, query: str, params: tuple = ()) -> list:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                return cursor.fetchall()
        except sqlite3.Error as e:
            logging.error(f"Database error: {e}")
            raise

class BarcodeScanner:
    @staticmethod
    def read_barcode(image_path: str) -> Tuple[Optional[str], Optional[np.ndarray]]:
        try:
            # Read original image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Failed to load image")
            
            # Preprocessing techniques
            def preprocess_image(image):
                preprocessed_images = []
                
                # Convert to multiple color spaces
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                preprocessed_images.append(gray)
                
                # Enhance contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced_gray = clahe.apply(gray)
                preprocessed_images.append(enhanced_gray)
                
                # Noise reduction
                denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
                preprocessed_images.append(denoised)
                
                # Edge enhancement
                edges = cv2.Canny(gray, 100, 200)
                preprocessed_images.append(edges)
                
                return preprocessed_images
            
            # Try preprocessing techniques
            preprocessed_images = preprocess_image(img)
            
            for processed_img in preprocessed_images:
                try:
                    # Ensure grayscale
                    if len(processed_img.shape) > 2:
                        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
                    
                    # Safer decoding with error handling
                    decoded = decode(processed_img)
                    
                    if decoded:
                        barcode = decoded[0]
                        barcode_data = barcode.data.decode("utf-8", errors='ignore')
                        
                        if barcode_data:
                            # Draw bounding box
                            points = barcode.polygon
                            if len(points) == 4:
                                pts = np.array(points, np.int32)
                                pts = pts.reshape((-1, 1, 2))
                                cv2.polylines(img, [pts], True, (0, 255, 0), 3)
                            
                            return barcode_data, img
                except Exception as decode_error:
                    logging.error(f"Decoding error: {decode_error}")
                    continue
            
            return None, None
        except Exception as e:
            logging.error(f"Barcode reading error: {e}")
            return None, None
    
    @staticmethod
    def validate_barcode(barcode: str) -> bool:
        # Basic validation - can be extended based on specific barcode format requirements
        return bool(re.match(r'^[A-Za-z0-9-_.]+$', barcode))    

class ProductManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.scanner = BarcodeScanner()
    
    def store_product(self, image_path: str, barcode: Optional[str] = None) -> bool:
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            if not barcode:
                detected_barcode, _ = self.scanner.read_barcode(image_path)
                barcode = detected_barcode
            
            if not barcode or not self.scanner.validate_barcode(barcode):
                raise ValueError("No barcode detected in image")
            
            with open(image_path, "rb") as file:
                image_data = file.read()
                if not image_data:
                    raise ValueError("Image file is empty")
                
                Image.open(io.BytesIO(image_data))
            
            try:
                self.db_manager.execute_query(
                    """INSERT INTO products 
                       (barcode, serial_number, product_image, created_at) 
                       VALUES (?, ?, ?, ?)""",
                    (barcode, str(uuid.uuid4())[:8], image_data, datetime.now())
                )
                return True
            except sqlite3.IntegrityError:
                raise ValueError("Barcode already exists in database")
                
        except Exception as e:
            logging.error(f"Error storing product: {str(e)}")
            return False
    
    def retrieve_product(self, barcode: str) -> Optional[Product]:
        try:
            result = self.db_manager.execute_query(
                """SELECT barcode, serial_number, product_image, created_at 
                   FROM products WHERE barcode = ?""",
                (barcode,)
            )
            return Product(*result[0]) if result else None
        except Exception as e:
            logging.error(f"Error retrieving product: {e}")
            return None
    
    def get_all_products(self) -> list[Product]:
        try:
            results = self.db_manager.execute_query(
                """SELECT barcode, serial_number, product_image, created_at 
                   FROM products ORDER BY created_at DESC"""
            )
            return [Product(*row) for row in results]
        except Exception as e:
            logging.error(f"Error retrieving products: {e}")
            return []
    
    def delete_product(self, barcode: str) -> bool:
        try:
            self.db_manager.execute_query(
                "DELETE FROM products WHERE barcode = ?",
                (barcode,)
            )
            return True
        except Exception as e:
            logging.error(f"Error deleting product: {e}")
            return False

class ProductUI:
    def __init__(self, root: tk.Tk, product_manager: ProductManager):
        self.root = root
        self.product_manager = product_manager
        self.image_refs = {}  # Store image references to prevent garbage collection
        self.setup_ui()
    
    def cleanup_image_refs(self, window_id: str) -> None:
        if window_id in self.image_refs:
            del self.image_refs[window_id]

    def setup_ui(self) -> None:
        self.root.title("Smart Product Management")
        self.root.geometry("600x500")
        
        style = ttk.Style()
        style.configure("TLabel", font=("Helvetica", 12))
        style.configure("TButton", font=("Helvetica", 10), padding=10)
        
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = ttk.Label(
            main_frame, 
            text="Product Management System", 
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=(0, 20))
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        buttons = [
            ("📦 Store New Product", self.browse_and_store),
            ("📦 Store Multiple Products", self.store_multiple_products),
            ("🔍 Scan & Retrieve Product", self.browse_and_retrieve),
            ("📋 View Product Inventory", self.view_products),
            ("📂 Import Dataset", self.import_dataset)
        ]
        
        for text, command in buttons:
            btn = ttk.Button(
                button_frame,
                text=text,
                command=command,
                style="TButton"
            )
            btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

    def store_multiple_products(self) -> None:
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Image Files", "*.bmp;*.png;*.jpg;*.jpeg")]
        )
        if not file_paths:
            return
            
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Processing Images")
        progress_window.geometry("400x200")
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(
            progress_window,
            variable=progress_var,
            maximum=len(file_paths)
        )
        progress_bar.pack(pady=20, padx=10, fill=tk.X)
        
        status_label = ttk.Label(progress_window, text="Processing images...")
        status_label.pack(pady=10)
        
        results_text = tk.Text(progress_window, height=5, width=40)
        results_text.pack(pady=10, padx=10)
        
        success_count = 0
        failed_images = []
        
        for index, file_path in enumerate(file_paths):
            status_label.config(text=f"Processing image {index + 1} of {len(file_paths)}")
            barcode, _ = self.product_manager.scanner.read_barcode(file_path)
            
            if barcode and self.product_manager.store_product(file_path, barcode):
                success_count += 1
                results_text.insert(tk.END, f"✓ Success: {os.path.basename(file_path)}\n")
            else:
                failed_images.append(os.path.basename(file_path))
                results_text.insert(tk.END, f"✗ Failed: {os.path.basename(file_path)}\n")
            
            progress_var.set(index + 1)
            progress_window.update()
            results_text.see(tk.END)
        
        status_label.config(text="Processing complete!")
        
        summary = (
            f"Successfully processed {success_count} of {len(file_paths)} images.\n"
            f"Failed: {len(failed_images)} images."
        )
        messagebox.showinfo("Processing Complete", summary)

    def browse_and_store(self) -> None:
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Image Files", "*.bmp;*.png;*.jpg;*.jpeg")]
        )
        if not file_paths:
            return
        
        preview_window = tk.Toplevel(self.root)
        preview_window.title("Barcode Detection")
        preview_window.geometry("800x700")
        
        canvas_frame = ttk.Frame(preview_window)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        def process_image(file_path):
            barcode, detected_image = self.product_manager.scanner.read_barcode(file_path)
            
            if detected_image is not None:
                detected_image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(detected_image_rgb)
            else:
                pil_image = Image.open(file_path)
                detected_image = cv2.imread(file_path)
            
            pil_image.thumbnail((500, 500), Image.LANCZOS)
            photo = ImageTk.PhotoImage(pil_image)
            
            image_frame = ttk.Frame(scrollable_frame)
            image_frame.pack(pady=10, fill=tk.X)
            
            image_label = ttk.Label(image_frame, image=photo)
            image_label.image = photo
            image_label.pack(pady=5)
            
            if barcode:
                barcode_label = ttk.Label(
                    image_frame, 
                    text=f"Detected Barcode: {barcode}", 
                    font=("Helvetica", 12)
                )
                barcode_label.pack(pady=5)
                
                def save_product():
                    try:
                        result = self.product_manager.store_product(file_path, barcode)
                        if result:
                            messagebox.showinfo("Success", f"Product {barcode} stored successfully!")
                            preview_window.destroy()
                        else:
                            error_message = "Failed to store product. Please check the image and try again."
                            messagebox.showerror("Save Error", error_message)
                            preview_window.destroy()
                    except Exception as e:
                        error_message = f"Error saving product: {str(e)}"
                        messagebox.showerror("Save Error", error_message)
                        preview_window.destroy()
                    
                save_btn = ttk.Button(
                    image_frame, 
                    text="Save Product", 
                    command=save_product
                )
                save_btn.pack(pady=5)
            else:
                ttk.Label(
                    image_frame, 
                    text="No barcode detected", 
                    foreground="red", 
                    font=("Helvetica", 12)
                ).pack(pady=5)
        
        for file_path in file_paths:
            process_image(file_path)
        
        close_btn = ttk.Button(
            preview_window, 
            text="Close", 
            command=preview_window.destroy
        )
        close_btn.pack(pady=10)
    
    def browse_and_retrieve(self) -> None:
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.bmp;*.png;*.jpg;*.jpeg")]
        )
        if not file_path:
            return
            
        barcode = self.product_manager.scanner.read_barcode(file_path)[0]
        if not barcode:
            messagebox.showerror("Error", "No barcode found in image")
            return
            
        product = self.product_manager.retrieve_product(barcode)
        if product:
            messagebox.showinfo(
                "Product Found",
                f"Barcode: {product.barcode}\n"
                f"Serial Number: {product.serial_number}\n"
                f"Created: {product.created_at}"
            )
        else:
            messagebox.showerror("Error", "Product not found in database")
    
    def view_products(self) -> None:
        view_window = tk.Toplevel(self.root)
        view_window.title("Product Inventory")
        view_window.geometry("1000x600")

        
        
        tree_frame = ttk.Frame(view_window)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tree_scroll = ttk.Scrollbar(tree_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        tree = ttk.Treeview(
            tree_frame,
            columns=("Select", "Barcode", "Serial Number", "Created"),
            show="headings",
            yscrollcommand=tree_scroll.set
        )
        tree_scroll.config(command=tree.yview)
        
        tree.heading("Select", text="Select")
        tree.heading("Barcode", text="Barcode")
        tree.heading("Serial Number", text="Serial Number")
        tree.heading("Created", text="Created")
        
        tree.column("Select", width=50, anchor='center')
        tree.column("Barcode", width=200)
        tree.column("Serial Number", width=200)
        tree.column("Created", width=200)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree.bind("<Double-1>", self.show_product_preview)

        def toggle_selection(event):
            region = tree.identify("region", event.x, event.y)
            if region == "cell":
                column = tree.identify_column(event.x)
                if column == "#1":  # First column (Select column)
                    item = tree.identify_row(event.y)
                    current_value = tree.item(item, 'values')[0]
                    new_value = "☑" if current_value == "☐" else "☐"
                    
                    updated_values = list(tree.item(item, 'values'))
                    updated_values[0] = new_value
                    tree.item(item, values=tuple(updated_values))
        
        def delete_selected_products():
            selected_items = [
                tree.item(item)['values'][1]  # Get barcode from selected items
                for item in tree.get_children() 
                if tree.item(item)['values'][0] == "☑"
            ]
            
            if not selected_items:
                messagebox.showwarning("Warning", "No products selected")
                return
            
            confirm_message = f"Are you sure you want to delete {len(selected_items)} products?"
            if messagebox.askyesno("Confirm", confirm_message):
                deleted_count = 0
                for barcode in selected_items:
                    if self.product_manager.delete_product(barcode):
                        deleted_count += 1
                
                # Refresh the view
                for item in list(tree.get_children()):
                    if tree.item(item)['values'][1] in selected_items:
                        tree.delete(item)
                
                messagebox.showinfo("Success", f"Deleted {deleted_count} products")

        
        products = self.product_manager.get_all_products()
        for product in products:
            tree.insert(
                "",
                tk.END,
                values=(
                    "☐",
                    product.barcode,
                    product.serial_number,
                    product.created_at
                )
            )
        
        tree.bind("<Button-1>", toggle_selection)
        tree.bind("<Double-1>", self.show_product_preview)

        
        delete_btn = ttk.Button(
            view_window, 
            text="Delete Selected Products", 
            command=delete_selected_products
        )
        delete_btn.pack(pady=10)
    
    def show_product_preview(self, event) -> None:
        tree = event.widget  # Get the Treeview widget from the event
        selected_item = tree.selection()
        if not selected_item:
            return

        item_values = tree.item(selected_item, "values")
        if not item_values:
            return

        barcode = item_values[1]  # Extract barcode from selected row

        product = self.product_manager.retrieve_product(barcode)
        if not product:
            messagebox.showerror("Error", "Product details not found!")
            return

        window_id = str(uuid.uuid4())
        preview_window = tk.Toplevel(self.root)  # Use self.root as parent
        preview_window.title(f"Product Preview - {product.barcode}")
        preview_window.geometry("500x600")

        image = Image.open(io.BytesIO(product.image_data))
        image.thumbnail((400, 400), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)

        self.image_refs[window_id] = photo  # Store reference

        image_label = ttk.Label(preview_window, image=photo)
        image_label.pack(pady=10)

        details_frame = ttk.Frame(preview_window)
        details_frame.pack(pady=10)

        for label, value in [
            ("Barcode", product.barcode),
            ("Serial Number", product.serial_number),
            ("Created At", str(product.created_at))
        ]:
            ttk.Label(details_frame, text=f"{label}: {value}", font=("Helvetica", 12)).pack(anchor='w')

        def on_close():
            if window_id in self.image_refs:
                del self.image_refs[window_id]  
            preview_window.destroy()  

        preview_window.protocol("WM_DELETE_WINDOW", on_close)
    
    def import_dataset(self) -> None:
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("CSV Files", "*.csv"),
                ("Excel Files", "*.xlsx"),
                ("All Files", "*.*")
            ],
            initialdir=os.path.expanduser("~")
        )
        if not file_path:
            return
        
        try:
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path, encoding='utf-8')
            elif file_path.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            else:
                messagebox.showerror("Error", "Unsupported file format")
                return
            
            required_columns = ['barcode', 'image_path']
            if not all(col in df.columns for col in required_columns):
                messagebox.showerror(
                    "Error", 
                    "Dataset must contain 'barcode' and 'image_path' columns"
                )
                return
            
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Importing Products")
            progress_window.geometry("300x150")
            
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(
                progress_window,
                variable=progress_var,
                maximum=len(df)
            )
            progress_bar.pack(pady=20, padx=10, fill=tk.X)
            
            status_label = ttk.Label(progress_window, text="Importing products...")
            status_label.pack(pady=10)
            
            success_count = 0
            for index, row in df.iterrows():
                if os.path.exists(row['image_path']):
                    if self.product_manager.store_product(
                        row['image_path'], 
                        str(row['barcode'])
                    ):
                        success_count += 1
                progress_var.set(index + 1)
                progress_window.update()
            
            progress_window.destroy()
            messagebox.showinfo(
                "Import Complete",
                f"Successfully imported {success_count} products"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import dataset: {str(e)}")

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='product_manager.log'
    )
    
    try:
        db_path = Path(__file__).parent / "products.db"
        db_manager = DatabaseManager(str(db_path))
        product_manager = ProductManager(db_manager)
        
        root = tk.Tk()
        app = ProductUI(root, product_manager)
        root.mainloop()
    except Exception as e:
        logging.critical(f"Application failed to start: {e}")
        raise

if __name__ == "__main__":
    main()