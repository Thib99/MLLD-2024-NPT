import os
from PIL import Image

def resize_images(input_folders, output_folders):
    # Obtener la lista de archivos en ambas carpetas de entrada
    all_files = []
    for folder in input_folders:
        all_files.extend([os.path.join(folder, file) for file in os.listdir(folder)])
    
    # Variables para almacenar el tamaño mínimo encontrado
    min_width = float('inf')
    min_height = float('inf')
    
    # Iterar sobre cada archivo para encontrar el tamaño mínimo
    for file_path in all_files:
        # Abrir la imagen y obtener su tamaño
        img = Image.open(file_path)
        width, height = img.size
        
        # Actualizar el tamaño mínimo si es necesario
        min_width = min(min_width, width)
        min_height = min(min_height, height)
    
    # Iterar sobre cada carpeta de entrada y salida
    for input_folder, output_folder in zip(input_folders, output_folders):
        # Obtener la lista de archivos en la carpeta de entrada
        files = os.listdir(input_folder)
        
        # Iterar sobre cada archivo para reescalarlo al tamaño mínimo
        for file in files:
            # Combinar la ruta de la carpeta de entrada con el nombre de archivo
            file_path = os.path.join(input_folder, file)
            
            # Abrir la imagen y reescalarla al tamaño mínimo
            img = Image.open(file_path)
            img_resized = img.resize((min_width, min_height))
            
            # Combinar la ruta de la carpeta de salida con el nombre de archivo
            output_file_path = os.path.join(output_folder, file)
            
            # Guardar la imagen reescalada en la carpeta de salida como PNG
            img_resized.save(output_file_path, "PNG")

# Especifica las carpetas de entrada y salida para las imágenes
input_folders = ["MLLD-2024-NPT/Dataset/no", "MLLD-2024-NPT/Dataset/yes"]
output_folders = ["MLLD-2024-NPT/Dataset/no", "MLLD-2024-NPT/Dataset/yes"]
# Llama a la función para reescalar las imágenes
resize_images(input_folders, output_folders)
