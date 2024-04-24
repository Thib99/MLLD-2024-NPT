import os
from PIL import Image

def resize_images(input_folder, output_folder):
    # Obtener la lista de archivos en la carpeta de entrada
    files = os.listdir(input_folder)
    
    # Variable para almacenar el tamaño mínimo encontrado
    min_width = float('inf')
    min_height = float('inf')
    
    # Iterar sobre cada archivo para encontrar el tamaño mínimo
    for file in files:
        # Combinar la ruta de la carpeta de entrada con el nombre de archivo
        file_path = os.path.join(input_folder, file)
        
        # Abrir la imagen y obtener su tamaño
        img = Image.open(file_path)
        width, height = img.size
        
        # Actualizar el tamaño mínimo si es necesario
        min_width = min(min_width, width)
        min_height = min(min_height, height)
    
    # Iterar sobre cada archivo para reescalarlo al tamaño mínimo
    for file in files:
        # Combinar la ruta de la carpeta de entrada con el nombre de archivo
        file_path = os.path.join(input_folder, file)
        
        # Abrir la imagen y reescalarla al tamaño mínimo
        img = Image.open(file_path)
        img_resized = img.resize((min_width, min_height))
        
        # Combinar la ruta de la carpeta de salida con el nombre de archivo
        output_file_path = os.path.join(output_folder, file)
        
        # Guardar la imagen reescalada en la carpeta de salida
        img_resized.save(output_file_path)

# Especifica las carpetas de entrada y salida
input_folder = "MLLD-2024-NPT/Dataset/no"
output_folder = "MLLD-2024-NPT/Dataset/yes-r"

# Llama a la función para reescalar las imágenes
resize_images(input_folder, output_folder)
