import pandas as pd
import os
import concurrent.futures
import numpy as np

# Directorios de los archivos
callgraph_dir = "../MSCallGraphExtracted"
resource_dir = "../MSResource"
output_file = "filtered_data_um.csv"

# Obtener listas de archivos
callgraph_files = sorted([f for f in os.listdir(callgraph_dir) if f.startswith("MSCallGraph_") and f.endswith(".csv")])
resource_files = sorted([f for f in os.listdir(resource_dir) if f.startswith("MSResource_") and f.endswith(".csv")])

# Dividir los archivos en 8 grupos para paralelismo
num_threads = 8
callgraph_chunks = np.array_split(callgraph_files, num_threads)

def process_files(callgraph_subset, thread_id):
    local_output_file = f"temp_filtered_data_{thread_id}.csv"
    first_write = True  # Para escribir el encabezado solo en la primera iteración

    for callgraph_file in callgraph_subset:
        callgraph_df = pd.read_csv(os.path.join(callgraph_dir, callgraph_file), index_col=0)
        
        for resource_file in resource_files:
            resource_df = pd.read_csv(os.path.join(resource_dir, resource_file), index_col=0)
            
            # Realizar el merge
            merged_df = pd.merge(callgraph_df, resource_df, how="inner",
                                 left_on=["um", "timestamp"], 
                                 right_on=["msname", "timestamp"])
            
            # Guardar en un archivo temporal
            merged_df.to_csv(local_output_file, index=False, mode="w" if first_write else "a", header=first_write)
            first_write = False  # Después de la primera escritura, los demás serán append

            del resource_df, merged_df  # Liberar memoria
            print(f"se acaba de mezclar el archivo {callgraph_file} con el archivo {resource_file}")
        
        del callgraph_df  # Liberar memoria después de procesar un archivo

    print(f"Thread {thread_id} ha terminado su procesamiento.")

# Ejecutar en paralelo
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = {executor.submit(process_files, chunk, i): i for i, chunk in enumerate(callgraph_chunks)}
    concurrent.futures.wait(futures)

# Combinar los archivos temporales en el resultado final
with open(output_file, "w") as outfile:
    first_file = True
    for i in range(num_threads):
        temp_file = f"temp_filtered_data_{i}.csv"
        if os.path.exists(temp_file):
            with open(temp_file, "r") as infile:
                if first_file:
                    outfile.write(infile.read())  # Copiar con encabezado
                    first_file = False
                else:
                    next(infile)  # Saltar encabezado
                    outfile.write(infile.read())  # Copiar sin encabezado
            os.remove(temp_file)  # Eliminar archivo temporal

print(f"El archivo final '{output_file}' ha sido generado exitosamente.")

