import pandas as pd
import os
import multiprocessing as mp
from functools import partial

# Rutas
base_dir = 'D:\\preprocess_sem_5'
log_file = os.path.join('D:\\preprocess_sem_5', 'filtered_data_um.csv')
louvain_dir = os.path.join(base_dir, 'ICPE2024_DataChallenge_jan_27', 'louvain_new')
output_file = os.path.join(base_dir, 'ICPE2024_DataChallenge_jan_27\\filtered_data_um_with_extra_columns_parallel.csv')

# Columnas para centrality
centrality_metrics = ['betweenness_centrality', 'closeness_centrality', 'degree_centrality',
                      'eigenvector_centrality', 'pagerank', 'community', 'eccentricity']

# Columnas para metrics
metrics_keys = ['number_of_nodes', 'number_of_edges', 'density', 'average_clustering',
                'average_degree', 'diameter', 'radius']

def read_centrality(rpcid):
    path = os.path.join(louvain_dir, f"{rpcid}_centrality.csv.csv")
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except:
            return pd.DataFrame()
    return pd.DataFrame()

def read_metrics(rpcid):
    path = os.path.join(louvain_dir, f"{rpcid}_metrics.csv.txt")
    metrics = {}
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                for line in f:
                    if ':' in line:
                        k, v = line.strip().split(':')
                        metrics[k.strip()] = float(v.strip())
        except:
            pass
    return metrics

def process_row(row, queue=None, total=None):
    rpcid = str(row['rpcid'])
    um = str(row['um'])
    dm = str(row['dm'])

    # Copia la fila original como diccionario
    result = row.to_dict()

    # Leer centrality y metrics
    centrality_df = read_centrality(rpcid)
    metrics_dict = read_metrics(rpcid)

    # Extraer métricas um y dm
    if not centrality_df.empty:
        for node_type, node_val in [('um', um), ('dm', dm)]:
            node_row = centrality_df[centrality_df['node'] == node_val]
            if not node_row.empty:
                for metric in centrality_metrics:
                    result[f"{metric}_{node_type}"] = node_row.iloc[0][metric]
            else:
                for metric in centrality_metrics:
                    result[f"{metric}_{node_type}"] = None
    else:
        for node_type in ['um', 'dm']:
            for metric in centrality_metrics:
                result[f"{metric}_{node_type}"] = None

    # Agregar métricas generales
    for key in metrics_keys:
        result[key] = metrics_dict.get(key, None)

    # Progreso
    if queue is not None and total is not None:
        queue.put(1)
        done = queue.qsize()
        if done % max(1, total // 100) == 0:
            print(f'Progreso: {done}/{total} filas ({int(done / total * 100)}%)')

    return result

def main():
    # Leer archivo principal
    df = pd.read_csv(log_file)
    total_rows = len(df)

    # Multiprocessing setup
    num_processes = max(2, mp.cpu_count() - 1)
    manager = mp.Manager()
    queue = manager.Queue()

    # Preparar función parcial
    process_func = partial(process_row, queue=queue, total=total_rows)

    # Procesar en paralelo
    print(f'Procesando {total_rows} filas con {num_processes} procesos...')
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_func, [row for _, row in df.iterrows()])

    # Crear nuevo DataFrame con resultados
    new_df = pd.DataFrame(results)

    # Guardar en archivo
    new_df.to_csv(output_file, index=False)
    print(f'\n✅ Archivo generado: {output_file}')

if __name__ == '__main__':
    main()
