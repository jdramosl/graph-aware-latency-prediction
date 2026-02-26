import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import joblib
import time
import sys
import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =====================================================
# CONFIGURACIONES
# =====================================================
log_file = "model_evaluation_log_240_512.txt"
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
log = open(log_file, "w", buffering=1)
sys.stdout = Tee(sys.stdout, log)

save_models = True
load_saved_models = False
models_dir = "saved_models_512"
os.makedirs(models_dir, exist_ok=True)

# =====================================================
# CARGA DE DATOS
# =====================================================
scaler = MinMaxScaler()
df = pd.read_csv('/home/juan/Documents/lstm/filtered_data_um_with_extra_columns_parallel.csv')

numerical_columns_simple = ['instance_cpu_usage', 'instance_memory_usage', 'timestamp']
numerical_columns_complex = [
    'instance_cpu_usage', 'instance_memory_usage', 'betweenness_centrality_um',
    'closeness_centrality_um', 'degree_centrality_um', 'eigenvector_centrality_um',
    'pagerank_um', 'community_um', 'betweenness_centrality_dm', 'closeness_centrality_dm',
    'degree_centrality_dm', 'eigenvector_centrality_dm', 'pagerank_dm', 'community_dm','eccentricity_um','eccentricity_dm',
    'number_of_nodes', 'number_of_edges', 'density', 'average_clustering',
    'average_degree', 'diameter', 'radius', 'timestamp'
]

df_dropna_simple = df[numerical_columns_simple + ['rt']].dropna().reset_index(drop=True)
X_dropna_simple = scaler.fit_transform(df_dropna_simple[numerical_columns_simple])
y_dropna_simple = df_dropna_simple['rt']

df_dropna_complex = df[numerical_columns_complex + ['rt']].dropna().reset_index(drop=True)
X_dropna_complex = scaler.fit_transform(df_dropna_complex[numerical_columns_complex])
y_dropna_complex = df_dropna_complex['rt']

# =====================================================
# FUNCIÓN DE EVALUACIÓN LSTM
# =====================================================
def evaluate_lstm(X_train, y_train, X_test, y_test, feature_names, scenario, model_path):
    # Reshape para LSTM [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    start_time = time.time()

    if load_saved_models and os.path.exists(model_path):
        print(f"Cargando modelo LSTM guardado: {model_path}")
        model = load_model(model_path)
    else:
        print(f"Entrenando modelo LSTM guardado: {model_path}")
        model = Sequential([
            LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), activation='tanh', return_sequences=True),
            LSTM(64, return_sequences=True),
            LSTM(32, return_sequences=True),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mae')
        history = model.fit(X_train, y_train, epochs=720, batch_size=1024, verbose=1, validation_split=0.2)
        if save_models:
            model.save(model_path)
            print(f"Modelo LSTM guardado en: {model_path}")

    train_time = time.time() - start_time
    y_pred = model.predict(X_test).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{scenario} - LSTM - MAE: {mae:.4f} - Tiempo de entrenamiento: {train_time:.2f} segundos")

    # Gráfico de convergencia (loss)
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(f"LSTM Loss ({scenario})")
    plt.xlabel('Epoch')
    plt.ylabel('MAE Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"lstm_loss_{scenario}_512.png")
    plt.close()

    return model, mae, train_time

# =====================================================
# SEPARAR DATOS
# =====================================================
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_dropna_simple, y_dropna_simple, test_size=0.2, random_state=42
)
X_train_complex, X_test_complex, y_train_complex, y_test_complex = train_test_split(
    X_dropna_complex, y_dropna_complex, test_size=0.2, random_state=42
)

# =====================================================
# ENTRENAMIENTO LSTM PARA AMBOS ESCENARIOS
# =====================================================
for scenario, (X_train, X_test, y_train, y_test, feature_names) in {
    "Simple": (X_train_simple, X_test_simple, y_train_simple, y_test_simple, numerical_columns_simple)
}.items():
    print(f"\nEvaluación LSTM en el escenario {scenario}:")
    model_path = os.path.join(models_dir, f"LSTM_{scenario}.keras")
    evaluate_lstm(X_train, y_train, X_test, y_test, feature_names, scenario, model_path)

# =====================================================
# FINALIZACIÓN
# =====================================================
sys.stdout = sys.__stdout__
log.close()
print("✅ Entrenamiento LSTM finalizado. Resultados guardados en 'model_evaluation_log.txt'")

