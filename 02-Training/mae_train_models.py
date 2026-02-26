import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import time
import sys
import os
import numpy as np

# =====================================================
# CONFIGURACIONES
# =====================================================
log_file = "model_evaluation_log_dist_opt.txt"
sys.stdout = open(log_file, "w")

save_models = True         # Guardar modelos después del entrenamiento
load_saved_models = False  # Si es True, carga los modelos guardados y no entrena
models_dir = "saved_models_dist_opt"
os.makedirs(models_dir, exist_ok=True)

# =====================================================
# CARGA DE DATOS
# =====================================================
scaler = MinMaxScaler()
df = pd.read_csv('D:/preprocess_sem_5/preprocessed_Data/filtered_data_um_with_extra_columns_parallel.csv')

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
# FUNCIÓN DE EVALUACIÓN
# =====================================================
def evaluate_model(model, X_train, y_train, X_test, y_test, feature_names, model_name, scenario):
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    train_time = time.time() - start_time
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{scenario} - {model_name} - MAE: {mae:.4f} - Tiempo de entrenamiento: {train_time:.2f} segundos")

    # Importancias de características
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        top_indices = np.argsort(importance)[-6:][::-1]
        print(f"Top 6 features más importantes ({model_name} - {scenario}):")
        for i in top_indices:
            print(f"  {feature_names[i]}: {importance[i]:.4f}")
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importance)), importance)
        plt.xticks(range(len(importance)), feature_names, rotation=90)
        plt.title(f"Importancia de Variables - {model_name} ({scenario})")
        plt.tight_layout()
        plt.savefig(f"feature_importance_{model_name}_{scenario}.png")
        plt.close()

    elif isinstance(model, LinearRegression):
        importance = model.coef_
        top_indices = np.argsort(np.abs(importance))[-6:][::-1]
        print(f"Top 6 coeficientes más altos ({model_name} - {scenario}):")
        for i in top_indices:
            print(f"  {feature_names[i]}: {importance[i]:.4f}")
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importance)), importance)
        plt.xticks(range(len(importance)), feature_names, rotation=90)
        plt.title(f"Coeficientes - {model_name} ({scenario})")
        plt.tight_layout()
        plt.savefig(f"coefficients_{model_name}_{scenario}.png")
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
# MODELOS
# =====================================================
model_scenarios = {
    'Linear Regression': {
        'default': LinearRegression(),
    },

    'Decision Tree': {
        'dt_depth_5': DecisionTreeRegressor(max_depth=5, random_state=42),
        'dt_depth_10': DecisionTreeRegressor(max_depth=10, random_state=42),
        'dt_depth_20': DecisionTreeRegressor(max_depth=20, random_state=42),
        'dt_min_samples_5': DecisionTreeRegressor(min_samples_leaf=5, random_state=42),
        'dt_min_samples_10': DecisionTreeRegressor(min_samples_leaf=10, random_state=42),
    },

    'Random Forest': {
        'rf_100_depth_10': RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        'rf_200_depth_15': RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        'rf_300_depth_20': RandomForestRegressor(
            n_estimators=300, max_depth=20, random_state=42, n_jobs=-1
        ),
        'rf_200_minleaf_5': RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        'rf_500_depth_25': RandomForestRegressor(
            n_estimators=500, max_depth=25, random_state=42, n_jobs=-1
        ),
    },

    'XGBoost': {
        'xgb_depth_3_lr_01': xgb.XGBRegressor(
            max_depth=3, learning_rate=0.1, n_estimators=300,
            objective="reg:squarederror", random_state=42
        ),
        'xgb_depth_5_lr_005': xgb.XGBRegressor(
            max_depth=5, learning_rate=0.05, n_estimators=500,
            objective="reg:squarederror", random_state=42
        ),
        'xgb_depth_7_lr_003': xgb.XGBRegressor(
            max_depth=7, learning_rate=0.03, n_estimators=700,
            objective="reg:squarederror", random_state=42
        ),
        'xgb_subsample_08': xgb.XGBRegressor(
            max_depth=5, learning_rate=0.05, n_estimators=500,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", random_state=42
        ),
        'xgb_regularized': xgb.XGBRegressor(
            max_depth=5, learning_rate=0.05, n_estimators=500,
            reg_alpha=1.0, reg_lambda=1.0,
            objective="reg:squarederror", random_state=42
        ),
    }
}


# =====================================================
# ENTRENAMIENTO / CARGA DE MODELOS
# =====================================================
for scenario, (X_train, X_test, y_train, y_test, feature_names) in {
    "Simple": (X_train_simple, X_test_simple, y_train_simple, y_test_simple, numerical_columns_simple),
    "Complejo": (X_train_complex, X_test_complex, y_train_complex, y_test_complex, numerical_columns_complex)
}.items():
    print(f"\nEvaluación en el escenario {scenario}:")

    for model_name, scenarios in model_scenarios.items():
        for scenario_name, model in scenarios.items():

            full_model_name = f"{model_name}_{scenario_name}"
            model_path = os.path.join(
                models_dir, f"{full_model_name.replace(' ', '_')}_{scenario}.pkl"
            )


            if load_saved_models and os.path.exists(model_path):
                print(f"Cargando modelo guardado: {model_path}")
                model = joblib.load(model_path)
            else:
                model, mae, train_time = evaluate_model(model, X_train, y_train, X_test, y_test, feature_names, full_model_name, scenario)

                if save_models:
                    joblib.dump(model, model_path)
                    print(f"Modelo guardado en: {model_path}")

# =====================================================
# FINALIZACIÓN
# =====================================================
sys.stdout.close()
sys.stdout = sys.__stdout__
print("✅ Entrenamiento finalizado. Resultados guardados en 'model_evaluation_log.txt'")
