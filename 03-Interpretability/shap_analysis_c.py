import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import os
import numpy as np

# =====================================================
# CONFIGURACIÓN
# =====================================================
models_dir = "saved_models"  # carpeta donde están los .pkl
output_dir = "shap_plots_d"
os.makedirs(output_dir, exist_ok=True)

# =====================================================
# CARGA DE DATOS (solo escenario COMPLEJO)
# =====================================================
scaler = MinMaxScaler()
df = pd.read_csv('D:/preprocess_sem_5/ICPE2024_DataChallenge_jan_27/filtered_data_um_with_extra_columns_parallel.csv')

numerical_columns_complex = [
    'instance_cpu_usage', 'instance_memory_usage', 'betweenness_centrality_um',
    'closeness_centrality_um', 'degree_centrality_um', 'eigenvector_centrality_um',
    'pagerank_um', 'community_um', 'betweenness_centrality_dm', 'closeness_centrality_dm',
    'degree_centrality_dm', 'eigenvector_centrality_dm', 'pagerank_dm', 'community_dm', 'eccentricity_um', 'eccentricity_dm',
    'number_of_nodes', 'number_of_edges', 'density', 'average_clustering',
    'average_degree', 'diameter', 'radius', 'timestamp'
]

df_dropna_complex = df[numerical_columns_complex + ['rt']].dropna().reset_index(drop=True)
X_complex_scaled = scaler.fit_transform(df_dropna_complex[numerical_columns_complex])
X_complex = pd.DataFrame(X_complex_scaled, columns=numerical_columns_complex)
y_complex = df_dropna_complex['rt']

# Para ahorrar tiempo, tomamos una muestra (opcional)
X_sample = shap.utils.sample(X_complex, 1000, random_state=42)
y_sample = y_complex[:len(X_sample)]

# =====================================================
# MODELOS A CARGAR
# =====================================================
models = {
    'Linear Regression': {
        'file': 'Linear_Regression_default_Complejo.pkl',
        'type': 'linear'
    },
    'Decision Tree': {
        'file': 'Decision_Tree_dt_min_samples_10_Complejo.pkl',
        'type': 'tree'
    },
    'Random Forest': {
        'file': 'Random_Forest_rf_200_minleaf_5_Complejo.pkl',
        'type': 'tree'
    },
    'XGBoost': {
        'file': 'XGBoost_xgb_depth_7_lr_003_Complejo.pkl',
        'type': 'tree'
    }
}


# =====================================================
# FUNCIÓN PARA GENERAR PLOTS SHAP
# =====================================================
def generate_shap_plot(model_name, model, model_type, X, feature_names):
    print(f"Generando SHAP para {model_name}...")

    if model_type == 'linear':
        explainer = shap.Explainer(model, X, feature_names=feature_names)
    elif model_type == 'tree':
        explainer = shap.TreeExplainer(model)
    else:
        print(f"⚠️ Tipo de modelo no soportado: {model_type}")
        return

    shap_values = explainer(X)

    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, show=False, max_display=30)
    plt.title(f"Importancia de los parámetros - {model_name}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            f"feature_importance_whole_{model_name.replace(' ', '_')}.png"
        ),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()



# =====================================================
# CARGAR MODELOS GUARDADOS Y GENERAR GRÁFICAS
# =====================================================
for model_name, info in models.items():
    model_path = os.path.join(models_dir, info['file'])

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        generate_shap_plot(
            model_name,
            model,
            info['type'],
            X_sample,
            numerical_columns_complex
        )
    else:
        print(f"⚠️ No se encontró el modelo {model_path}")


print("\n🎨 Todos los gráficos SHAP del escenario 'Complejo' han sido generados y guardados.")
