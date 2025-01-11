from Scripts.preprocess import load_data, clean_data, feature_engineering
from Scripts.train_model import train_model
from Scripts.evaluate import evaluate_model
import os

# Cargar datos
data_path = os.path.join('data', 'Prueba.xlsx')
df = load_data(data_path)

# Preparar datos
df = clean_data(df)
df = feature_engineering(df)

# Entrenar modelo
model, X_test, y_test = train_model(df)

# Evaluar modelo
evaluate_model(model, X_test, y_test)
print("¡El archivo main.py se ejecutó correctamente!")