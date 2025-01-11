
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_model(df):
    """Entrena un modelo de predicción."""
    # Seleccionar características y objetivo
    X = df[['Consumo', 'Temperatura', 'Mes', 'DíaSemana']]  # Ajusta con tus columnas
    y = df['Precio_kWh']

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar modelo
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Guardar el modelo
    joblib.dump(model, 'results/model.pkl')

    return model, X_test, y_test
