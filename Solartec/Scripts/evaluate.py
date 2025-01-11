from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt

def evaluate_model(model, X_test, y_test):
    """Evalúa el modelo y genera gráficas."""
    y_pred = model.predict(X_test)

    # Calcular métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)

    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')

    # Graficar resultados
    plt.plot(y_test.values, label='Real')
    plt.plot(y_pred, label='Predicción')
    plt.legend()
    plt.show()
