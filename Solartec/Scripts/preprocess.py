import pandas as pd

def load_data(file_path):
    """Carga datos desde un archivo Excel."""
    return pd.read_excel(file_path)

def clean_data(df):
    """Limpia y prepara los datos."""
    # Eliminar filas con valores nulos
    df = df.dropna()

    # Convertir fecha si aplica
    if 'Fecha' in df.columns:
        df['Fecha'] = pd.to_datetime(df['Fecha'])

    return df

def feature_engineering(df):
    """Crea características adicionales."""
    if 'Fecha' in df.columns:
        df['Mes'] = df['Fecha'].dt.month
        df['DíaSemana'] = df['Fecha'].dt.dayofweek

    return df
