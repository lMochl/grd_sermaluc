import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm 
import numpy as np
import pickle
import os

file_path = "GRD_2022.csv"
df = pd.read_csv(file_path)

df['FECHA_NACIMIENTO'] = pd.to_datetime(df['FECHA_NACIMIENTO'], errors='coerce')
df['FECHA_INGRESO'] = pd.to_datetime(df['FECHA_INGRESO'], errors='coerce')
df['FECHAALTA'] = pd.to_datetime(df['FECHAALTA'], errors='coerce')

df['AÑO_NACIMIENTO'] = df['FECHA_NACIMIENTO'].dt.year
df['AÑO_INGRESO'] = df['FECHA_INGRESO'].dt.year
df['MES_INGRESO'] = df['FECHA_INGRESO'].dt.month
df['DIA_INGRESO'] = df['FECHA_INGRESO'].dt.day
df['AÑO_ALTA'] = df['FECHAALTA'].dt.year
df['MES_ALTA'] = df['FECHAALTA'].dt.month
df['DIA_ALTA'] = df['FECHAALTA'].dt.day

df['FECHA_NACIMIENTO'].fillna(pd.NaT, inplace=True)
df['FECHA_INGRESO'].fillna(pd.NaT, inplace=True)
df['FECHAALTA'].fillna(pd.NaT, inplace=True)

categorical_cols = ['SEXO', 'TIPO_INGRESO', 'DIAGNOSTICO1']
top_categories = {
    'SEXO': ['HOMBRE', 'MUJER'],
    'TIPO_INGRESO': ['URGENCIA', 'PROGRAMADA', 'OBSTETRICA'],
    'DIAGNOSTICO1': df['DIAGNOSTICO1'].value_counts().nlargest(80).index.tolist()
}

for col, categories in top_categories.items():
    df[col] = pd.Categorical(df[col], categories=categories)

diagnostico_mapping = {i: top_categories['DIAGNOSTICO1'][i] for i in range(len(top_categories['DIAGNOSTICO1']))}
print("Lista de diagnósticos disponibles:")
for index, diag in diagnostico_mapping.items():
    print(f"Posición {index}: {diag}")

sexo_mapping = {i: top_categories['SEXO'][i] for i in range(len(top_categories['SEXO']))}
print("\nLista de sexos disponibles:")
for index, sexo in sexo_mapping.items():
    print(f"Posición {index}: {sexo}")

tipo_ingreso_mapping = {i: top_categories['TIPO_INGRESO'][i] for i in range(len(top_categories['TIPO_INGRESO']))}
print("\nLista de tipos de ingreso disponibles:")
for index, tipo in tipo_ingreso_mapping.items():
    print(f"Posición {index}: {tipo}")

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df[['AÑO_NACIMIENTO', 'AÑO_INGRESO', 'MES_INGRESO', 'DIA_INGRESO'] + 
       [col for col in df.columns if col.startswith('SEXO_') or col.startswith('TIPO_INGRESO_')  or col.startswith('DIAGNOSTICO1_')]]
y = df[['AÑO_ALTA', 'MES_ALTA', 'DIA_ALTA']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_path = 'modelo_random_forest.pkl'
if os.path.exists(model_path):
    print("Cargando el modelo entrenado...")
    with open(model_path, 'rb') as file:
        modelo = pickle.load(file)
else:
    n_estimators = 50
    modelo = RandomForestRegressor(n_estimators=n_estimators)

    with tqdm(total=n_estimators, desc="Entrenando el modelo") as pbar:
        for i in range(n_estimators):
            modelo.n_estimators = i + 1
            modelo.fit(X_train, y_train)
            pbar.update(1)

    with open(model_path, 'wb') as file:
        pickle.dump(modelo, file)

y_pred = modelo.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

print(f"Precisión del modelo: {modelo.score(X_test, y_test)}")
print(f"Error absoluto medio (MAE): {mae}")

paciente_simulado = {
    'AÑO_NACIMIENTO': [1980],
    'AÑO_INGRESO': [2024],
    'MES_INGRESO': [9],
    'DIA_INGRESO': [15],
    'SEXO': [1],
    'TIPO_INGRESO': [1],
    'DIAGNOSTICO1': [1],
}

paciente_df = pd.DataFrame(paciente_simulado)

for col in X_train.columns:
    if col not in paciente_df.columns:
        paciente_df[col] = 0

paciente_df = paciente_df[X_train.columns]

prediccion_alta = modelo.predict(paciente_df)
prediccion_alta_redondeada = np.round(prediccion_alta).astype(int)

print(f"Predicción de alta para el paciente simulado (mes, día): {prediccion_alta_redondeada}")
