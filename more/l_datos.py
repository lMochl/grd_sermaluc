import pandas as pd

file_path = "GRD_PUBLICO_EXTERNO_2022.csv"
df = pd.read_csv(file_path, delimiter=',', encoding='utf-8')

columnas_a_mantener = [
    'SEXO', 'FECHA_NACIMIENTO', 'TIPO_INGRESO', 'FECHA_INGRESO', 'FECHAALTA','DIAGNOSTICO1'
]

df_limpio = df[columnas_a_mantener]

df_limpio.to_csv("GRD_2022.csv", index=False, encoding='utf-8')

print(df_limpio.head())
