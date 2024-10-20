import pandas as pd

file_path = 'GRD_PUBLICO_EXTERNO_2022.txt'

df = pd.read_csv(file_path, delimiter='|', encoding='utf-16', on_bad_lines='skip')

print(df.head())

df.to_csv('GRD_PUBLICO_EXTERNO_2022.csv', index=False, encoding='utf-8')
