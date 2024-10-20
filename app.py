from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import gdown

model_path = 'modelo_random_forest.pkl'

with open(model_path, 'rb') as file:
    modelo = pickle.load(file)

diagnosticos = {i: f"Diagnóstico {i}" for i in range(1, 81)}

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    fecha_nacimiento = request.form['fecha_nacimiento']
    año_nacimiento = datetime.strptime(fecha_nacimiento, '%Y-%m-%d').year

    fecha_ingreso = request.form['fecha_ingreso']
    fecha_ingreso_dt = datetime.strptime(fecha_ingreso, '%Y-%m-%d')
    año_ingreso = fecha_ingreso_dt.year
    mes_ingreso = fecha_ingreso_dt.month
    dia_ingreso = fecha_ingreso_dt.day

    sexo = int(request.form['sexo'])
    tipo_ingreso = int(request.form['tipo_ingreso'])
    diagnostico = int(request.form['diagnostico'])

    paciente_simulado = {
        'AÑO_NACIMIENTO': [año_nacimiento],
        'AÑO_INGRESO': [año_ingreso],
        'MES_INGRESO': [mes_ingreso],
        'DIA_INGRESO': [dia_ingreso],
        'SEXO': [sexo],
        'TIPO_INGRESO': [tipo_ingreso],
        'DIAGNOSTICO1': [diagnostico],
    }

    paciente_df = pd.DataFrame(paciente_simulado)

    for col in modelo.feature_names_in_:
        if col not in paciente_df.columns:
            paciente_df[col] = 0

    paciente_df = paciente_df[modelo.feature_names_in_]

    prediccion_alta = modelo.predict(paciente_df)
    prediccion_alta_redondeada = np.round(prediccion_alta).astype(int)

    mes_alta = prediccion_alta_redondeada[0][1]
    dia_alta = prediccion_alta_redondeada[0][2]

    return render_template('index.html', diagnosticos=diagnosticos, prediccion=f"Fecha estimada de alta: {mes_alta}-{dia_alta}")

if __name__ == '__main__':
    app.run(debug=True)
