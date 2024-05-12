from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from g4f.client import Client
from g4f.client import AsyncClient
import asyncio

app = Flask(__name__, template_folder='.')
socketio = SocketIO(app)
client = AsyncClient()

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@app.route('/', methods=['GET', 'POST'])
async def index():
    if request.method == 'POST':
        # Get data from form
        edad = int(request.form.get('edad'))
        sexo = request.form.get('sexo')
        # Convert 'sexo' to 0/1
        sexo = 0 if sexo == 'M' else 1
        fiebre = int(request.form.get('fiebre', 0))
        cefalea = int(request.form.get('cefalea', 0))
        dolrretroo = int(request.form.get('dolrretroo', 0))
        malgias = int(request.form.get('malgias', 0))
        artralgia = int(request.form.get('artralgia', 0))
        erupcionr = int(request.form.get('erupcionr', 0))

        # Paso 1: Cargar el dataset y preprocesamiento de datos
        data = pd.read_csv('dengue.csv')
        # Convertir variables categóricas 'Si'/'No' a 1/0 y 'sexo' a 0/1
        yes_no_columns = ['fiebre', 'cefalea', 'dolrretroo', 'malgias', 'artralgia', 'erupcionr', 'dolor_abdo', 'vomito', 'diarrea', 'hipotensio', 'hepatomeg', 'pac_hos']
        for col in yes_no_columns:
            data[col] = data[col].map({'Si': 1, 'No': 0})
        data['sexo'] = data['sexo'].map({'M': 0, 'F': 1})
        # Encoding other categorical variables using LabelEncoder
        label_columns = ['tip_cas', 'clasfinal', 'conducta']
        label_encoders = {}
        for col in label_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

        # Define features
        features = ['edad', 'sexo', 'fiebre', 'cefalea', 'dolrretroo', 'malgias', 'artralgia', 'erupcionr']

        # Paso 2: División de datos
        X = data[features].values
        y_clasfinal = data['clasfinal']
        y_tip_cas = data['tip_cas']
        X_train, X_test, y_clasfinal_train, y_clasfinal_test, y_tip_cas_train, y_tip_cas_test = train_test_split(X, y_clasfinal, y_tip_cas, test_size=0.2, random_state=42)

        # Paso 3 y 4: Selección y entrenamiento del modelo
        model_clasfinal = RandomForestClassifier(n_estimators=100, random_state=42)
        model_tip_cas = RandomForestClassifier(n_estimators=100, random_state=42)
        model_clasfinal.fit(X_train, y_clasfinal_train)
        model_tip_cas.fit(X_train, y_tip_cas_train)

        # Make predictions
        user_data = [edad, sexo, fiebre, cefalea, dolrretroo, malgias, artralgia, erupcionr]
        predicted_clasfinal = model_clasfinal.predict([user_data])[0]
        predicted_tip_cas = model_tip_cas.predict([user_data])[0]

        # Decode the predictions
        predicted_clasfinal = label_encoders['clasfinal'].inverse_transform([predicted_clasfinal])[0]
        predicted_tip_cas = label_encoders['tip_cas'].inverse_transform([predicted_tip_cas])[0]

        # Make the OpenAI API call
        user_symptoms = f"Edad: {edad}, Sexo: {sexo}, Fiebre: {'Sí' if fiebre else 'No'}, Dolor de cabeza: {'Sí' if cefalea else 'No'}, Dolor detrás de los ojos: {'Sí' if dolrretroo else 'No'}, Dolores musculares: {'Sí' if malgias else 'No'}, Dolor en las articulaciones: {'Sí' if artralgia else 'No'}, Erupción cutánea: {'Sí' if erupcionr else 'No'}"
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un asistente virtual medico"},
                {"role": "user", "content": f"Dado los siguientes síntomas: {user_symptoms}, el tipo de caso predicho es: {predicted_tip_cas}, y la clasificación final predicha es: {predicted_clasfinal}. Por favor, dame una respuesta médica adecuada."}
            ],
            stream=True
        )

        gpt_response = ""
        async for chunk in stream:
            partial_response = chunk.choices[0].delta.content or ""
            gpt_response += chunk.choices[0].delta.content or ""
            # Emit the partial response to the client
            socketio.emit('chunk', {'data': partial_response})

        # Emit the final response to the client
        socketio.emit('response', {'predicted_clasfinal': predicted_clasfinal, 'predicted_tip_cas': predicted_tip_cas, 'gpt_response': gpt_response})

    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, debug=True)