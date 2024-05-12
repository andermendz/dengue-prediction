from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from g4f.client import Client

app = Flask(__name__, template_folder='.')
client = Client()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get data from form
        edad = int(request.form.get('edad'))
        sexo = 1 if request.form.get('sexo') == 'F' else 0
        fiebre = 1 if request.form.get('fiebre') else 0
        cefalea = 1 if request.form.get('cefalea') else 0
        dolrretroo = 1 if request.form.get('dolrretroo') else 0
        malgias = 1 if request.form.get('malgias') else 0
        artralgia = 1 if request.form.get('artralgia') else 0
        erupcionr = 1 if request.form.get('erupcionr') else 0

        # Paso 1: Cargar el dataset y preprocesamiento de datos
        data = pd.read_csv('dengue.csv')
        # Convertir variables categóricas 'Si'/'No' a 1/0 y 'sexo' a 0/1
        yes_no_columns = ['fiebre', 'cefalea', 'dolrretroo', 'malgias', 'artralgia', 'erupcionr', 'dolor_abdo', 'vomito', 'diarrea', 'hipotensio', 'hepatomeg', 'pac_hos']
        for col in yes_no_columns:
            data[col] = data[col].map({'Si': 1, 'No': 0})
        # Mapping 'sexo' to 0/1
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
        X = data[features]
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

        # AI API call
        user_symptoms = f"Edad: {edad}, Sexo: {'F' if sexo == 1 else 'M'}, Fiebre: {'Sí' if fiebre == 1 else 'No'}, Dolor de cabeza: {'Sí' if cefalea == 1 else 'No'}, Dolor detrás de los ojos: {'Sí' if dolrretroo == 1 else 'No'}, Dolores musculares: {'Sí' if malgias == 1 else 'No'}, Dolor en las articulaciones: {'Sí' if artralgia == 1 else 'No'}, Erupción cutánea: {'Sí' if erupcionr == 1 else 'No'}"
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Eres un asistente virtual medico"},
                {"role": "user", "content": f"Dado los siguientes síntomas: {user_symptoms}, el tipo de caso predicho es: {predicted_tip_cas}, y la clasificación final predicha es: {predicted_clasfinal}. Por favor, dame una respuesta médica adecuada."}
            ]
        )
        gpt_response = response.choices[0].message.content

        # Return the result to the HTML page
        return jsonify(predicted_clasfinal=predicted_clasfinal, predicted_tip_cas=predicted_tip_cas, gpt_response=gpt_response)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)