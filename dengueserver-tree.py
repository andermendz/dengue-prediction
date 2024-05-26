from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
import os
import graphviz

app = Flask(__name__, template_folder='.')
socketio = SocketIO(app)

# Diccionario para almacenar los codificadores de etiquetas
label_encoders = {}

# Preprocesamiento y entrenamiento del modelo (UNA SOLA VEZ)
data = pd.read_csv('dengue2.csv')

# Verificar las columnas en el DataFrame
print(data.columns)

# Convertir 'Si'/'No' a 1/0 y 'sexo' a 0/1
yes_no_columns = ['fiebre', 'cefalea', 'dolrretroo', 'malgias', 'artralgia', 'erupcionr', 'dolor_abdo', 'vomito', 'diarrea', 'hipotensio', 'hepatomeg']
for col in yes_no_columns:
    if col in data.columns:
        data[col] = data[col].map({1: 1, 0: 0})
    else:
        print(f"La columna '{col}' no se encuentra en el DataFrame y se omitirá.")

if 'sexo' in data.columns:
    data['sexo'] = data['sexo'].map({'M': 0, 'F': 1})
else:
    raise KeyError("La columna 'sexo' no se encuentra en el DataFrame. Verifique el archivo CSV.")

# Codificar otras variables categóricas
label_columns = ['clasfinal', 'conducta', 'def_clas_edad']  # Incluir def_clas_edad
for col in label_columns:
    if col in data.columns:
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])
    else:
        print(f"La columna '{col}' no se encuentra en el DataFrame y se omitirá.")

# Dividir los datos en características (X) y etiquetas (y)
X = data.drop(['clasfinal', 'conducta'], axis=1)
y_clasfinal = data['clasfinal']
y_conducta = data['conducta']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train_clasfinal, X_test_clasfinal, y_train_clasfinal, y_test_clasfinal = train_test_split(X, y_clasfinal, test_size=0.2, random_state=42)
X_train_conducta, X_test_conducta, y_train_conducta, y_test_conducta = train_test_split(X, y_conducta, test_size=0.2, random_state=42)

# Entrenar el modelo de árbol de decisión para 'clasfinal'
tree_clasfinal = DecisionTreeClassifier(random_state=42)
tree_clasfinal.fit(X_train_clasfinal, y_train_clasfinal)

# Entrenar el modelo de árbol de decisión para 'conducta'
tree_conducta = DecisionTreeClassifier(random_state=42)
tree_conducta.fit(X_train_conducta, y_train_conducta)

# Crear el directorio 'static' si no existe
if not os.path.exists('static'):
    os.makedirs('static')

# Visualizar los árboles de decisión con Graphviz
def save_tree_graphviz(tree, feature_names, class_names, filename):
    dot_data = export_graphviz(tree, out_file=None, 
                               feature_names=feature_names, 
                               class_names=class_names, 
                               filled=True, rounded=True, 
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.format = 'png'
    graph.render(filename)

# Guardar los árboles de decisión como imágenes
save_tree_graphviz(tree_clasfinal, X.columns, label_encoders['clasfinal'].classes_, 'static/tree_clasfinal')
save_tree_graphviz(tree_conducta, X.columns, label_encoders['conducta'].classes_, 'static/tree_conducta')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_tree/<tree_type>')
def get_tree(tree_type):
    if tree_type == 'clasfinal':
        return jsonify({'tree_image': 'static/tree_clasfinal.png'})
    elif tree_type == 'conducta':
        return jsonify({'tree_image': 'static/tree_conducta.png'})
    else:
        return jsonify({'error': 'Invalid tree type'})

if __name__ == '__main__':
    socketio.run(app)
