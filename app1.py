# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

app = Flask(__name__)
CORS(app)

# ------------------------
# 1. Chargement des datasets
# ------------------------
df_commercial = pd.read_csv('movie_dataset_supervised_cleaned.csv')
df_critique = pd.read_csv('movie_dataset_supervised_critique_cleaned.csv')

# ------------------------
# 2. Préparation KNN pour succès commercial
# ------------------------
# Imputation des colonnes numériques
num_cols_commercial = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
for col in df_commercial[num_cols_commercial].columns:
    df_commercial[col] = df_commercial[col].fillna(df_commercial[col].mean())

# Colonnes catégorielles
cat_cols_commercial = ['original_language', 'director']

# Préprocessing KNN avec encodage OneHot
preprocessor_commercial = ColumnTransformer(transformers=[
    ('num', SimpleImputer(strategy='mean'), num_cols_commercial),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols_commercial)
])

X_commercial = df_commercial[num_cols_commercial + cat_cols_commercial]
y_commercial = df_commercial['success']

knn_model = Pipeline([
    ('preprocessor', preprocessor_commercial),
    ('classifier', KNeighborsClassifier(n_neighbors=5))
])

knn_model.fit(X_commercial, y_commercial)

# ------------------------
# 3. Préparation Random Forest pour succès critique
# ------------------------
num_cols_critique = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
cat_cols_critique = ['original_language', 'director']

preprocessor_critique = ColumnTransformer(transformers=[
    ('num', SimpleImputer(strategy='mean'), num_cols_critique),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols_critique)
])

X_critique = df_critique[num_cols_critique + cat_cols_critique]
y_critique = df_critique['critique_success']

rf_model = Pipeline([
    ('preprocessor', preprocessor_critique),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

rf_model.fit(X_critique, y_critique)

# ------------------------
# 4. Routes API
# ------------------------

@app.route('/')
def home():
    return "API Movie Success Prediction is running."

# --- Langues disponibles ---
@app.route('/api/languages', methods=['GET'])
def get_languages():
    languages = df_critique['original_language'].dropna().unique().tolist()
    return jsonify({'languages': languages})

# --- Réalisateurs disponibles ---
@app.route('/api/directors', methods=['GET'])
def get_directors():
    directors = df_critique['director'].dropna().unique().tolist()
    return jsonify({'directors': directors})

# --- Prédiction succès commercial ---
@app.route('/api/predict/commercial', methods=['POST'])
def predict_commercial():
    data = request.json
    input_df = pd.DataFrame([data])
    pred = knn_model.predict(input_df)[0]
    return jsonify({'commercial_success': int(pred)})

# --- Prédiction succès critique ---
@app.route('/api/predict/critique', methods=['POST'])
def predict_critique():
    data = request.json
    input_df = pd.DataFrame([data])
    pred = rf_model.predict(input_df)[0]
    return jsonify({'critique_success': int(pred)})

# ------------------------
if __name__ == '__main__':
    app.run(debug=True)
