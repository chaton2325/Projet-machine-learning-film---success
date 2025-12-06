from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# INITIALISATION FLASK + CORS
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# CHARGEMENT DES DATASETS (ENCODÉS)
# -----------------------------
df_crit = pd.read_csv("movie_dataset_critical_success.csv")
df_com = pd.read_csv("movie_dataset_commercial_success.csv")

# -----------------------------
# ENCODAGE IDENTIQUE À TES MODÈLES
# -----------------------------

def encode_dataset(df, target_col):
    df_model = df.copy()
    categorical_cols = [c for c in ['genres', 'director', 'original_language'] if c in df_model.columns]
    df_model = pd.get_dummies(df_model, columns=categorical_cols)

    # convertir bool → int
    for col in df_model.columns:
        if df_model[col].dtype == "bool":
            df_model[col] = df_model[col].astype(int)

    X = df_model.drop(columns=[target_col])
    y = df_model[target_col]

    return df_model, X, y


# Succès critique
df_crit_model, X_crit, y_crit = encode_dataset(df_crit, "succes_critique")
crit_features = X_crit.columns

# Succès commercial
df_com_model, X_com, y_com = encode_dataset(df_com, "succes_commercial")
com_features = X_com.columns

# -----------------------------
# ENTRAÎNEMENT DES MODÈLES
# -----------------------------

# Succès critique → modèle retenu
model_critique = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    max_depth=10
)
model_critique.fit(X_crit, y_crit)

# Succès commercial → modèle retenu
model_commercial = RandomForestClassifier(
    n_estimators=8,
    random_state=42,
    max_features=None,
    class_weight='balanced'
)
model_commercial.fit(X_com, y_com)

# -----------------------------
# FONCTION UTILITAIRES
# -----------------------------

def prepare_input(user_input, feature_list):
    """Crée un input propre : ajoute colonnes manquantes + met l'ordre exact"""
    df = pd.DataFrame([user_input])

    # Toute colonne absente = 0
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0

    # On garde SEULEMENT les colonnes utiles, dans l'ordre du modèle
    df = df[feature_list]

    return df


# -----------------------------
# ENDPOINTS API
# -----------------------------

@app.route("/predict/critique", methods=["POST"])
def predict_critique():
    data = request.json
    X = prepare_input(data, crit_features)
    pred = model_critique.predict(X)[0]
    return jsonify({"succes_critique": int(pred)})


@app.route("/predict/commercial", methods=["POST"])
def predict_commercial():
    data = request.json
    X = prepare_input(data, com_features)
    pred = model_commercial.predict(X)[0]
    return jsonify({"succes_commercial": int(pred)})


# -----------------------------
# DONNER LISTES POUR LE FRONTEND
# -----------------------------

@app.route("/list/directors", methods=["GET"])
def list_directors():
    directors = sorted(df_crit["director"].dropna().unique().tolist())
    return jsonify(directors)

@app.route("/list/languages", methods=["GET"])
def list_languages():
    langs = sorted(df_crit["original_language"].dropna().unique().tolist())
    return jsonify(langs)

@app.route("/list/genres", methods=["GET"])
def list_genres():
    genres_raw = df_crit["genres"].fillna("").unique()
    genre_set = set()
    for g in genres_raw:
        for part in g.split(" "):
            genre_set.add(part.strip())
    return jsonify(sorted(list(genre_set)))

# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
