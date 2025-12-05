from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
CORS(app)  # Allow all origins

# -------------------------------
# CONFIGURATION MODIFIABLE
# -------------------------------
N_ESTIMATORS_CRITIQUE = 200
N_ESTIMATORS_COMMERCIAL = 200
DEFAULT_K_CV_COMMERCIAL = 44
# -------------------------------

# --- Charger les datasets ---
df_critique = pd.read_csv("movie_dataset_supervised_critique_cleaned.csv")
df_commercial = pd.read_csv("movie_dataset_supervised_cleaned.csv")

# --- Colonnes à utiliser pour chaque modèle ---
CRITIQUE_FEATURES = ['budget', 'original_language', 'popularity', 'revenue', 'runtime', 'vote_count', 'director']
COMMERCIAL_FEATURES = ['budget', 'original_language', 'popularity', 'runtime', 'vote_average', 'vote_count', 'director']

def prepare_pipeline(df, features, target_column, n_estimators=200):
    X = df[features]
    y = df[target_column]

    numeric_cols = X.select_dtypes(include=['int64','float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1))
    ])

    return X_train, X_test, y_train, y_test, clf

# --- Préparer pipelines ---
# Critique (sans cross-validation)
X_train_critique, X_test_critique, y_train_critique, y_test_critique, clf_critique = prepare_pipeline(
    df_critique, CRITIQUE_FEATURES, target_column='critique_success', n_estimators=N_ESTIMATORS_CRITIQUE
)
clf_critique.fit(X_train_critique, y_train_critique)

# Commercial (avec cross-validation possible)
X_train_commercial, X_test_commercial, y_train_commercial, y_test_commercial, clf_commercial = prepare_pipeline(
    df_commercial, COMMERCIAL_FEATURES, target_column='success', n_estimators=N_ESTIMATORS_COMMERCIAL
)

# --- Endpoints ---
@app.route('/predict/<model_type>', methods=['POST'])
def predict(model_type):
    data = request.json
    df_input = pd.DataFrame([data])
    
    if model_type == 'critique':
        features = CRITIQUE_FEATURES
        clf = clf_critique
    else:
        features = COMMERCIAL_FEATURES
        clf = clf_commercial

    df_input = df_input[features]
    prediction = clf.predict(df_input)[0]
    return jsonify({"model_type": model_type, "prediction": int(prediction)})

@app.route('/cross_validation/commercial', methods=['GET'])
def cross_validation_commercial():
    try:
        k = int(request.args.get('k', DEFAULT_K_CV_COMMERCIAL))
        scores = cross_val_score(clf_commercial, X_train_commercial, y_train_commercial, cv=k, scoring='accuracy')
        return jsonify({
            "model_type": "commercial",
            "k": k,
            "mean_accuracy": scores.mean(),
            "std_accuracy": scores.std()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/languages/<model_type>', methods=['GET'])
def get_languages(model_type):
    df = df_critique if model_type == 'critique' else df_commercial
    if 'original_language' in df.columns:
        langs = df['original_language'].dropna().unique().tolist()
        return jsonify({"languages": langs})
    return jsonify({"error": "Colonne 'original_language' manquante"}), 400

@app.route('/directors/<model_type>', methods=['GET'])
def get_directors(model_type):
    df = df_critique if model_type == 'critique' else df_commercial
    if 'director' in df.columns:
        directors = df['director'].dropna().unique().tolist()
        return jsonify({"directors": directors})
    return jsonify({"error": "Colonne 'director' manquante"}), 400

if __name__ == '__main__':
    app.run(debug=True)
