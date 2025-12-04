#Succès commercial K plus proche voisins

# Maintenant on va appliquer K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import pandas as pd

# --- 1. Charger le dataset supervisé ---
df_supervised = pd.read_csv('movie_dataset_supervised_cleaned.csv')

# --- 2. Imputer les valeurs manquantes ---
imputer = SimpleImputer(strategy='mean')
for col in df_supervised.select_dtypes(include=['float64', 'int64']).columns:
    df_supervised[col] = imputer.fit_transform(df_supervised[[col]])

# --- 3. Encodage des variables catégorielles ---
df_supervised_encoded = pd.get_dummies(df_supervised, drop_first=True)

# --- 4. Séparation variables explicatives / variable cible ---
X = df_supervised_encoded.drop('success', axis=1)
y = df_supervised_encoded['success']

# --- 5. Split des données ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# --- 6. Création du modèle ---
model = KNeighborsClassifier(n_neighbors=5)

# --- 7. Entraînement ---
model.fit(X_train, y_train)

# --- 8. Prédiction ---
y_pred = model.predict(X_test)

# --- 9. Évaluation du modèle ---
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Nombre de correctements classifiés
correct_predictions = (y_test == y_pred).sum()
total_predictions = y_test.shape[0]
print(f"Nombre de prédictions correctes: {correct_predictions} sur {total_predictions}")

# Score de précision
accuracy = correct_predictions / total_predictions
print(f"Précision du modèle: {accuracy:.2f}")





#Succès critique avec Foret aleatoire
# --- 1. Imports ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd

# --- 2. Chargement du dataset ---
df = pd.read_csv('movie_dataset_supervised_critique_cleaned.csv')

# --- 3. Features / Target ---
X = df[['budget', 'original_language', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count', 'director']]
y = df['critique_success']

# --- 4. Colonnes numériques / catégoriques ---
categorical_cols = ['original_language', 'director']
numeric_cols = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']

# --- 5. Préprocessing ---

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# --- 6. Pipeline avec Forêt Aléatoire ---
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,       # nombre d'arbres
        max_depth=None,        # profondeur illimitée
        random_state=42,
        n_jobs=-1              # utilise tous les cœurs CPU
    ))
])

# --- 7. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# --- 8. Entraînement ---
clf.fit(X_train, y_train)

# --- 9. Prédictions ---
y_pred = clf.predict(X_test)

# --- 10. Évaluation ---
print("Random Forest Classifier Results:")
print(f"Accuracy: {clf.score(X_test, y_test):.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Score de précision
correct_predictions = (y_test == y_pred).sum()
accuracy = correct_predictions / len(y_test)
print(f"\nModel Accuracy: {accuracy:.2f}")
