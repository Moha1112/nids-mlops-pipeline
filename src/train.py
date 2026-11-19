import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

# 1. Configuration des chemins avec pathlib
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "cicids2017_cleaned.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "xgboost_nids_model.json"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
MLFLOW_DB_URI = f"sqlite:///{BASE_DIR / 'mlflow.db'}"

def train_pipeline(sample_size: int, n_estimators: int, max_depth: int, learning_rate: float):
    print(f"Chargement des donnees depuis : {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Nettoyage de securite (valeurs infinies ou nulles)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Echantillonnage stratifie pour menager le CPU et la RAM en local
    if sample_size and sample_size < len(df):
        print(f"Echantillonnage de {sample_size} lignes sur {len(df)}...")
        df, _ = train_test_split(
            df, train_size=sample_size, stratify=df["Attack Type"], random_state=42
        )

    X = df.drop("Attack Type", axis=1)
    y = df["Attack Type"]

    # 2. Encodage des labels (aligne sur ta version locale de scikit-learn !)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    # 3. Configuration de MLflow (base SQLite locale legere)
    mlflow.set_tracking_uri(MLFLOW_DB_URI)
    mlflow.set_experiment("NIDS_XGBoost_Detection")

    with mlflow.start_run(run_name=f"xgb_depth{max_depth}_est{n_estimators}"):
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "tree_method": "hist",
            "device": "cpu",
            "random_state": 42,
            "n_jobs": -1
        }
        
        # Log des hyperparametres dans MLflow
        mlflow.log_params(params)
        mlflow.log_param("sample_size", len(df))
        mlflow.log_param("num_features", X.shape[1])

        print("Entrainement du modele XGBoost en cours...")
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        # 4. Evaluation et log des metriques
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)

        print(f"Resultats -> Accuracy: {acc:.4f} | F1 Weighted: {f1_weighted:.4f} | F1 Macro: {f1_macro:.4f}")

        # 5. Sauvegarde des artefacts pour l'API et dans MLflow
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save_model(str(MODEL_PATH))
        joblib.dump(encoder, ENCODER_PATH)

        mlflow.log_artifact(str(MODEL_PATH), artifact_path="models")
        mlflow.log_artifact(str(ENCODER_PATH), artifact_path="models")
        print("Modele et encodeur sauvegardes et indexes dans MLflow avec succes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrainement NIDS avec suivi MLflow")
    parser.add_argument("--sample-size", type=int, default=100000, help="Nombre de lignes (0 pour tout le dataset)")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    train_pipeline(args.sample_size, args.n_estimators, args.max_depth, args.lr)