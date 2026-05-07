import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
    use_mlflow: bool = True
) -> float:
    """
    Huấn luyện mô hình và ghi nhận kết quả vào MLflow (tùy chọn).
    """

    # Đọc dữ liệu huấn luyện và đánh giá
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # Tách đặc trưng và nhãn
    X_train = df_train.drop("target", axis=1)
    y_train = df_train["target"]
    X_eval = df_eval.drop("target", axis=1)
    y_eval = df_eval["target"]

    # Khởi tạo mô hình
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    # Tính accuracy và f1_score
    preds = model.predict(X_eval)
    acc = accuracy_score(y_eval, preds)
    f1 = f1_score(y_eval, preds, average="weighted")

    # In kết quả ra màn hình
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

    # Ghi nhận kết quả vào MLflow nếu được yêu cầu
    if use_mlflow:
        # Thiết lập MLflow tracking URI nếu chưa được set trong env
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
        if not os.environ.get("MLFLOW_ARTIFACT_ROOT"):
            os.environ["MLFLOW_ARTIFACT_ROOT"] = "./mlartifacts"

        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(model, "model")

    # Lưu metrics và model ra file (luôn thực hiện để CI/CD có thể đọc)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/metrics.json", "w") as f:
        json.dump({"accuracy": acc, "f1_score": f1}, f)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
