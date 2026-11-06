import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA_PATH = 'data/processed/ctr_cleaned.csv'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_ctr.pkl')
TARGET = 'Clicked on Ad'


def main():
    print("Đang đọc dữ liệu...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print("Thứ tự cột đặc trưng:", list(X.columns))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print("Đang huấn luyện Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print("\nBáo cáo chi tiết:")
    print(classification_report(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    print(f"\nĐã lưu mô hình tại: {MODEL_PATH}")


if __name__ == '__main__':
    main()