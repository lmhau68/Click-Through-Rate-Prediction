from pathlib import Path
from functools import lru_cache
from typing import Mapping

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "random_forest_ctr.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

FEATURE_COLUMNS = (
    "Daily Time Spent on Site",
    "Age",
    "Area Income",
    "Daily Internet Usage",
    "Gender",
)
DEFAULT_SCALER_COLUMNS = FEATURE_COLUMNS[:4]


@lru_cache(maxsize=1)
def _load_artifacts():
    missing_artifacts = [
        str(path) for path in (MODEL_PATH, SCALER_PATH) if not path.is_file()
    ]
    if missing_artifacts:
        raise FileNotFoundError(
            "Missing model artifact(s): "
            + ", ".join(missing_artifacts)
            + ". Obtain random_forest_ctr.pkl and scaler.pkl from the team."
        )

    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


def predict_ctr(user_data: Mapping[str, float | int]) -> tuple[int, float]:
    missing_features = [column for column in FEATURE_COLUMNS if column not in user_data]
    if missing_features:
        raise ValueError(f"Missing input feature(s): {', '.join(missing_features)}")

    df = pd.DataFrame(
        [{column: user_data[column] for column in FEATURE_COLUMNS}],
        columns=FEATURE_COLUMNS,
    )
    rf_model, scaler = _load_artifacts()

    # Scale toàn bộ 5 cột dữ liệu khớp với lúc training đúng theo hướng dẫn
    scaled_values = scaler.transform(df)

    # Dự đoán trực tiếp trên tập scaled_values (NumPy array)
    prediction = rf_model.predict(scaled_values)[0]
    classes = list(rf_model.classes_)
    if 1 not in classes:
        raise ValueError("The trained model does not contain the click class (1).")
    probability = rf_model.predict_proba(scaled_values)[0][classes.index(1)]

    return int(prediction), float(probability)