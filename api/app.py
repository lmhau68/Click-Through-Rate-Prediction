from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
from typing import Literal

# Trở đường dẫn để lấy hàm predict_ctr từ src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.predict_model import predict_ctr

app = FastAPI(title="Hệ thống dự đoán CTR")

# Cho phép Web (Frontend) gọi API mà không bị chặn lỗi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Định nghĩa cấu trúc JSON nhận từ Web khớp 5 đặc trưng
class UserData(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    time_spent: float = Field(ge=0)
    age: int = Field(gt=0)
    income: float = Field(ge=0)
    internet_usage: float = Field(ge=0)
    gender: Literal[0, 1]


@app.post("/api/predict")
def get_prediction(data: UserData):
    # Khớp tên biến với đúng tên cột lúc AI học
    input_dict = {
        'Daily Time Spent on Site': data.time_spent,
        'Age': data.age,
        'Area Income': data.income,
        'Daily Internet Usage': data.internet_usage,
        'Gender': data.gender
    }
    
    try:
        pred, prob = predict_ctr(input_dict)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    
    # Trả kết quả về cho Web
    return {
        "status": "success",
        "click_prediction": pred, # 1 là Có click, 0 là Không click
        "probability": round(prob * 100, 2) # Tính theo %
    }