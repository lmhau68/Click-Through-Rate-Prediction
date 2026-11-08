# 🚀 Hệ Thống Dự Đoán Tỷ Lệ Nhấp Chuột (CTR Prediction)

Dự án học máy (Machine Learning) nhằm mục đích phân tích và dự đoán khả năng một người dùng sẽ nhấp vào quảng cáo (Click-Through Rate). Hệ thống không chỉ dừng lại ở bước huấn luyện mô hình (Model Training) mà còn được đóng gói hoàn chỉnh thành một Web API độc lập, đi kèm với giao diện người dùng trực quan.

## 🛠️ Công Nghệ Sử Dụng (Tech Stack)

*   **Trí tuệ nhân tạo (AI Core):** `Python`, `Scikit-Learn`, `Pandas`, `Seaborn`.
*   **Back-end & API:** `FastAPI`, `Uvicorn`, `Pydantic`.
*   **Front-end (Giao diện):** `HTML5`, `CSS3`, `JavaScript` (Vanilla, Fetch API).
*   **Đóng gói mô hình:** `joblib`.

---

## 📁 Cấu Trúc Thư Mục (Project Structure)

Dự án được tổ chức theo chuẩn kiến trúc phần mềm, phân tách rõ ràng giữa môi trường nghiên cứu (Jupyter Notebook) và môi trường triển khai thực tế (Production).

```text
Click_Through_Rate_Prediction/
│
├── api/                     # Chứa mã nguồn khởi chạy máy chủ API
│   └── app.py               # File chính của FastAPI, định nghĩa Endpoints
│
├── data/                    # Thư mục chứa tập dữ liệu
│   └── raw/                 # Dữ liệu nguyên bản (ad_10000records.csv)
│
├── frontend/                # Thư mục chứa Giao diện người dùng
│   └── index.html           # Trang web tĩnh dùng để kiểm thử mô hình
│
├── models/                  # Nơi lưu trữ các tệp trọng số mô hình đã huấn luyện
│   ├── random_forest_ctr.pkl
│   └── scaler.pkl           
│
├── src/                     # Mã nguồn lõi (AI Core)
│   └── models/
│       ├── predict_model.py # Xử lý dữ liệu chuẩn hóa và gọi mô hình suy luận
│       └── train_model.py   # Script huấn luyện tự động
│
├── 1.0-data-exploration.ipynb # Notebook Phân tích khám phá dữ liệu (EDA)
├── 2.0-model-training.ipynb   # Notebook Thử nghiệm và đánh giá các mô hình
├── README.md                # Tài liệu hướng dẫn dự án
└── requirements.txt         # Danh sách các thư viện cần thiết
```

---

## 📊 Kết Quả Huấn Luyện (Model Performance)

Sau quá trình khám phá dữ liệu (EDA) và thực nghiệm đánh giá đa chiều với 3 thuật toán phân lớp phổ biến (Logistic Regression, Naive Bayes, Random Forest), hệ thống đã thống nhất lựa chọn **Random Forest** làm động cơ cốt lõi.

**Đánh giá trên tập kiểm thử độc lập (Test Set):**
*   🏆 **Thuật toán tốt nhất:** Random Forest (Rừng ngẫu nhiên)
*   🎯 **Độ chính xác (Accuracy):** `78.3%`
*   📈 **F1-Score:** `0.78`
*   **Ưu điểm:** Khả năng xử lý các tương tác đặc trưng phi tuyến tính phức tạp, kìm hãm hiệu quả hiện tượng quá khớp (Overfitting) thông qua cơ chế học tập kết hợp (Ensemble Learning), và duy trì tỷ lệ Lỗi loại I, Lỗi loại II ở mức thấp nhất trong số các mô hình được thử nghiệm.

## ⚙️ Hướng Dẫn Cài Đặt & Khởi Chạy (Installation & Running)

Vì các tệp trọng số mô hình (Model Artifacts) không được lưu trữ trực tiếp trên mã nguồn GitHub, bạn cần thực hiện tuần tự các bước dưới đây để tải dữ liệu, tự huấn luyện lại lõi AI và khởi chạy hệ thống trên máy cá nhân (Localhost). Yêu cầu: Python 3.9+.

### Bước 1: Sao chép dự án và Tạo môi trường ảo

Mở Terminal / Command Prompt và chạy các lệnh sau để tải mã nguồn và tạo môi trường ảo (giúp cô lập thư viện):

```bash
# Clone dự án về máy
git clone https://github.com/lmhau68/Click-Through-Rate-Prediction
cd Click_Through_Rate_Prediction

# Khởi tạo môi trường ảo có tên là venv
python -m venv venv

# Kích hoạt môi trường ảo
# Đối với Windows:
.\venv\Scripts\activate
# Đối với macOS/Linux:
source venv/bin/activate
```

### Bước 2: Cài đặt thư viện phụ thuộc

Khi môi trường ảo đã được kích hoạt (có chữ `(venv)` ở đầu dòng lệnh), tiến hành cài đặt các gói thư viện cần thiết:

```bash
pip install -r requirements.txt
```

### Bước 3: Chuẩn bị dữ liệu gốc

1. Truy cập Kaggle và tải tệp Dataset gốc của bài toán Click-Through Rate.
2. Đổi tên tệp (nếu cần) thành `click_through_rate.csv` (hoặc tên tương ứng trong code).
3. Đặt tệp dữ liệu này vào thư mục `data/raw/` trên máy tính của bạn.

### Bước 4: Tiền xử lý dữ liệu (Data Preprocessing)

Thực hiện bước này để làm sạch dữ liệu và xuất ra tệp dữ liệu chuẩn bị cho huấn luyện. 
Nếu dùng VS Code, bạn làm theo hướng dẫn sau:
1. Đảm bảo đã cài đặt 2 extension: **Python** và **Jupyter** của Microsoft.
2. Mở file `1.0-data-exploration.ipynb` trong thư mục `notebooks/Exploratory/`.
3. Nhìn lên góc trên cùng bên phải, nhấp vào **Select Kernel** -> **Python Environments** -> Chọn đúng môi trường ảo `venv` (`.\venv\Scripts\python.exe`).
4. (Nếu VS Code yêu cầu cài đặt gói `ipykernel` ở góc dưới bên phải, hãy bấm **Install**).
5. Bấm nút **Run All** ở thanh công cụ trên cùng để chạy toàn bộ file. Kết quả sẽ tạo ra một tệp dữ liệu sạch (VD: `ctr_cleaned.csv`).

### Bước 5: Huấn luyện lõi AI (Model Training)

Mở lại Terminal (đảm bảo vẫn đang ở môi trường `(venv)`), chạy lệnh sau để tiến hành huấn luyện AI:

```bash
python src/models/train_model.py
```

* **Kiểm tra:** Terminal sẽ in ra thông báo độ chính xác (Accuracy, F1-Score). Hãy mở thư mục `models/` ra kiểm tra, nếu thấy xuất hiện 2 tệp `random_forest_ctr.pkl` và `scaler.pkl` là bạn đã sở hữu trọn bộ lõi AI thành công!

### Bước 6: Khởi chạy Máy chủ API (Backend)

Lúc này, hệ thống đã sẵn sàng. Chạy lệnh sau trên Terminal để khởi động máy chủ FastAPI:

```bash
uvicorn api.app:app --reload
```

* **Kiểm tra:** Nếu Terminal hiện `Uvicorn running on http://127.0.0.1:8000`, server đã chạy thành công. Hãy giữ nguyên Terminal này.

### Bước 7: Khởi chạy Giao diện người dùng (Frontend)

Mở thêm một cửa sổ **Terminal 2** mới (vẫn kích hoạt `venv`), chạy lệnh thiết lập máy chủ web tĩnh:

```bash
python -m http.server 5500 --bind 127.0.0.1 --directory frontend
```

* **Sử dụng:** Giữ phím `Ctrl` và click vào đường link `http://127.0.0.1:5500/` trên Terminal để mở trang Giao diện dự đoán trên trình duyệt web của bạn và bắt đầu trải nghiệm.

---

## 🔌 Hướng Dẫn Sử Dụng API (API Usage)

Hệ thống cung cấp một Endpoint RESTful API, cho phép các nền tảng khác (Web, Mobile App, CRM) gọi mô hình Học máy để lấy dự đoán mà không cần thông qua giao diện frontend.

* **Endpoint:** `/api/predict`
* **Method:** `POST`
* **Content-Type:** `application/json`

### 1. Cấu trúc Dữ liệu Đầu vào (Request Body)

Dữ liệu gửi lên phải tuân thủ nghiêm ngặt định dạng JSON với 5 trường thông số:

```json
{
  "time_spent": 65,     
  "age": 35,               
  "income": 60000,       
  "internet_usage": 120,
  "gender": 0              
}
```
*(Ghi chú: `time_spent` giới hạn 0-1440, `age` giới hạn 1-120, `gender` quy ước 0: Nữ, 1: Nam).*

### 2. Ví dụ Gọi API bằng cURL

Bạn có thể test trực tiếp API từ Terminal:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/api/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "time_spent": 100,
  "age": 20,
  "income": 70000,
  "internet_usage": 5000,
  "gender": 0
}'
```

### 3. Cấu trúc Kết quả Trả về (Response)

**Trường hợp 1: Thành công (HTTP 200 OK)**
Mô hình chạy thành công sẽ trả về dự đoán phân lớp và tỷ lệ xác suất:

```json
{
  "status": "success",
  "click_prediction": 0,    // 0: Không click, 1: Có click quảng cáo
  "probability": 23.45      // Tỷ lệ xác suất (tính theo %)
}
```

**Trường hợp 2: Lỗi Dữ liệu (HTTP 422 Unprocessable Entity)**
Nếu truyền dữ liệu sai định dạng/vượt quá biên vật lý, hệ thống Pydantic sẽ chặn lại và báo lỗi:

```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": ["body", "time_spent"],
      "msg": "Input should be less than or equal to 1440",
      "input": 10000.0
    }
  ]
}
```