Machine Learning Model API Wrapper
A robust, production-ready FastAPI service designed to wrap machine learning models and expose them via a REST API. Features fail-safe loading mechanisms ideal for containerized environments.

[!IMPORTANT] 🤖 Development Story

This project was built by my proprietary AI Development Orchestration System.

Engine: Gemini 2.5 Flash (free tier)

Result: Functional app in 1 shot

The system itself remains private for now, but this API wrapper proves what it can do.

Interested in the tech or potential collaboration? Reach out: tarocha1019@icloud.com

🚀 Features
Robust Startup: Implements a fail-safe model loading mechanism within the application startup event. Prevents container crash loops (e.g., in Docker/Kubernetes) even if the model file is missing or corrupted.

Auto-Generated Dummy Model: Automatically creates a dummy scikit-learn model if no model.pkl is found, allowing for immediate testing and deployment.

Strict Input Validation: Enforces input shapes and types using Pydantic and NumPy to ensure model stability.

Health Check Endpoint: Provides a /health endpoint that accurately reflects the model's loading status.

🛠️ Tech Stack
Python 3.10+

FastAPI (High-performance web framework)

Scikit-learn (Model inference)

NumPy (Data processing)

Uvicorn (ASGI server)

📦 Installation & Usage
1. Clone the repository
Bash

git clone https://github.com/EMMA019/ml-model-api-wrapper.git
cd ml-model-api-wrapper
2. Install dependencies
Bash

pip install -r requirements.txt
3. Run the server
Bash

python main.py
# OR
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
The server will start on port 8000. If model.pkl is missing, a dummy model will be generated automatically.

🔌 API Endpoints
Predict
POST /predict

Returns a prediction based on 5 numerical features.

Request Body:

JSON

{
  "features": [0.1, 0.5, 1.2, 0.3, 0.9]
}
Response:

JSON

{
  "prediction": 0.85
}
Health Check
GET /health

Checks if the service is running and the model is loaded.

JSON

{
  "status": "healthy",
  "model_loaded": true
}
📂 Project Structure
.
├── main.py           # FastAPI application & startup logic
├── model_wrapper.py  # Model loading & prediction logic
├── requirements.txt  # Dependencies
├── model.pkl         # Serialized model (auto-generated if missing)
└── README.md         # Project documentation

🤝 Contact
Created by [Tarocha]. Open to feedback and collaboration on AI orchestration systems.
📧 tarocha1019@icloud.com
