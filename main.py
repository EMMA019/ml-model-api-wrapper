import sys  # Import sys for potential graceful exit or error logging, though not used for exit here
from typing import List
import uvicorn
from pydantic import BaseModel, Field, validator
from fastapi import FastAPI, HTTPException
To fix `main.py` and make it more robust, especially in a Docker environment where startup failures can manifest as "Docker not started" (e.g., if the application crashes immediately upon import ), we should handle the import of `model_wrapper.py`'s components within a FastAPI `startup` event. This allows the application to catch potential `ImportError` or other exceptions that might occur during `model_wrapper.py`'s initial model loading, preventing an immediate crash of the entire application and allowing the FastAPI server to start, albeit in an "unhealthy" state if the model loading failed.

The `EXPECTED_FEATURES` issue(where it's not exported by `model_wrapper.py`) is retained with the local definition in `main.py`, as described in the comments.

Here's the corrected `main.py`:

    # Define EXPECTED_FEATURES locally, reflecting the value hardcoded in model_wrapper.py.
    # This temporary definition allows main.py to function while adhering to its internal logic.
    # Ideally, model_wrapper.py should export EXPECTED_FEATURES as the single
    # source of truth.
EXPECTED_FEATURES = 5

# These global variables will hold the imported model object and prediction function.
# They are initialized to None and populated during the FastAPI startup event.
_model_instance = None
_predict_function = None

# --- FastAPI App Initialization ---
app = FastAPI(
    title="ML Model API Wrapper",
    description="A FastAPI service for making predictions with a pre-trained ML model.",
    version="1.0.0",
)

# --- Startup Event for Model Loading and Predictor Function ---


@app.on_event("startup")
async def load_model_and_predictor():
    """
    Loads the ML model and prediction function from model_wrapper.py during application startup.
    This enhances robustness by catching potential errors during model loading,
    preventing an immediate application crash.
    """
    global _model_instance, _predict_function
    try:
        # Attempt to import MODEL and validate_and_predict from model_wrapper.py.
        # This action triggers model_wrapper.py's top-level code, including its model loading logic.
        # If model_wrapper.py crashes (e.g., due to file permissions for model.pkl or other issues),
        # an ImportError or other exception will be caught here.
        from model_wrapper import MODEL, validate_and_predict
        _model_instance = MODEL
        _predict_function = validate_and_predict

        if _model_instance is None:
            # This is a safeguard. If model_wrapper.py imported successfully but somehow
            # failed to set MODEL, this warning will be logged.
            print(
                "WARNING: model_wrapper.py imported successfully, but MODEL variable is None.")
            _predict_function = None  # Ensure predictor is also None if model is missing
        else:
            print(
                "Model and prediction function loaded successfully from model_wrapper.py.")

    except ImportError as ie:
        # Catch specific ImportError if model_wrapper.py fails during its initialization
        # (e.g., due to unhandled exceptions in its global scope).
        print(
            f"ERROR: Failed to import model_wrapper components. Is there an issue in model_wrapper.py? Detail: {ie}")
        _model_instance = None
        _predict_function = None
    except Exception as e:
        # Catch any other unexpected exceptions during the startup process.
        print(
            f"ERROR: An unexpected error occurred during model/predictor loading: {e}")
        _model_instance = None
        _predict_function = None

# --- Input Data Model ---


class PredictionInput(BaseModel):
    """
    Defines the expected structure and types for input data.
    Features are expected as a list of floats.
    """
    features: List[float] = Field(
        ...,
        description=f"A list of {EXPECTED_FEATURES} numerical features for prediction.")

    # Pydantic validator to ensure the list has the correct number of features.
    @validator('features')
    def check_features_length(cls, v):
        if len(v) != EXPECTED_FEATURES:
            raise ValueError(
                f"Expected {EXPECTED_FEATURES} features, but received {len(v)}.")
        return v

# --- Prediction Endpoint ---


@app.post("/predict")
async def predict(input_data: PredictionInput):
    """
    Receives input features, validates them, and returns the model's prediction.
    """
    global _model_instance, _predict_function

    # Check if the model and prediction function were loaded successfully at startup.
    # If not, return a 503 Service Unavailable error.
    if _model_instance is None or _predict_function is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available or failed to load during startup. Please check server logs for details.")

    try:
        # Pydantic has already validated the input_data.features list (type and length).
        # Now, pass the features to the model_wrapper's prediction function.
        # The wrapper handles conversion to numpy array and actual prediction,
        # adhering to input validation best practices.
        predictions = _predict_function(input_data.features)

        # Assuming a single prediction from a single input
        # The validate_and_predict function from model_wrapper returns a List[float],
        # so we extract the first element as the prediction result.
        return {"prediction": predictions[0]}

    except ValueError as ve:
        # Catch specific validation errors that might arise from within the
        # model_wrapper
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        # Catch any other unexpected errors during the prediction process.
        print(f"An error occurred during prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during prediction.")

# --- Health Check Endpoint ---


@app.get("/health")
async def health_check():
    """
    A simple endpoint to check if the API is running and the model is loaded.
    """
    global _model_instance
    # The health check now relies on the _model_instance populated at startup.
    if _model_instance is None:
        return {"status": "unhealthy", "model_loaded": False,
                "detail": "Model failed to load at startup."}
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    print("Starting FastAPI server...")
    # Uvicorn serves the FastAPI application.
    # The model loading now occurs within the FastAPI startup event,
    # before uvicorn.run fully starts the server for requests.
    uvicorn.run(app, host="0.0.0.0", port=8000)
