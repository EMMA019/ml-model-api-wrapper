import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import List

# Load the model once when the module is imported (startup time)
# Create a dummy model and save it for demonstration purposes
# In a real application, you would have your trained model.pkl file.
try:
    # Attempt to load the model if it exists
    MODEL = joblib.load('model.pkl')
except FileNotFoundError:
    # If model.pkl does not exist, create a dummy one
    print("model.pkl not found. Creating a dummy model.")
    dummy_model = LogisticRegression()
    # Fit with some dummy data if necessary for the dummy model to have a structure
    # For a simple LogisticRegression, fitting is not strictly necessary for just loading,
    # but in more complex models it might be.
    # Here we just create it to have a placeholder.
    X_dummy = np.random.rand(10, 5)  # 10 samples, 5 features
    y_dummy = np.random.randint(0, 2, 10)
    dummy_model.fit(X_dummy, y_dummy)
    joblib.dump(dummy_model, 'model.pkl')
    MODEL = dummy_model
    print("Dummy model.pkl created.")


def validate_and_predict(data: List[float]) -> List[float]:
    """
    Validates input data, converts it to a NumPy array, and performs prediction.

    Args:
        data: A list of floats representing the input features.

    Returns:
        A list of floats representing the model's predictions.

    Raises:
        ValueError: If the input data is not in the expected format or shape.
    """
    if not isinstance(data, list):
        raise ValueError("Input data must be a list.")

    try:
        input_array = np.array(data, dtype=np.float32)
    except ValueError:
        raise ValueError("Input data must contain numerical values.")

    # Assuming the model expects a specific number of features.
    # Adjust 'expected_features' based on your actual model.
    expected_features = 5
    if input_array.ndim != 1 or input_array.shape[0] != expected_features:
        raise ValueError(
            f"Input data must be a 1D array with exactly {expected_features} features."
        )

    # Reshape for prediction if the model expects a 2D array (e.g., one sample)
    input_reshaped = input_array.reshape(1, -1)

    # Perform prediction
    # The output of predict_proba is typically a 2D array of shape (n_samples, n_classes)
    # and predict is typically a 1D array of shape (n_samples,)
    # We'll return probabilities for demonstration, as it's often more
    # informative.
    try:
        # If you need class labels, use MODEL.predict(input_reshaped)[0]
        # If you need probabilities, use MODEL.predict_proba(input_reshaped)[0]
        # For LogisticRegression, predict_proba returns probabilities for each class.
        # Let's return the probability of the positive class for simplicity.
        predictions = MODEL.predict_proba(input_reshaped)[0]
        # Return probability of the positive class
        return predictions.tolist()
    except Exception as e:
        # Catch potential errors during prediction (e.g., model not fitted
        # correctly)
        raise RuntimeError(f"Error during model prediction: {e}")
