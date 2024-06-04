import joblib
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('Classification.keras')
def get_input():
    sepal_length = float(input("Enter sepal length: "))
    sepal_width = float(input("Enter sepal width: "))
    petal_length = float(input("Enter petal length: "))
    petal_width = float(input("Enter petal width: "))

    return np.array([[sepal_length, sepal_width, petal_length, petal_width]])

def prediction(model, input_data):
    Prediction = model.predict(input_data)
    return Prediction

if __name__ == "__main__":
    
    # Get custom input
    input_data = get_input()
    
    # Make predictions
    logits = model.predict(input_data)
    
    # Apply softmax to get probabilities
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1)
    
    # Get the index with the highest probability
    predicted_class_index = np.argmax(probabilities)
    
    print(f"The predicted class is: {predicted_class_index}")