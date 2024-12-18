import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Load the Iris dataset (for displaying class names)
iris = load_iris()

# Load the trained model from the .pkl file
with open('models/iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize the standard scaler (use the same one used during training)
scaler = StandardScaler()

# Streamlit UI
st.title('Iris Flower Prediction')

# Description
st.write("""
This is a simple ML app that uses the Iris dataset. You can input the features of a flower, and the model will predict its species.
""")

# User inputs for Sepal and Petal Length and Width
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

# Prepare input for prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Standardize the input data using the same scaler used for training
input_data = scaler.fit_transform(input_data)

# Make prediction
prediction = model.predict(input_data)
prediction_label = iris.target_names[prediction][0]

# Show the prediction
st.write(f"Prediction: {prediction_label}")
