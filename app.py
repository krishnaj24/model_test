import streamlit as st
import numpy as np
import keras

# Load trained model
model = keras.models.load_model("iris_model.h5")

# Iris class names
class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

st.title("🌸 Iris Flower Prediction (No sklearn)")

# User inputs
sepal_length = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width  = st.number_input("Sepal Width (cm)", 2.0, 5.0, 3.0)
petal_length = st.number_input("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width  = st.number_input("Petal Width (cm)", 0.1, 3.0, 1.2)

if st.button("Predict"):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(data, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    
    st.success(f"🌼 Predicted Species: **{class_names[predicted_class]}**")
    st.write(f"Confidence: {confidence:.2f}%")
