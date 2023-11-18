import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))

# Loss function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Forward propagation
def forward_propagation(X, parameters):
    A = X
    caches = []

    for i in range(len(parameters) // 2):
        W = parameters[f'W{i + 1}']
        b = parameters[f'b{i + 1}']
        Z = np.dot(W, A) + b
        A = sigmoid(Z)

        cache = (A, W, b, Z)
        caches.append(cache)

    return A, caches

# Backward propagation
def backward_propagation(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    dAL = 2 * (AL - Y)

    for i in reversed(range(L)):
        A_prev, W, b, Z = caches[i]

        dZ = dAL * sigmoid(Z, derivative=True)
        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dAL = np.dot(W.T, dZ)

        grads[f'dW{i + 1}'] = dW
        grads[f'db{i + 1}'] = db

    return grads

# Update parameters
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for i in range(L):
        parameters[f'W{i + 1}'] -= learning_rate * grads[f'dW{i + 1}']
        parameters[f'b{i + 1}'] -= learning_rate * grads[f'db{i + 1}']

    return parameters

# Train neural network
def train_neural_network(X, Y, layers, learning_rate, num_iterations):
    np.random.seed(42)
    parameters = initialize_parameters(layers)

    for i in range(num_iterations):
        AL, caches = forward_propagation(X, parameters)
        cost = mean_squared_error(AL, Y)
        grads = backward_propagation(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
            st.write(f"Cost after iteration {i}: {cost}")

    return parameters

# Function to make predictions
def predict(X, parameters):
    AL, _ = forward_propagation(X, parameters)
    return AL

# Initialize parameters
def initialize_parameters(layers):
    parameters = {}
    L = len(layers)

    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layers[l], layers[l - 1]) * 0.01
        parameters[f'b{l}'] = np.zeros((layers[l], 1))

    return parameters

# Generate random time-series data for three days
def generate_random_data():
    np.random.seed(42)
    timestamps = pd.date_range(start="2023-01-01", end="2023-01-03 23:59:00", freq='T')
    values = np.random.normal(loc=0, scale=1, size=len(timestamps))
    data = pd.DataFrame({'timestamp': timestamps, 'value': values})
    return data

# Streamlit app
def main():
    st.title("Neural Network Asset Health Prediction App")

    # Generate random data for three days
    data = generate_random_data()

    # Sidebar - Neural Network Configuration
    st.sidebar.header("Neural Network Configuration")
    num_layers = st.sidebar.slider("Number of Layers", min_value=2, max_value=5, value=3)
    layer_sizes = [st.sidebar.slider(f"Layer {i} Size", min_value=1, max_value=100, value=10) for i in range(1, num_layers + 1)]
    learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
    num_iterations = st.sidebar.slider("Number of Iterations", min_value=100, max_value=5000, value=1000, step=100)
    threshold = st.sidebar.slider("Threshold for Health Prediction", min_value=0.1, max_value=0.9, value=0.5, step=0.1)

    # Prepare data
    X = data['value'].values.reshape(1, -1)
    Y = np.array([[1]])  # Placeholder for future health label, 1 for healthy, 0 for unhealthy

    # Train neural network
    parameters = train_neural_network(X, Y, layer_sizes, learning_rate, num_iterations)

    # Make predictions for the original data
    predictions_original = predict(X, parameters)

    # Label data using threshold
    data['health_label'] = np.where(predictions_original > threshold, 'Healthy', 'Unhealthy')

    # Plot original data with health labels
    st.subheader("Original Data Plot with Health Labels")
    fig = px.line(data, x='timestamp', y='value', color='health_label', labels={'value': 'Original Data'})
    st.plotly_chart(fig)

    # Generate random data for the fourth day
    data_future = generate_random_data()

    # Make predictions for the future data
    X_future = data_future['value'].values.reshape(1, -1)
    predictions_future = predict(X_future, parameters)

    # Label future data using threshold
    data_future['health_label'] = np.where(predictions_future > threshold, 'Healthy', 'Unhealthy')

    # Plot forecasted data with health labels
    st.subheader("Forecasted Data Plot with Health Labels")
    fig_future = px.line(data_future, x='timestamp', y='value', color='health_label', labels={'value': 'Forecasted Data'})
    st.plotly_chart(fig_future)

if __name__ == "__main__":
    main()
