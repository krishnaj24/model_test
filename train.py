import numpy as np
import keras
from keras import layers

# Iris dataset (manually defined: 150 samples, 4 features)
# Sepal length, Sepal width, Petal length, Petal width
X = np.array([
    [5.1,3.5,1.4,0.2],[4.9,3.0,1.4,0.2],[4.7,3.2,1.3,0.2],[4.6,3.1,1.5,0.2],[5.0,3.6,1.4,0.2],
    [5.4,3.9,1.7,0.4],[4.6,3.4,1.4,0.3],[5.0,3.4,1.5,0.2],[4.4,2.9,1.4,0.2],[4.9,3.1,1.5,0.1],
    [5.4,3.7,1.5,0.2],[4.8,3.4,1.6,0.2],[4.8,3.0,1.4,0.1],[4.3,3.0,1.1,0.1],[5.8,4.0,1.2,0.2],
    [5.7,4.4,1.5,0.4],[5.4,3.9,1.3,0.4],[5.1,3.5,1.4,0.3],[5.7,3.8,1.7,0.3],[5.1,3.8,1.5,0.3],
    # ... (you can paste the rest of the 150 iris samples here for full dataset)
])

# Labels (0=setosa, 1=versicolor, 2=virginica)
y = np.array([0]*50 + [1]*50 + [2]*50)

# One-hot encode labels
y_encoded = np.zeros((y.size, 3))
y_encoded[np.arange(y.size), y] = 1

# Shuffle dataset
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y_encoded = y_encoded[indices]

# Train/test split (80/20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_encoded[:split], y_encoded[split:]

# Build model
model = keras.Sequential([
    layers.Dense(10, activation="relu", input_shape=(4,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(3, activation="softmax")
])

# Compile
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {acc:.2f}")

# Save model
model.save("iris_model.h5")
print("✅ Model saved as iris_model.h5")
