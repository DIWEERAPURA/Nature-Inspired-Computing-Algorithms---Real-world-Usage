import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ------------- Data Loading and Preprocessing -------------
# Here we use the MNIST dataset for demonstration.
# When you have your dataset, replace this section with your own data loading code.
(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()
x_train_full = x_train_full.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Create a validation split from the full training data
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
)


# ------------- CNN Model Creation Function -------------
def create_cnn_model(filters, filter_size, neurons, learning_rate):
    """
    Creates a CNN model with the given hyperparameters.

    Parameters:
      filters (int): Number of filters in the first Conv2D layer.
      filter_size (int): Size of the convolution kernel.
      neurons (int): Number of neurons in the dense layer.
      learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
      A compiled Keras model.
    """
    model = Sequential([
        # First convolutional block
        Conv2D(int(filters), (int(filter_size), int(filter_size)), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),

        # Second convolutional block (doubling the filters for a deeper representation)
        Conv2D(int(filters) * 2, (int(filter_size), int(filter_size)), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(int(neurons), activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ------------- Objective Function for PSO -------------
def objective_function(hyperparams):
    """
    Objective function that builds and trains a CNN with the given hyperparameters,
    and returns the negative validation accuracy. PSO will minimize this value.

    Parameters:
      hyperparams (array-like): [filters, filter_size, neurons, learning_rate]

    Returns:
      Negative validation accuracy.
    """
    # Unpack and round integer parameters appropriately
    filters, filter_size, neurons, learning_rate = hyperparams
    filters = int(np.round(filters))
    filter_size = int(np.round(filter_size))
    neurons = int(np.round(neurons))

    # Create the CNN model with the given hyperparameters
    model = create_cnn_model(filters, filter_size, neurons, learning_rate)

    # Train the model for a few epochs for faster evaluation (adjust epochs as needed)
    history = model.fit(x_train, y_train, epochs=3, batch_size=128, validation_data=(x_val, y_val), verbose=0)

    # Get the validation accuracy from the last epoch
    val_accuracy = history.history['val_accuracy'][-1]

    # Return negative accuracy (since PSO minimizes the objective)
    return -val_accuracy


# ------------- PSO Algorithm Implementation -------------
def pso_optimize(objective, bounds, num_particles=10, iterations=10):
    """
    Performs Particle Swarm Optimization to minimize the given objective function.

    Parameters:
      objective (callable): The objective function to minimize.
      bounds (np.array): Array of shape (num_dimensions, 2) defining min and max for each dimension.
      num_particles (int): Number of particles in the swarm.
      iterations (int): Number of iterations to run the optimization.

    Returns:
      (best_position, best_score): The best hyperparameters found and the corresponding objective value.
    """
    num_dimensions = bounds.shape[0]

    # Initialize particles' positions and velocities
    positions = np.zeros((num_particles, num_dimensions))
    velocities = np.zeros((num_particles, num_dimensions))
    for d in range(num_dimensions):
        positions[:, d] = np.random.uniform(bounds[d, 0], bounds[d, 1], size=num_particles)

    # Initialize personal best positions and scores
    pbest_positions = positions.copy()
    pbest_scores = np.array([objective(pos) for pos in positions])

    # Identify the global best position
    gbest_index = np.argmin(pbest_scores)
    gbest_position = pbest_positions[gbest_index].copy()
    gbest_score = pbest_scores[gbest_index]

    # PSO hyperparameters
    w = 0.7  # Inertia weight
    c1 = 1.5  # Cognitive (particle) coefficient
    c2 = 1.5  # Social (swarm) coefficient

    # Optimization loop
    for iter in range(iterations):
        for i in range(num_particles):
            # Update velocity
            r1 = np.random.rand(num_dimensions)
            r2 = np.random.rand(num_dimensions)
            cognitive = c1 * r1 * (pbest_positions[i] - positions[i])
            social = c2 * r2 * (gbest_position - positions[i])
            velocities[i] = w * velocities[i] + cognitive + social

            # Update position
            positions[i] = positions[i] + velocities[i]

            # Clip position to bounds
            for d in range(num_dimensions):
                positions[i, d] = np.clip(positions[i, d], bounds[d, 0], bounds[d, 1])

            # Evaluate new position
            score = objective(positions[i])

            # Update personal best if necessary
            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest_positions[i] = positions[i].copy()

        # Update global best
        gbest_index = np.argmin(pbest_scores)
        if pbest_scores[gbest_index] < gbest_score:
            gbest_score = pbest_scores[gbest_index]
            gbest_position = pbest_positions[gbest_index].copy()

        print(f"Iteration {iter + 1}/{iterations} - Best validation score: {-gbest_score:.4f}")

    return gbest_position, gbest_score


# ------------- Define Hyperparameter Bounds -------------
# Each row corresponds to: [min, max] for filters, filter_size, neurons, learning_rate
bounds = np.array([
    [16, 128],  # filters
    [3, 5],  # filter_size
    [128, 1024],  # neurons
    [0.0001, 0.01]  # learning_rate
])

# ------------- Run PSO to Optimize Hyperparameters -------------
best_params, best_obj_score = pso_optimize(objective_function, bounds, num_particles=10, iterations=10)

# Round and unpack best parameters for clarity
best_filters = int(np.round(best_params[0]))
best_filter_size = int(np.round(best_params[1]))
best_neurons = int(np.round(best_params[2]))
best_learning_rate = best_params[3]

print("\nOptimal Hyperparameters Found:")
print(f"Filters: {best_filters}")
print(f"Filter Size: {best_filter_size}")
print(f"Neurons: {best_neurons}")
print(f"Learning Rate: {best_learning_rate:.6f}")

# ------------- Train Final Model with Optimized Hyperparameters -------------
# For the final model, we use the full training data (you might combine x_train and x_val)
x_full_train = np.concatenate((x_train, x_val), axis=0)
y_full_train = np.concatenate((y_train, y_val), axis=0)

final_model = create_cnn_model(best_filters, best_filter_size, best_neurons, best_learning_rate)
final_history = final_model.fit(x_full_train, y_full_train, epochs=20, batch_size=128, validation_data=(x_test, y_test),
                                verbose=1)

# ------------- Evaluation -------------
# Evaluate the final model on the test set
test_loss, test_accuracy = final_model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Generate classification report and confusion matrix
y_pred = np.argmax(final_model.predict(x_test), axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()
plt.show()
