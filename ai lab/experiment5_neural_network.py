"""
Experiment 5: Neural Network for Classification Using Backpropagation
Implements a feed-forward neural network with backpropagation for classification.
"""

import numpy as np
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', output_activation='sigmoid', verbose=True):
        """
        Initialize neural network
        input_size: number of input features
        hidden_sizes: list of hidden layer sizes
        output_size: number of output neurons
        activation: activation for hidden layers ('relu', 'sigmoid', 'tanh')
        output_activation: activation for output layer ('sigmoid', 'softmax')
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.output_activation = output_activation
        self.verbose = verbose

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        # Input to first hidden layer
        self.weights.append(np.random.randn(input_size, hidden_sizes[0]) * 0.01)
        self.biases.append(np.zeros((1, hidden_sizes[0])))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.weights.append(np.random.randn(hidden_sizes[i], hidden_sizes[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, hidden_sizes[i + 1])))

        # Last hidden to output layer
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * 0.01)
        self.biases.append(np.zeros((1, output_size)))

        # Store activations and gradients for backpropagation
        self.activations = []
        self.z_values = []
        self.d_weights = [np.zeros_like(w) for w in self.weights]
        self.d_biases = [np.zeros_like(b) for b in self.biases]

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def tanh(self, z):
        return np.tanh(z)

    def tanh_derivative(self, z):
        return 1 - np.tanh(z) ** 2

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward_propagation(self, X):
        """Forward pass through the network"""
        self.activations = [X]
        self.z_values = []

        a = X

        # Hidden layers
        for i in range(len(self.hidden_sizes)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.z_values.append(z)

            if self.activation == 'relu':
                a = self.relu(z)
            elif self.activation == 'sigmoid':
                a = self.sigmoid(z)
            elif self.activation == 'tanh':
                a = self.tanh(z)

            self.activations.append(a)

        # Output layer
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)

        if self.output_activation == 'sigmoid':
            a = self.sigmoid(z)
        elif self.output_activation == 'softmax':
            a = self.softmax(z)

        self.activations.append(a)
        return a

    def compute_loss(self, y_true, y_pred):
        """Compute loss (binary or categorical cross-entropy)"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        if self.output_activation == 'sigmoid':
            # Binary cross-entropy
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            # Categorical cross-entropy
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def backward_propagation(self, X, y, y_pred):
        """Backward pass to compute gradients"""
        m = X.shape[0]

        # Output layer gradient
        if self.output_activation == 'sigmoid':
            dz = y_pred - y
        else:  # softmax
            dz = y_pred - y

        # Initialize gradients
        self.d_weights = [np.zeros_like(w) for w in self.weights]
        self.d_biases = [np.zeros_like(b) for b in self.biases]

        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            # Gradient for weights and biases
            self.d_weights[i] = (1 / m) * np.dot(self.activations[i].T, dz)
            self.d_biases[i] = (1 / m) * np.sum(dz, axis=0, keepdims=True)

            if i > 0:
                # Gradient for next layer
                if self.activation == 'relu':
                    dz = np.dot(dz, self.weights[i].T) * self.relu_derivative(self.z_values[i - 1])
                elif self.activation == 'sigmoid':
                    dz = np.dot(dz, self.weights[i].T) * self.sigmoid_derivative(self.z_values[i - 1])
                elif self.activation == 'tanh':
                    dz = np.dot(dz, self.weights[i].T) * self.tanh_derivative(self.z_values[i - 1])
            else:
                # Input layer - no activation derivative needed
                pass

    def update_parameters(self, learning_rate):
        """Update weights and biases using gradient descent"""
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.d_weights[i]
            self.biases[i] -= learning_rate * self.d_biases[i]

    def fit(self, X, y, epochs=1000, learning_rate=0.01, batch_size=32, X_val=None, y_val=None):
        """Train the neural network"""
        loss_history = []
        val_loss_history = []

        for epoch in range(epochs):
            # Mini-batch gradient descent
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Forward propagation
                y_pred = self.forward_propagation(X_batch)

                # Backward propagation
                self.backward_propagation(X_batch, y_batch, y_pred)

                # Update parameters
                self.update_parameters(learning_rate)

            # Compute training loss
            y_pred_train = self.forward_propagation(X)
            loss = self.compute_loss(y, y_pred_train)
            loss_history.append(loss)

            # Compute validation loss if validation data provided
            if X_val is not None and y_val is not None:
                y_pred_val = self.forward_propagation(X_val)
                val_loss = self.compute_loss(y_val, y_pred_val)
                val_loss_history.append(val_loss)

            # Print progress
            if epoch % 100 == 0 and self.verbose:
                if X_val is not None:
                    print(f"Epoch {epoch}: Train Loss = {loss:.4f}, Val Loss = {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch}: Train Loss = {loss:.4f}")

        return loss_history, val_loss_history

    def predict(self, X):
        """Make predictions"""
        y_pred = self.forward_propagation(X)

        if self.output_activation == 'sigmoid':
            return (y_pred >= 0.5).astype(int)
        else:
            return np.argmax(y_pred, axis=1).reshape(-1, 1)

    def predict_proba(self, X):
        """Predict class probabilities"""
        return self.forward_propagation(X)

def evaluate_model(y_true, y_pred, task_type='binary', verbose=True):
    """Calculate evaluation metrics"""
    if task_type == 'binary':
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    else:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    conf_matrix = confusion_matrix(y_true, y_pred)

    if verbose:
        print("\n" + "=" * 60)
        print("MODEL EVALUATION METRICS")
        print("=" * 60)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("=" * 60)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }

def plot_training_curves(loss_history, val_loss_history=None):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss')
    if val_loss_history is not None:
        plt.plot(val_loss_history, label='Validation Loss')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_configurations():
    """Compare different network configurations"""
    print("\n" + "=" * 70)
    print("NETWORK CONFIGURATION COMPARISON")
    print("=" * 70)

    # Generate binary classification dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                               n_redundant=5, n_clusters_per_class=1, random_state=42)
    y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Split training into train/validation
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )

    configurations = [
        {'hidden_sizes': [4], 'learning_rate': 0.01, 'activation': 'relu', 'name': 'Small Network (4 neurons)'},
        {'hidden_sizes': [8], 'learning_rate': 0.01, 'activation': 'relu', 'name': 'Medium Network (8 neurons)'},
        {'hidden_sizes': [16], 'learning_rate': 0.01, 'activation': 'relu', 'name': 'Large Network (16 neurons)'},
        {'hidden_sizes': [8], 'learning_rate': 0.01, 'activation': 'sigmoid', 'name': 'Sigmoid Activation'},
        {'hidden_sizes': [8], 'learning_rate': 0.001, 'activation': 'relu', 'name': 'Low Learning Rate (0.001)'},
    ]

    results = []

    for config in configurations:
        print(f"\n{'=' * 70}")
        print(f"Configuration: {config['name']}")
        print(f"Hidden Layers: {config['hidden_sizes']}, LR: {config['learning_rate']}, Activation: {config['activation']}")
        print("=" * 70)

        nn = NeuralNetwork(
            input_size=X_train.shape[1],
            hidden_sizes=config['hidden_sizes'],
            output_size=1,
            activation=config['activation'],
            output_activation='sigmoid',
            verbose=False
        )

        loss_history, val_loss_history = nn.fit(
            X_train_final, y_train_final,
            epochs=500,
            learning_rate=config['learning_rate'],
            batch_size=32,
            X_val=X_val,
            y_val=y_val
        )

        y_pred = nn.predict(X_test_scaled)
        metrics = evaluate_model(y_test, y_pred, task_type='binary', verbose=False)

        results.append({
            'config': config['name'],
            'metrics': metrics,
            'final_train_loss': loss_history[-1],
            'final_val_loss': val_loss_history[-1] if val_loss_history else None
        })

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"{'Configuration':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 70)
    for result in results:
        print(f"{result['config']:<30} {result['metrics']['accuracy']:<10.4f} "
              f"{result['metrics']['precision']:<10.4f} {result['metrics']['recall']:<10.4f} "
              f"{result['metrics']['f1']:<10.4f}")
    print("=" * 70)

def iris_classification():
    """Test on Iris dataset (multiclass)"""
    print("\n" + "=" * 70)
    print("IRIS DATASET CLASSIFICATION (MULTICLASS)")
    print("=" * 70)

    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target.reshape(-1, 1)

    # One-hot encode labels
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )

    print(f"Input features: {X.shape[1]}")
    print(f"Output classes: {y_onehot.shape[1]}")
    print(f"Training samples: {X_train_final.shape[0]}")

    nn = NeuralNetwork(
        input_size=X.shape[1],
        hidden_sizes=[8],
        output_size=y_onehot.shape[1],
        activation='relu',
        output_activation='softmax',
        verbose=True
    )

    print("\nTraining neural network...")
    loss_history, val_loss_history = nn.fit(
        X_train_final, y_train_final,
        epochs=300,
        learning_rate=0.05,
        batch_size=16,
        X_val=X_val,
        y_val=y_val
    )

    plot_training_curves(loss_history, val_loss_history)

    # Predictions
    y_pred_proba = nn.predict_proba(X_test_scaled)
    y_pred = np.argmax(y_pred_proba, axis=1).reshape(-1, 1)
    y_test_labels = np.argmax(y_test, axis=1).reshape(-1, 1)

    metrics = evaluate_model(y_test_labels, y_pred, task_type='multiclass')

def main():
    print("=" * 70)
    print("NEURAL NETWORK FOR CLASSIFICATION USING BACKPROPAGATION")
    print("=" * 70)

    # Binary classification example
    print("\n" + "=" * 70)
    print("BINARY CLASSIFICATION EXAMPLE")
    print("=" * 70)

    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                               n_redundant=5, n_clusters_per_class=1, random_state=42)
    y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )

    print(f"Input features: {X.shape[1]}")
    print(f"Training samples: {X_train_final.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    nn = NeuralNetwork(
        input_size=X.shape[1],
        hidden_sizes=[8],
        output_size=1,
        activation='relu',
        output_activation='sigmoid'
    )

    print("\nTraining neural network...")
    loss_history, val_loss_history = nn.fit(
        X_train_final, y_train_final,
        epochs=500,
        learning_rate=0.01,
        batch_size=32,
        X_val=X_val,
        y_val=y_val
    )

    plot_training_curves(loss_history, val_loss_history)

    y_pred = nn.predict(X_test_scaled)
    metrics = evaluate_model(y_test, y_pred, task_type='binary')

    # Compare different configurations
    compare_configurations()

    # Multiclass example (Iris)
    iris_classification()

if __name__ == "__main__":
    main()
