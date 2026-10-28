"""
Experiment 4: Logistic Regression for Binary Classification
Implements Logistic Regression from scratch with performance evaluation.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, threshold=0.5, verbose=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.threshold = threshold
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.loss_history = []

    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))

    def binary_cross_entropy(self, y, y_pred):
        """Binary cross-entropy loss function"""
        epsilon = 1e-15  # Small value to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def fit(self, X, y):
        """Train the model using gradient descent"""
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []

        # Gradient descent
        for i in range(self.n_iterations):
            # Forward propagation
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)

            # Compute loss
            loss = self.binary_cross_entropy(y, y_pred)
            self.loss_history.append(loss)

            # Backward propagation (gradients)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Print progress every 100 iterations
            if i % 100 == 0 and self.verbose:
                print(f"Iteration {i}: Loss = {loss:.4f}")

    def predict_proba(self, X):
        """Predict class probabilities"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X):
        """Predict class labels"""
        y_pred_proba = self.predict_proba(X)
        return (y_pred_proba >= self.threshold).astype(int)

def evaluate_model(y_true, y_pred, y_pred_proba=None, verbose=True):
    """Calculate and print evaluation metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
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

        if y_pred_proba is not None:
            try:
                auc = roc_auc_score(y_true, y_pred_proba)
                print(f"ROC-AUC:   {auc:.4f}")
            except:
                print("ROC-AUC:   N/A (cannot compute)")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }

def plot_loss_curve(loss_history, title="Training Loss Over Iterations"):
    """Plot the training loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.grid(True)
    plt.show()

def plot_roc_curve(y_true, y_pred_proba):
    """Plot ROC curve"""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def experiment_parameters():
    """Test different learning rates, iterations, and thresholds"""
    print("\n" + "=" * 70)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)

    # Generate dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                               n_redundant=5, n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Test different learning rates
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    print("\nEffect of Learning Rate (iterations=1000, threshold=0.5):")
    print("-" * 70)
    print(f"{'Learning Rate':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 70)

    for lr in learning_rates:
        model = LogisticRegression(learning_rate=lr, n_iterations=1000, threshold=0.5, verbose=False)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        metrics = evaluate_model(y_test, y_pred, model.predict_proba(X_test_scaled), verbose=False)
        print(f"{lr:<15.3f} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")

    # Test different numbers of iterations
    iterations = [100, 500, 1000, 2000]
    print("\nEffect of Number of Iterations (learning_rate=0.01, threshold=0.5):")
    print("-" * 70)
    print(f"{'Iterations':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 70)

    for n_iter in iterations:
        model = LogisticRegression(learning_rate=0.01, n_iterations=n_iter, threshold=0.5, verbose=False)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        metrics = evaluate_model(y_test, y_pred, model.predict_proba(X_test_scaled), verbose=False)
        print(f"{n_iter:<15} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")

    # Test different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print("\nEffect of Classification Threshold (learning_rate=0.01, iterations=1000):")
    print("-" * 70)
    print(f"{'Threshold':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 70)

    for thresh in thresholds:
        model = LogisticRegression(learning_rate=0.01, n_iterations=1000, threshold=thresh, verbose=False)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        metrics = evaluate_model(y_test, y_pred, model.predict_proba(X_test_scaled), verbose=False)
        print(f"{thresh:<15.2f} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")

def compare_with_sklearn():
    """Compare custom implementation with sklearn's LogisticRegression"""
    print("\n" + "=" * 70)
    print("COMPARISON WITH SKLEARN LOGISTIC REGRESSION")
    print("=" * 70)

    from sklearn.linear_model import LogisticRegression as SklearnLR

    # Generate dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                               n_redundant=5, n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Custom implementation
    print("\nCustom Implementation:")
    custom_model = LogisticRegression(learning_rate=0.01, n_iterations=1000, threshold=0.5)
    custom_model.fit(X_train_scaled, y_train)
    y_pred_custom = custom_model.predict(X_test_scaled)
    metrics_custom = evaluate_model(y_test, y_pred_custom, custom_model.predict_proba(X_test_scaled))

    # Sklearn implementation
    print("\nSklearn Implementation:")
    sklearn_model = SklearnLR(random_state=42, max_iter=1000)
    sklearn_model.fit(X_train_scaled, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test_scaled)
    y_pred_proba_sklearn = sklearn_model.predict_proba(X_test_scaled)[:, 1]
    metrics_sklearn = evaluate_model(y_test, y_pred_sklearn, y_pred_proba_sklearn)

    # Comparison
    print("\n" + "-" * 70)
    print(f"{'Metric':<15} {'Custom':<15} {'Sklearn':<15} {'Difference':<15}")
    print("-" * 70)
    print(f"{'Accuracy':<15} {metrics_custom['accuracy']:<15.4f} {metrics_sklearn['accuracy']:<15.4f} {abs(metrics_custom['accuracy'] - metrics_sklearn['accuracy']):<15.4f}")
    print(f"{'Precision':<15} {metrics_custom['precision']:<15.4f} {metrics_sklearn['precision']:<15.4f} {abs(metrics_custom['precision'] - metrics_sklearn['precision']):<15.4f}")
    print(f"{'Recall':<15} {metrics_custom['recall']:<15.4f} {metrics_sklearn['recall']:<15.4f} {abs(metrics_custom['recall'] - metrics_sklearn['recall']):<15.4f}")
    print(f"{'F1-Score':<15} {metrics_custom['f1']:<15.4f} {metrics_sklearn['f1']:<15.4f} {abs(metrics_custom['f1'] - metrics_sklearn['f1']):<15.4f}")
    print("-" * 70)

def main():
    print("=" * 70)
    print("LOGISTIC REGRESSION FOR BINARY CLASSIFICATION")
    print("=" * 70)

    # Generate synthetic dataset
    print("\nGenerating synthetic dataset...")
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                               n_redundant=5, n_clusters_per_class=1, random_state=42)
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: Class 0: {np.sum(y == 0)}, Class 1: {np.sum(y == 1)}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features normalized using StandardScaler")

    # Train model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(learning_rate=0.01, n_iterations=1000, threshold=0.5)
    model.fit(X_train_scaled, y_train)

    # Plot training loss
    plot_loss_curve(model.loss_history, "Training Loss Over Iterations")

    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)

    # Evaluate model
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)

    # Plot ROC curve
    plot_roc_curve(y_test, y_pred_proba)

    # Parameter sensitivity analysis
    experiment_parameters()

    # Compare with sklearn
    compare_with_sklearn()

if __name__ == "__main__":
    main()
