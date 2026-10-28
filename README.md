# AI Laboratory Experiments

**Student Name:** Athar Bilal
**Enrollment Number:** 20204BITE009
**Course:** Artificial Intelligence Laboratory

## Overview

This repository contains implementations of 5 AI laboratory experiments covering search algorithms, genetic algorithms, game playing, and machine learning.

## Experiments

### Experiment 1: Intelligent 8-Puzzle Solver Using A* Search
- **File:** `ai lab/experiment1_8puzzle_astar.py`
- **Description:** Solves the 8-puzzle problem using A* search algorithm
- **Features:**
  - Manhattan distance heuristic
  - Misplaced tiles heuristic
  - Solvability detection
  - Performance comparison between heuristics
- **Run:** `python "ai lab/experiment1_8puzzle_astar.py"`

### Experiment 2: Genetic Algorithm for N-Queens Problem
- **File:** `ai lab/experiment2_nqueens_genetic.py`
- **Description:** Finds valid N-Queens arrangements using genetic algorithm
- **Features:**
  - Tournament selection
  - Single-point crossover with repair
  - Swap mutation
  - Elitism for preserving best solutions
  - Parameter sensitivity analysis
- **Run:** `python "ai lab/experiment2_nqueens_genetic.py"`

### Experiment 3: Minimax with Alpha-Beta Pruning
- **File:** `ai lab/experiment3_minimax_alphabeta.py`
- **Description:** Tic-Tac-Toe AI using Minimax with Alpha-Beta pruning
- **Features:**
  - Configurable search depth
  - Move ordering for efficiency
  - Performance comparison with/without pruning
  - Heuristic evaluation for non-terminal states
- **Run:** `python "ai lab/experiment3_minimax_alphabeta.py"`

### Experiment 4: Logistic Regression for Binary Classification
- **File:** `ai lab/experiment4_logistic_regression.py`
- **Description:** Logistic regression implementation from scratch
- **Features:**
  - Sigmoid activation function
  - Binary cross-entropy loss
  - Gradient descent optimization
  - Feature normalization
  - Parameter sensitivity analysis (learning rate, iterations, threshold)
  - Comparison with sklearn implementation
  - ROC-AUC evaluation
- **Dependencies:** `numpy scikit-learn matplotlib`
- **Run:** `python "ai lab/experiment4_logistic_regression.py"`

### Experiment 5: Neural Network for Classification Using Backpropagation
- **File:** `ai lab/experiment5_neural_network.py`
- **Description:** Feed-forward neural network with backpropagation
- **Features:**
  - Configurable hidden layers
  - Multiple activation functions (ReLU, sigmoid, tanh)
  - Mini-batch gradient descent
  - Binary and multiclass classification
  - Network configuration comparison
  - Iris dataset classification
- **Dependencies:** `numpy scikit-learn matplotlib`
- **Run:** `python "ai lab/experiment5_neural_network.py"`

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd DAA\ LAB
   ```

2. **Install required dependencies (for Experiments 4 & 5):**
   ```bash
   pip install numpy scikit-learn matplotlib
   ```

## Usage

Run each experiment individually:

```bash
cd "ai lab"
python experiment1_8puzzle_astar.py
python experiment2_nqueens_genetic.py
python experiment3_minimax_alphabeta.py
python experiment4_logistic_regression.py
python experiment5_neural_network.py
```

## Performance Results

### Experiment 1 (8-Puzzle)
- Manhattan distance: More efficient, fewer nodes expanded
- Misplaced tiles: Slightly more nodes, simpler to compute
- All test cases successfully solved

### Experiment 2 (N-Queens)
- 4-Queens: Solved in 1 generation
- 8-Queens: Solved in 2 generations
- 12-Queens: Requires more generations (2000 limit reached with 65/66 fitness)

### Experiment 3 (Minimax)
- Alpha-Beta pruning reduces node evaluation by 75-78%
- Significant speed improvement in complex game states

### Experiment 4 (Logistic Regression)
- Custom implementation: 97.5% accuracy
- Sklearn comparison: 98.0% accuracy
- Optimal learning rate: 0.1
- Optimal threshold: 0.5

### Experiment 5 (Neural Network)
- Binary classification: 98.5% accuracy
- Iris multiclass: 96.7% accuracy
- Best configuration: Single hidden layer with 8 neurons

## License

This repository is for educational purposes.

## Acknowledgments

- AI Laboratory Practice T-Sheet for problem specifications
- Python, NumPy, scikit-learn libraries
