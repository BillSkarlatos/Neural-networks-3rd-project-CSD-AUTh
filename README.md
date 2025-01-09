# Neural Networks - 3rd Project (CSD AUTh)

This repository contains the implementation of the third project for the Neural Networks course at the Computer Science Department of Aristotle University of Thessaloniki (CSD AUTh). The project involves the use of various machine learning techniques and algorithms for classification tasks, particularly on the CIFAR-10 dataset.

## Project Overview

The project includes the following main components:

- **Hebbian Learning**: An implementation of Hebbian learning for associative pattern recognition.
- **k-Nearest Neighbors (KNN)**: A classic supervised learning algorithm used for classification.
- **Nearest Class Centroid (NCC)**: A centroid-based classification method.

The CIFAR-10 dataset is used as the primary dataset for evaluation. Dimensionality reduction techniques like PCA are employed to preprocess the data before applying the algorithms.

## Repository Structure

```
.
├── data                # Directory for storing the CIFAR-10 dataset
├── scripts             # Python scripts for different algorithms and utilities
├── results             # Directory to store output plots, metrics, and results
├── README.md           # Project documentation
└── requirements.txt    # Required Python packages
```

## Features

- **CIFAR-10 Dataset Loading**: Automates downloading and loading the CIFAR-10 dataset.
- **Dimensionality Reduction**: Uses PCA for reducing the dimensionality of image data.
- **KNN Classification**: Implements the k-Nearest Neighbors algorithm with user-defined `k`.
- **NCC Classification**: Implements the Nearest Class Centroid classifier.
- **Hebbian Learning**: Implements a Hebbian learning-based classifier.

## Installation

To run the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/BillSkarlatos/Neural-networks-3rd-project-CSD-AUTh.git
cd Neural-networks-3rd-project-CSD-AUTh
pip install -r requirements.txt
```

## Usage

### Dataset Loading

The CIFAR-10 dataset is loaded and preprocessed using the `loadDatabase()` function. Transformations like normalization and PCA are applied to prepare the data for classification.

### Running Algorithms

- **Hebbian Learning**:

  ```python
  from scripts.hebbian_learning import hebbian_learning
  weights = hebbian_learning(x_train_pca, y_train_onehot, learning_rate=0.01, n_epochs=10)
  ```

- **k-Nearest Neighbors**:

  ```python
  from scripts.knn import KNN
  accuracy = KNN(k=3)
  ```

- **Nearest Class Centroid**:

  ```python
  from scripts.ncc import NCC
  accuracy = NCC()
  ```

### Results

Results such as accuracy and runtime are displayed in the console. Plots and distributions (e.g., weight histograms) are saved in the `results/` directory.

## Dependencies

The project requires the following Python libraries:

- `torch`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `tqdm`
- `torchvision`

You can install all dependencies using:

```bash
pip install -r deps.txt
```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, feel free to open an issue or create a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

Thank you for exploring this project!
