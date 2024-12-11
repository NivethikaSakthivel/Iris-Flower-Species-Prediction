# Iris Flower Classification using Machine Learning

This project implements a machine learning model to classify iris flowers into one of three species—Setosa, Versicolor, and Virginica—based on their measurements (sepal length, sepal width, petal length, and petal width). The project uses the **Iris dataset** from `scikit-learn` and a **Support Vector Machine (SVM)** model for classification.

## Project Overview

The dataset contains features of 150 iris flowers from three different species. The goal of the project is to create a machine learning model that can predict the species of an iris flower based on its measurements.

### The model is trained using the following steps:
1. **Data Loading**: The Iris dataset is loaded from `scikit-learn`'s built-in datasets.
2. **Data Preprocessing**: Data is split into training and testing sets. Features are scaled using `StandardScaler` to improve model performance.
3. **Model Training**: A Support Vector Machine (SVM) classifier with a linear kernel is trained on the dataset.
4. **Model Evaluation**: The accuracy of the model is tested on a separate test set and displayed.
5. **User Input**: The user can input the measurements of a new iris flower, and the model predicts its species.

### Features:
- Predicts the species of a new iris flower based on user-provided measurements.
- Utilizes machine learning (SVM) for classification.
- High accuracy on the Iris dataset (approximately 97.78%).

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/iris-flower-classification.git
   cd iris-flower-classification
   ```

2. Install the required libraries:
   ```bash
   pip install scikit-learn numpy pandas
   ```

3. Run the script:
   ```bash
   python iris_classifier.py
   ```

4. Follow the on-screen instructions to enter flower measurements and get the predicted species.

## Usage Example

After running the script, you'll be prompted to enter the following measurements for a new iris flower:
- Sepal length
- Sepal width
- Petal length
- Petal width

The model will then predict the species based on the provided inputs.

### Example:
```
Enter measurements for a new iris flower (in cm):
Sepal length: 5.1
Sepal width: 3.5
Petal length: 1.4
Petal width: 0.2

Predicted species: Setosa
```

## Technologies Used:
- Python
- scikit-learn
- numpy
- pandas
