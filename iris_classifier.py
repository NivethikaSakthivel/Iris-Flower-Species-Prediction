import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features (measurements)
y = iris.target  # Labels (species)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling (important for SVM models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model (Support Vector Machine)
model = SVC(kernel='linear')  # Using a linear kernel
model.fit(X_train, y_train)

# Test the model and print the accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Map numeric labels back to flower species names
species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

# Get user input for new flower measurements
print("\nEnter measurements for a new iris flower (in cm):")
sepal_length = float(input("Sepal length: "))
sepal_width = float(input("Sepal width: "))
petal_length = float(input("Petal length: "))
petal_width = float(input("Petal width: "))

# Predict the species of the new iris flower
new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
new_data = scaler.transform(new_data)  # Scale the input like the training data
prediction = model.predict(new_data)

# Output the predicted species
predicted_species = species_map[prediction[0]]
print(f"\nPredicted species: {predicted_species}")
