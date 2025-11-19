# BLENDED_LEARNING
# Implementation of Decision Tree Model for Tumor Classification

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import Libraries
Import pandas, scikit-learn, seaborn, and matplotlib for data handling, modeling, and visualization.
2. Load Dataset
Load the dataset from the provided URL and explore its structure.
3. Feature Selection
Define features (X) by excluding irrelevant columns like id and set diagnosis as the target (y).
4. Split Dataset
Split the data into training and testing sets (70-30 ratio).
5. Train Model
Train a Decision Tree Classifier on the training data.
6. Evaluate Model
Predict using the test set and calculate accuracy, generate a classification report, and confusion matrix.
7. Visualize Results
Plot a heatmap of the confusion matrix to assess prediction performance. 

## Program:
```
/*
Program to  implement a Decision Tree model for tumor classification.
Developed by: PAGADALA MITHUN KALYAN
RegisterNumber: 212223040142

#Import necessary libraries

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#Step 1: Data Loading

data = pd.read_csv('tumor.csv')

#Step 2: Data Exploration
#Display the first few rows and column names for verification
print(data.head())
print(data.columns)

#Step 3: Select features and target variable

#Drop id and other non-feature columns, using diagnosis as the target
x = data.drop(columns=['Class']) # Remove any irrelevant columns
y = data['Class'] # The target column indicating benign or malignant diagnosis

#Step 4: Data Splitting

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#Step 5: Model Training
#Initialize and train the Decision Tree Classifier

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

#Step 6: Model Evaluation
#Predicting on the test set

y_pred = model.predict(X_test)

#Calculate accuracy and print classification metrics

accuracy = accuracy_score(y_test, y_pred)
print("PAGADALA MITHUN KALYAN")
print("212223040142")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

#Confusion Matrix

conf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlOrRd")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
*/
```

## Output:

<img width="617" height="280" alt="image" src="https://github.com/user-attachments/assets/cc98dca0-f172-4ee0-aa0f-d001a1d52111" />

<img width="665" height="562" alt="image" src="https://github.com/user-attachments/assets/7997acd8-9a12-4e91-853b-6cf5a35a71d1" />

## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.
