# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries such as pandas, sklearn, seaborn, and matplotlib
2. Load the dataset food_items_binary.csv using pandas.read_csv().
3. Select the input features and the target variable
4. plit the dataset into training and testing sets using train_test_split().
5. Standardize the data using StandardScaler to normalize the feature values
6. nitialize the SVM classifier using SVC()
7. Apply GridSearchCV to tune hyperparameters (C, kernel, gamma) using cross-validation
8. Train the model and make predictions, then evaluate it using accuracy score, classification report, and confusion matrix
9. Visualize the confusion matrix using a Seaborn heatmap and end the program

## Program:
```
/*
Program to implement SVM for food classification for diabetic patients.
Developed by: sri jai.v
RegisterNumber: 25018437 
*/
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('food_items_binary.csv')

print(data.head())
print(data.columns)

features = ['Calories', 'Total Fat', 'Saturated Fat', 'Sugars', 'Dietary Fiber', 'Protein']
target = 'class' 

X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm = SVC()

param_grid = {
    'C': [0.1, 1, 10, 100],          
    'kernel': ['linear', 'rbf'],
    'gamma':['scale','auto']
}


grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Name:SRIJAI V")
print("Reg. No: 25018437")
print("Best Parameters:", grid_search.best_params_)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Name:SRI JAI.V")
print("Reg No:25018437")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## Output:

<img width="1423" height="684" alt="Screenshot 2026-03-09 145313" src="https://github.com/user-attachments/assets/2f7373e6-5b09-42ef-91f8-1682e8520dbc" />
<img width="786" height="394" alt="Screenshot 2026-03-31 003756" src="https://github.com/user-attachments/assets/56c95799-5252-4e63-a4fe-a4caebcd564f" />
<img width="924" height="677" alt="Screenshot 2026-03-31 003812" src="https://github.com/user-attachments/assets/4d2f7cf4-bcc0-49c0-a6b2-16ecc3f76c02" />



## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
