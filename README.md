# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: R.Bharathi Shankar
RegisterNumber: 212225230032 
*/
import pandas as pd

# Load Dataset
data = pd.read_csv(r"C:\Users\acer\Downloads\Employee.csv")

# Display first 5 rows
print("data.head():")
print(data.head())

# Dataset information
print("\ndata.info():")
print(data.info())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Display column names
print("\nColumn Names:")
print(data.columns)

# Count values of target column
print("\nEmployee Left Count:")
print(data["left"].value_counts())

# Encode salary column
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

# Input Features
x = data[[
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident"
]]

# Output Feature
y = data["left"]

# Split Dataset
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=100
)

# Create Decision Tree Model
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion="entropy")

# Train Model
dt.fit(x_train, y_train)

# Predict Test Data
y_pred = dt.predict(x_test)

# Accuracy
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test, y_pred)

print("\nAccuracy value:")
print(accuracy)

# Single Prediction
print("\nData Prediction:")

sample = pd.DataFrame(
    [[0.5, 0.8, 9, 260, 6, 0]],
    columns=[
        "satisfaction_level",
        "last_evaluation",
        "number_project",
        "average_montly_hours",
        "time_spend_company",
        "Work_accident"
    ]
)

prediction = dt.predict(sample)

if prediction[0] == 1:
    print("Employee may leave the company")
else:
    print("Employee may stay in the company")

# Plot Decision Tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))

plot_tree(
    dt,
    feature_names=x.columns,
    class_names=["Stay", "Left"],
    filled=True
)

plt.show()
```

## Output:
<img width="919" height="255" alt="Screenshot 2026-05-27 105516" src="https://github.com/user-attachments/assets/d2bb41cf-b2f8-4707-bc4e-254db8a2567b" />
<img width="438" height="107" alt="Screenshot 2026-05-27 105537" src="https://github.com/user-attachments/assets/70f33f56-64af-4cf4-b845-8b21469cc35b" />
<img width="561" height="349" alt="Screenshot 2026-05-27 105555" src="https://github.com/user-attachments/assets/5c3a5fa7-07a4-4fa4-9f46-f5d4d098a5fc" />
<img width="994" height="322" alt="Screenshot 2026-05-27 105610" src="https://github.com/user-attachments/assets/d3b68b19-6715-481d-a064-3f23e7bb122f" />
<img width="432" height="211" alt="Screenshot 2026-05-27 105621" src="https://github.com/user-attachments/assets/612e2352-42d8-4631-ac0a-46b3a25fa685" />
<img width="1184" height="358" alt="Screenshot 2026-05-27 105657" src="https://github.com/user-attachments/assets/82d7043c-cb19-4f59-b6a2-409a8460e7ba" />
<img width="1184" height="358" alt="Screenshot 2026-05-27 105657" src="https://github.com/user-attachments/assets/f1496ec0-e14c-4df3-bdb9-0ea97899007f" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
