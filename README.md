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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

data=pd.read_csv("Employee.csv")
print("data.head():")
data.head()

print("data.info():")
data.info()

print("isnull() and sum():")
data.isnull().sum()

print("data value counts():")
data["left"].value_counts()

le=LabelEncoder()

print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()

print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_ac"]]
x.head()

y=data["left"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

print("Accuracy value:")
accuracy=metrics.accuracy_score(y_test,y_pred)
print(accuracy)

print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()

```

## Output:
<img width="1827" height="332" alt="image" src="https://github.com/user-attachments/assets/92bba902-7702-4b98-93fd-2f58a553b897" />

<img width="1802" height="658" alt="image" src="https://github.com/user-attachments/assets/2e8d433e-8756-4c77-bafe-953eb39c67d7" />

<img width="1752" height="525" alt="image" src="https://github.com/user-attachments/assets/d1c27ee2-3dd8-4386-8593-ea5c54e5109d" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
