# Step 1 - Importing the Libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Loading or creating the Dataset 
iris = load_iris()


# Raw Dataset - DataFrame

data = pd.DataFrame(iris.data, columns=iris.feature_names)

data["target"] = iris.target
# print(data.head())

x = data.drop("target", axis = 1)
y = data["target"]

# Next Step - Split the Data

x_train, x_test , y_train , y_test = train_test_split(x,y, test_size = 0.2,random_state =42 )

# Training the ML model
# Decision Tree

model = DecisionTreeClassifier()



# Train Decision Tree
model.fit(x_train,y_train)

# Model Evaluation , Prediction
prediction = model.predict(x_test)
print(prediction)

# Check Accuracy
# Accuracy tells how many predictions are correct
accuracy = accuracy_score(y_test,prediction)

print(accuracy)

# Make a new prediction on real world data

new_flower = [[5.8,2.7,3.9,1.2]]

predictions = model.predict(new_flower)
print(iris.target_names[predictions])
predictions
