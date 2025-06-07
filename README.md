## BLENDED_LEARNING
## Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor
### DATE:15-05-2025
## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import necessary libraries (pandas, numpy, sklearn, matplotlib).
2. Load the dataset using pandas.
3. Preprocess the data:

   * Drop unnecessary columns (e.g., CarName, car\_ID).
   * Convert categorical variables using one-hot encoding.
4. Split the dataset into features (X) and target (Y), then into training and testing sets.
5. Standardize the features and target using StandardScaler.
6. Initialize the SGDRegressor model with appropriate parameters.
7. Train the model on the training data.
8. Predict the target values for the test data.
9. Evaluate the model using Mean Squared Error and R² score.
10. Display model coefficients and intercept.
11. Visualize actual vs predicted values with a scatter plot.
12. End of workflow.

## Program:
```py
Program to implement SGD Regressor for linear regression.
Developed by: YASHWANTH RAJA DURAI V
RegisterNumber: 212222040184
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('CarPrice_Assignment.csv')
print(data.head())
print("\n\n")
print(data.info())

# Data preprocessing
# Dropping unnecessary columns and handling categorical variables
data = data.drop(['CarName', 'car_ID'], axis=1)
data = pd.get_dummies(data, drop_first=True)

# Splitting the data into features and target variable
x = data.drop('price', axis=1)
y = data['price']

# Standardizing the data
scaler = StandardScaler()
x = scaler.fit_transform(x)
y = scaler.fit_transform(np.array(y).reshape(-1, 1)).ravel()


# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Creating the SGD Regressor model
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)

# Fitting the model on the training data
sgd_model.fit(x_train, y_train)

# Making predictions
y_pred = sgd_model.predict(x_test)

# Evaluating model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print()
print()
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Print model coefficients
print("\n\n")
print("Model Coefficients")
print("Coefficients:", sgd_model.coef_)
print("Intercept:", sgd_model.intercept_)

# Visualizing actual vs predicted prices
print("\n\n")
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices using SGD Regressor")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Perfect prediction line
plt.show()

```

## Output:
### LOAD THE DATASET:
![Screenshot 2025-05-14 134927](https://github.com/user-attachments/assets/ef77303a-ac3b-47df-8ff6-696a6914dd27)
![Screenshot 2025-05-14 134932](https://github.com/user-attachments/assets/7fdc9780-4a58-4a75-a69f-8c1226869532)

### EVALUATION METRICS:
![Screenshot 2025-05-14 134937](https://github.com/user-attachments/assets/7db473af-3d52-4090-87d6-78412a7e83c1)
### MODEL COEFFICIENTS:
![Screenshot 2025-05-14 134941](https://github.com/user-attachments/assets/4c1ec1c6-0eff-42b5-be27-628072441618)
### VISUALIZATION OF ACTUAL VS PREDICTED VALUES:
![Screenshot 2025-05-14 134945](https://github.com/user-attachments/assets/35ff3fb4-5eb7-4bd3-9a02-0f996c1ec3de)

## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
