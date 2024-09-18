# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries such as NumPy, Pandas, Matplotlib, and metrics from sklearn.
2. Load the dataset into a Pandas DataFrame and preview it using `head()` and `tail()`.
3. Extract the independent variable X and dependent variable Y from the dataset.
4. Initialize the slope m and intercept c to zero. Set the learning rate L and define the number of epochs.
5. In a loop over the number of epochs:
   - Compute the predicted value Y_pred using the formula

     ![image](https://github.com/user-attachments/assets/ebd849e9-b41f-43c8-804f-d731cf9fd2fa)

   - Calculate the gradients:

     ![image](https://github.com/user-attachments/assets/794b5516-9a45-45c3-b86f-c732ec4f0b60)

   - Update the parameters m and c using the gradients and learning rate.
   - Track and store the error in each epoch.
6. Plot the error against the number of epochs to visualize the convergence.
7. Display the final values of m and c, and the error plot.

## Program:

# Program to implement the linear regression using gradient descent.
# Developed by: SARANYA S
# RegisterNumber:  212223110044
```
import numpy as np
import pandas as pd
from sklearn.metrics import  mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
```
```
dataset = pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
```
# OUTPUT:
![image](https://github.com/user-attachments/assets/f8f6f4c7-d0bb-4af2-94bd-b78151350a61)
```
dataset.info()
```
# OUTPUT:
![image](https://github.com/user-attachments/assets/3b214f2e-5bdc-4132-ab96-c7c5bd56b6b7)

```
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
```
# OUTPUT:
![image](https://github.com/user-attachments/assets/807f36ce-d4d2-4458-aa7a-d4e0ca1b800b)
![image](https://github.com/user-attachments/assets/3ce0e8bf-2868-42ea-8e30-8b2586eac026)

```
print(X.shape)
print(Y.shape)
```
# OUTPUT:
![image](https://github.com/user-attachments/assets/86a832bf-2943-40d7-9afc-091547b71796)
```
m=0
c=0
L=0.0001
epochs=5000
n=float(len(X))
error=[]
for i in range(epochs):
    Y_pred = m*X +c
    D_m=(-2/n)*sum(X *(Y-Y_pred))
    D_c=(-2/n)*sum(Y -Y_pred)
    m=m-L*D_m
    c=c-L*D_c
    error.append(sum(Y-Y_pred)**2)
print(m,c)
type(error)
print(len(error))
```
# OUTPUT:
![image](https://github.com/user-attachments/assets/61b593aa-5cc1-4367-a135-7b10090b318c)

```
plt.plot(range(0,epochs),error)
```
# OUTPUT:
![image](https://github.com/user-attachments/assets/e461ca97-1b19-44e5-9049-0cd59f8ae323)






## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
