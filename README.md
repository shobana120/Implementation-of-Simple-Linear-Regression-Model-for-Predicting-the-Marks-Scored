# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: B.shobana
RegisterNumber: 212224230262
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        # Load dataset
        df = pd.read_csv("student_scores.csv")
        
        # Splitting into features (Hours) and target (Scores)
        X = df[['Hours']]
        Y = df['Scores']
        
        # Splitting the dataset into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
        
        # Creating and training the Linear Regression model
        regressor = LinearRegression()
        regressor.fit(X_train, Y_train)
        
        # Predicting test set results
        Y_pred = regressor.predict(X_test)
        
        # Visualizing the Training set
        plt.scatter(X_train, Y_train, color="blue")
        plt.plot(X_train, regressor.predict(X_train), color="black")
        plt.title("Hours vs Scores (Training Set)")
        plt.xlabel("Hours Studied")
        plt.ylabel("Marks Scored")
        plt.show()
        
        # Visualizing the Test set
        plt.scatter(X_test, Y_test, color="yellow")
        plt.plot(X_test, Y_pred, color="black")
        plt.title("Hours vs Scores (Test Set)")
        plt.xlabel("Hours Studied")
        plt.ylabel("Marks Scored")
        plt.show()
        
        # Model Evaluation
        mse = mean_squared_error(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)
        rmse = np.sqrt(mse)
        
        print(f"MSE  = {mse:.2f}")
        print(f"MAE  = {mae:.2f}")
        print(f"RMSE = {rmse:.2f}")

```

## Output:

<img width="464" height="547" alt="Screenshot 2025-09-03 154032" src="https://github.com/user-attachments/assets/5e2c07c0-ab0c-4015-911a-e698055edb6d" />



<img width="834" height="580" alt="Screenshot 2025-09-03 154041" src="https://github.com/user-attachments/assets/67510f8a-ff44-4711-9a8e-18450dbfd9fe" />



<img width="749" height="671" alt="Screenshot 2025-09-03 154057" src="https://github.com/user-attachments/assets/1b0e58ca-e30a-49fe-b61f-5959c27a326b" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
