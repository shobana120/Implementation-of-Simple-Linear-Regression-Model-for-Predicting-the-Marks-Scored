# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the Program.
2. Import the necessary packages.
3. Read the given csv file and display the few contents of the data.
4. Assign the features for x and y respectively.
5. Split the x and y sets into train and test sets.
6. Convert the Alphabetical data to numeric using CountVectorizer.
7. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
8. Find the accuracy of the model. 9.End the Program.


## Program:
```
*/
Program to implement the SVM For Spam Mail Detection..
Developed by: B SHOBANA
RegisterNumber:  212224230262
*/
```
```py
import pandas as pd

data=pd.read_csv("spam.csv",encoding="Windows-1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
<img width="320" height="569" alt="image" src="https://github.com/user-attachments/assets/a021a621-3e57-4688-90a5-3c6e522e5b4c" />
<img width="320" height="569" alt="image" src="https://github.com/user-attachments/assets/1c91e4c8-1070-49fb-81cb-e504ff8b605d" />
<img width="320" height="569" alt="image" src="https://github.com/user-attachments/assets/48d5e626-a0e6-40fb-bd77-c88280d3a0f8" />
<img width="776" height="591" alt="image" src="https://github.com/user-attachments/assets/2071eddf-3225-459a-a09b-7d39791dc06e" />
![Uploading image.png…]()
![Uploading image.png…]()









## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
