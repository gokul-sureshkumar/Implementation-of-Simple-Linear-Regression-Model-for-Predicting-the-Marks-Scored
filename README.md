# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

STEP 1. Start the program.

STEP 2.Import the standard Libraries.

STEP 3.Set variables for assigning dataset values.

STEP 4.Import linear regression from sklearn.

STEP 5.Assign the points for representing in the graph.

STEP 6.Predict the regression for marks by using the representation of the graph.

STEP 7.End the program.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: GOKUL S
RegisterNumber:  212222110011
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/SEC/Downloads/student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours Vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_train,regressor.predict(X_train),color='yellow')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)  
```

## Output:


### df.head()
![image](https://github.com/VARSHINI22009118/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119401150/b167b189-955e-4a58-98ed-81521b6a2307)

### df.tail()
![image](https://github.com/VARSHINI22009118/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119401150/6e5d3991-a656-40dd-b590-f0cde57a9df7)



### Array value of X
![image](https://github.com/VARSHINI22009118/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119401150/cbdade85-e2f8-4033-99ca-c150a2540f4d)

### Array value of Y
![image](https://github.com/VARSHINI22009118/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119401150/bd500e67-4f89-4d8e-ac02-7a30adb63ea3)

### Values of y prediction
![image](https://github.com/VARSHINI22009118/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119401150/0fb49543-18a3-462b-83e8-48da46913b68)



### Array values of Y test
![image](https://github.com/VARSHINI22009118/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119401150/b84929fa-8bad-4fdd-8c4c-9c4b27499f28)

### Training set graph
![Screenshot (1234)](https://github.com/VARSHINI22009118/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119401150/67edac83-7c77-495f-8a34-0f5d5a7659df)

### Test set graph
![Screenshot (1235)](https://github.com/VARSHINI22009118/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119401150/3fc3a7b4-9592-44cb-bd08-0db9b71100c7)

### Values of MSE,MAE and RMSE
![image](https://github.com/VARSHINI22009118/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119401150/be305b14-73bf-4a15-ae7f-7ade05dfec07)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
