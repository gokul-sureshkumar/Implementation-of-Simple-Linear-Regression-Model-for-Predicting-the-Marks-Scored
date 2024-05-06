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

*/

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: GOKUL S
RegisterNumber:  212222110011




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()
df.tail()

#segregating data to variables
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual values
y_test

#graph plot for training data

plt.scatter(x_train,y_train,color="black") 
plt.plot(x_train,regressor.predict(x_train),color="red") 
plt.title("Hours VS scores (learning set)") 
plt.xlabel("Hours") 
plt.ylabel("Scores") 
plt.show()

#Graph plot for test data

plt.scatter(x_test,y_test,color="cyan")
plt.plot(x_test,regressor.predict(x_test),color="green")
plt.title("Hours VS scores (learning set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

import numpy as np
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```


## Output:

df.head():


![image](https://github.com/gokul-sureshkumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121148715/9d6dd763-1312-4309-b389-2dd183edc4d3)

df.tail:

![image](https://github.com/gokul-sureshkumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121148715/6d478690-6e85-48c5-a88a-9db67ad1d50d)

Array value of X:


![image](https://github.com/gokul-sureshkumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121148715/bd15ff62-80cf-4c8a-ab65-5a6f0c6cb06f)

Array value of Y:


![image](https://github.com/gokul-sureshkumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121148715/e91a7b41-2d68-4dab-b06c-d65b570333e9)

Values of Y prediction:

![image](https://github.com/gokul-sureshkumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121148715/3c542143-02b3-4ef3-b3ff-1dc8b612fb2d)

Values of Y test:


![image](https://github.com/gokul-sureshkumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121148715/edb36dd2-b659-4365-b711-6c01f09d97ff)

Training Set Graph:


![image](https://github.com/gokul-sureshkumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121148715/0fc24422-02aa-42eb-a173-773ae046b894)

Test Set Graph:

![image](https://github.com/gokul-sureshkumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121148715/66f9a4c7-bd72-4fa3-a6b5-13674c9460c6)

Values of MSE, MAE and RMSE:

![image](https://github.com/gokul-sureshkumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121148715/599f66db-4482-4ae1-af5f-2a99cd840aa0)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
