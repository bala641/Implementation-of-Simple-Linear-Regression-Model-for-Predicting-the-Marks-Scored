# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the standard Libraries. 
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph
6. Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```PYTHON
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Bala B
RegisterNumber: 212224100005 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
DATASET :
![Screenshot 2025-03-08 144051](https://github.com/user-attachments/assets/96e58a35-63e3-41aa-a770-a3cdc399549d)

HEAD VALUES :
![Screenshot 2025-03-08 144101](https://github.com/user-attachments/assets/12196372-611f-41c2-98c1-296b027a92ac)

TAIL VALUES:
![Screenshot 2025-03-08 144110](https://github.com/user-attachments/assets/779e802c-ac01-4996-acf7-d4c76d27e6e7)

X AND Y VALUES :
![Screenshot 2025-03-08 144128](https://github.com/user-attachments/assets/bd63e730-3b5b-4e87-8b85-4ebdd64b9888)

PREDICTION VALUES OF X AND Y :
![Screenshot 2025-03-08 144146](https://github.com/user-attachments/assets/17deade3-541b-4183-8965-c640f1fa9761)

MSE,MAE,RMSE VALUES :
![Screenshot 2025-03-08 144233](https://github.com/user-attachments/assets/1c335a26-02b7-4ff4-99a1-3daff18dbfd6)

TRAINING SET :
![Screenshot 2025-03-08 144208](https://github.com/user-attachments/assets/0b4e5ec3-8877-48ae-871a-2348b7f38dad)

TESTING SET:
![Screenshot 2025-03-08 144226](https://github.com/user-attachments/assets/0f4d0e32-a153-4c10-876d-d379e8343290)






## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
