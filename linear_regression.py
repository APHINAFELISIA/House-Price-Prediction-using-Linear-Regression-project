import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
data=pd.read_csv("house_price_dataset_150.csv")
print(data.shape)
print(data.shape)
print(data.info())
print(data.describe())
X=data[["Area_sqft"]]
Y=data["Price_INR"]
print(X.shape)
print(Y.shape)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
plt.scatter(x_train,y_train)
model=LinearRegression()
plt.scatter(x_train,y_train)
model.fit(x_train,y_train)
print(model.coef_)
print(model.intercept_)
predict_train=model.predict(x_train)
plt.scatter(x_train,y_train)
plt.plot(x_train,predict_train,color='r')
predictions = model.predict(x_test)
result=pd.DataFrame({"Actual":y_test,"Predicted":predictions})
print(result)
result.shape
mae=mean_absolute_error(y_test,predictions)
print(mae)
r2 = r2_score(y_test,predictions)
print(r2)
mse=mean_squared_error(y_test,predictions)
print(mse)
Area = float(input("Enter the area:"))
print(f"house price of {Area} has predicted price of {model.predict([[Area]])}")