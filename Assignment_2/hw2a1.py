import PyQt5
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df=pd.read_excel('hw2.xlsx')
df['Average Market Value']=df['Average Market Value'].str.replace(',', '').astype(float)
df['Age']=df['Age'].str.replace(',', '.').astype(float)

#df=shuffle(df)

y = df['Pts']
X =df['Average Market Value'].values[:,np.newaxis]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print('Weight coefficients: ', regressor.coef_)
print('y-axis intercept: ', regressor.intercept_)

min_pt = X.min() * regressor.coef_[0] + regressor.intercept_
max_pt = X.max() * regressor.coef_[0] + regressor.intercept_

plt.plot([X.min(), X.max()], [min_pt, max_pt])
plt.plot(X_train, y_train, 'o')
plt.title("Regression fit using intercepts and train data")
plt.show()

y_pred_train = regressor.predict(X_train)

plt.plot(X_train, y_train, 'o', label="data")
plt.plot(X_train, y_pred_train, 'o', label="prediction")
#plt.plot([X.min(), X.max()], [min_pt, max_pt], label='fit')
plt.legend(loc='best')
plt.title("Predictions on train data")
plt.show()

y_pred_test = regressor.predict(X_test)

plt.plot(X_test, y_test, 'o', label="data")
plt.plot(X_test, y_pred_test, 'o', label="prediction")
#plt.plot([X.min(), X.max()], [min_pt, max_pt], label='fit')
plt.legend(loc='best')
plt.title("Predictions on test data")
plt.show()

print(regressor.score(X_test, y_test))
print(((y_pred_test - y_test) ** 2).mean())