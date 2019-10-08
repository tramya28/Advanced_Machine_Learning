import PyQt5
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold 


df=pd.read_excel('hw2.xlsx')
df['Average Market Value']=df['Average Market Value'].str.replace(',', '').astype(float)
k = 20
#n_samples = len(X)
fold_size = 2
scores = []
masks = []
best_x_train= [[] for _ in range(20)]
best_x_test  = [[] for _ in range(20)]
best_y_train=[[] for _ in range(20)]
best_y_true=[[] for _ in range(20)]
best_y_pred=[[] for _ in range(20)]
for fold in range(k):
    df_1=shuffle(df)
    y = df_1['Pts']
    X =df_1['Average Market Value'].values[:,np.newaxis]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)
    # fit the classifier
    best_x_train[fold].append(X_train)
    best_x_test[fold].append(X_test)
    best_y_train[fold].append(y_train)
    best_y_true[fold].append(y_test)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred_test = regressor.predict(X_test)
    best_y_pred[fold].append(y_pred_test)
    scores.append(regressor.score(X_test, y_test))
    
best=scores.index(max(scores))

plt.scatter(best_x_test[best],best_y_true[best], c='green')
plt.scatter(best_x_test[best],best_y_pred[best], c='red')
plt.title('Best of 20 predictions(red) vs Actual(green)')
plt.xlabel('Team')
plt.ylabel('Points')
plt.show()
print(scores.index(max(scores)))
#scores


df['Avg_Problem1']=float(0)
for j in range(0,20):
    for i in range(0,len(list(best_x_test[j][0]))):
        match=float(best_x_test[j][0][i])
        index=df.index[df['Average Market Value'] == match].tolist()[0]
        df['Avg_Problem1'][index]=df['Avg_Problem1'][index]+best_y_pred[j][0][i]
df['Avg_Problem1']= df['Avg_Problem1']/20

plt.scatter(df['Average Market Value'],df['Pts'], c='green')
plt.scatter(df['Average Market Value'],df['Avg_Problem1'], c='red')
plt.title('Average of 20 predictions(red) vs Actual(green)')
plt.xlabel('Team')
plt.ylabel('Points')
plt.show()

print(((df['Pts'] - df['Avg_Problem1']) ** 2).mean())
