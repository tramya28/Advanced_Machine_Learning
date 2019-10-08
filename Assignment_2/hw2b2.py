import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold 


df=pd.read_excel('hw2.xlsx')
df['Age']=df['Age'].str.replace(',', '').astype(float)

X =np.array(df[['Foreigners','Age']])#.values[:,np.newaxis]
y = df['Pts']
k = 20
n_samples = len(X)
fold_size = 2
scores = []
best_x_train= [[] for _ in range(20)]
best_x_test  = [[] for _ in range(20)]
best_y_train=[[] for _ in range(20)]
best_y_true=[[] for _ in range(20)]
best_y_pred=[[] for _ in range(20)]
for fold in range(20):
    df_1=shuffle(df)
    X =np.array(df_1[['Foreigners','Age']])#.values[:,np.newaxis]
    y = df_1['Pts']
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
    # compute the score and record it
    scores.append(regressor.score(X_test, y_test))

#print(scores)
best=scores.index(max(scores))
plt.scatter(np.arange(10),best_y_true[best], c='green')
plt.scatter(np.arange(10),best_y_pred[best], c='red')
plt.title('Best of 20 predictions(red) vs Actual(green)')
plt.xlabel('Team')
plt.ylabel('Points')
plt.show()

df['Avg_Problem2']=float(0)
for j in range(0,20):
    for i in range(0,len(list(best_x_test[j][0]))):
        match1=int(best_x_test[j][0][i][0])
        match2=float(best_x_test[j][0][i][1])
        index=df.index[(df['Foreigners'] == match1) & (df['Age'] == match2 )].tolist()[0]
        df['Avg_Problem2'][index]=df['Avg_Problem2'][index]+best_y_pred[j][0][i]
df['Avg_Problem2']= df['Avg_Problem2']/20

plt.scatter(np.arange(0,20),df['Pts'], c='green')
plt.scatter(np.arange(0,20),df['Avg_Problem2'], c='red')
plt.title('Average of 20 predictions(red) vs Actual(green)')
plt.xlabel('Team')
plt.ylabel('Points')
plt.show()

print(((np.array(best_y_true[best]) -np.array(best_y_pred[best])) ** 2).mean())