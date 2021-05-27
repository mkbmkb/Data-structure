





import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn import datasets
import numpy as np
import pandas as pd
import statsmodels.api as sm

#8.3(1)
'''
data = [[450.5,4,171.2],[507.7,	4,	174.2],
[613.9	,5	,204.3],
[563.4	,4,	218.7],
[501.5	,4,	219.4],
[781.5	,7,	240.4],
[541.8,	4	,273.5],
[611.1	,5	,294.8],
[1222.1,	10,	330.2],
[793.2,	7,	333.1],
[660.8,	5,	366],
[792.7,	6,	350.9],
[580.8,	4,	357.9],
[612.7,	5,	359],
[890.8	,7,	371.9],
[1121,	9,	435.3],
[1094.2	,8,	523.9],
[1253	,10,	604]]
'''

# columns = ['y', 'x1', 'x2']
# data = np.array(data)
#
# y = data[:,0]
# x = data[:,1:]
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
#
# lr = LinearRegression()
# lr.fit(x_train, y_train)  # 把训练集的自变量，因变量添加到函数fit()中，进行训练
# print(lr.coef_)  # lr.coef_得到的是每个自变量的权重系数
# print(lr.intercept_)  # lr.intercept_得到的是截距
#
# #8.3(2)
# print(103.92008448 * 10 + 0.46166575 * 480 -21.109324426838157 )
 # 1239.6910803731616
#URL:https://juejin.cn/post/6872980901308055566

#8.7
#手动添加一组常量
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.colors import ListedColormap
#
# def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
#     markers = ('s', 'x', 'o', '^', 'v')
#     colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
#     cmap = ListedColormap(colors[:len(np.unique(y))])
#     x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
#     Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#     Z = Z.reshape(xx1.shape)
#     plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
#     plt.xlim(xx1.min(), xx1.max())
#     plt.ylim(xx2.min(), xx2.max())
#     for idx, cl in enumerate(np.unique(y)):
#         plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)
#     if test_idx:
#         X_test, y_test = X[test_idx, :], y[test_idx]
#         plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1, marker='o', s=55, label='test set')
#
#
# data = [[5,0.16,0.000001],
# [10, 0.255,0.000001],
# [15, 0.35,0.000001],
# [20,0.515,0.000001],
# [30,0.74,0.000001]]
#
# data = np.array(data)
#
# y = data[:,0]
# x = data[:,1:]
#
# X_train,X_test,y_train,y_test= train_test_split(x,y,test_size=0.3)
# print(X_test)
#
#
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# sc.fit(X_train)
# X_train_std = sc.transform(X_train)
# X_test_std = sc.transform(X_test)
#
# X_combined_std = np.vstack((X_train_std, X_test_std))
# y_combined = np.hstack((y_train, y_test))
#
# from sklearn.linear_model import LogisticRegression
#
# lr = LogisticRegression()
# lr.fit(X_train, y_train)
#
# plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(1,5))
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')
# plt.legend(loc='upper left')
# plt.show()
#
#
# X_test = [[2.5e-01,1.00e-06],
#          [2.5e-01,1.00e-06]]
# print(lr.predict(X_test))


'''
# URL:https://blog.csdn.net/xlinsist/article/details/51289825
# https://www.pythonf.cn/read/138452
'''

#8.8
data = [[0,10,0.000001],
[1,17,0.000001],
[1,18,0.000001],
[0,14,0.000001],
[0,12,0.000001],
[1,9,0.000001],
[1,20,0.000001],
[0,13,0.000001],
[0,9,0.000001],
[1,	19,0.000001],
[0,	12,0.000001],
[0,	4,0.000001],
[1,	14,0.000001],
[1,	20,0.000001],
[0,	6,0.000001],
[1,	19,0.000001],
[0,	11,0.000001],
[0,	10,0.000001],
[1,	17,0.000001],
[0,	13,0.000001],
[1,	21,0.000001],
[1,	16,0.000001],
[0,	12,0.000001],
[0,11,0.000001],
[1,	16,0.000001],
[0,	11,0.000001],
[1,	20,0.000001],
[1,	18,0.000001],
[1,	16,0.000001],
[0,10,0.000001],
[0,	8,0.000001],
[0,	18,0.000001],
[1,	22,0.000001],
[1,	20,0.000001],
[0,11,0.000001],
[0,	8,0.000001],
[1,	17,0.000001],
[1,	16,0.000001],
[0,	7,0.000001],
[1,	17,0.000001],
[1,	15,0.000001],
[1,	10,0.000001],
[1,	25,0.000001],
[0,	15,0.000001],
[0,	12,0.000001],
[1,	17,0.000001],
[0,	17,0.000001],
[1,	16,0.000001],
[1,	18,0.000001],
[0,	11,0.000001]]

x = np.array(data)[:,1:]
y = np.array(data)[:,0]

X_train,X_test,y_train,y_test= train_test_split(x,y,test_size=0.3)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

from sklearn.metrics import classification_report

# 利用逻辑斯蒂回归自带的评分函数score获得模型在测试集上的准确定结果
print('精确率为：', lr.score(X_test, y_test))

##查看其对应的w
print('the weight of Logistic Regression:\n',lr.coef_)
##查看其对应的w0
print('the intercept(w0) of Logistic Regression:\n',lr.intercept_)
