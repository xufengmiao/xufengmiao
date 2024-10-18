# 最后修改时间：2024.10.11

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt



## 防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei'] 
mpl.rcParams['axes.unicode_minus'] = False 



## 划分特征值与目标值 
    # 6个输入X（p_sub、p_dis、T_sub、T_dis、M_sub、M_dis），2个输出y（K、lambda）
# data = pd.read_csv('6in2out_96_new.csv') # data为结构体 
# data = pd.read_csv('6in2out_96_M30_new.csv') # data为结构体 


data = pd.read_csv('C:\\Users\\Administrator\\Desktop\\CODE_repository\\python_code\\6in2out_96+96.csv') # data为结构体 


X = data[['p_suc','p_dis','T_suc','T_dis','M_suc','M_dis']] # 读取前六列 
# print(X)
y = data[['K_pre']] # 读取第七列
y = np.ravel(y)
# print(y) # Series用于存储一列有索引的数据，相当于只有一列的一维阵列。

## 将数据标准化
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

## 创建一个标准化器对象 
scaler = StandardScaler() 

## 使用标准化器拟合和转换数据集 # 归一化标准化 
X = scaler.fit_transform(X) 

## 将数据集划分为训练集和测试集 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=32, random_state=42) # 80个训练集，16个测试集
    # 3、test_size：样本占比（如果是整数的话就是样本的数量） 
    # 4、random_state：随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数 


# print("-----------------------------------------------------------------------------------------------------------------------")


## SVR.py调用PSO_SVR_3hypers.py 
import PSO_SVR_3hypers 

## 调整超参数后的SVR模型 
SVR_opted_model = PSO_SVR_3hypers.optimize_svm( X_train, y_train, X_test, y_test ) 

## 使用"训练集"，调参后的模型 
SVR_opted_model.fit( X_train, y_train ) 


## "测试集"的预测结果 
    # 即为y_pred_test 
result = SVR_opted_model.predict( X_test ) 
## "训练集"的预测结果 
    # 即为y_pred_train 
result_train = SVR_opted_model.predict( X_train ) 


## 计算：模型在"测试集"上的得分 
score = SVR_opted_model.score( X_test, y_test )             # R^2（决定系数）作为评估指标 
## 计算：模型在"训练集"上的得分 
score_train = SVR_opted_model.score( X_train, y_train ) 


# print("-----------------------------------------------------------------------------------------------------------------------")


## 存储"测试集"的真实标签和预测结果 

y_testRON = [] # 真实标签 
resultRON = [] # 预测结果 

for i in range(len(result)): 

    y_testRON.append(y_test[i]) 
    resultRON.append(result[i]) 


## 存储"训练集"的真实标签和预测结果 

y_trainRON = [] # 真实标签 
resultRON_train = [] # 预测结果 

for i in range(len(result_train)): 

    y_trainRON.append(y_train[i]) 
    resultRON_train.append(result_train[i]) 


# print("-----------------------------------------------------------------------------------------------------------------------") 


## 回归效果可视化 
fig = plt.figure() 
fig.subplots_adjust(hspace=0.4) # hspace：子图间高度内边距，距离单位为子图平均高度的比例（小数） 

# "训练集"的对比
plt.subplot(2, 1, 1) 
plt.plot(np.arange(len(result_train)), y_trainRON, "bo-", label="True value 总传热系数K - train") # 真实值                   # b：blue，o：circle 蓝色圆点
plt.plot(np.arange(len(result_train)), resultRON_train, "ro-", label="Predict value 总传热系数K - train") # 预测值           # r：red，o：circle 红色圆点
plt.title(f"train_SVR---R^2:{score_train}") # 图名 
plt.legend(loc="best") # 位置 

# "测试集"的对比
plt.subplot(2, 1, 2)
plt.plot(np.arange(len(result)), y_testRON, "bo-", label="True value 总传热系数K - test")
plt.plot(np.arange(len(result)), resultRON, "ro-", label="Predict value 总传热系数K - test")
plt.title(f"test_SVR---R^2:{score}")
plt.legend(loc="best")

plt.show()
 
## 可视化标准误差
    # test
RON = np.array(resultRON)
    # train
RONtrain = np.array(resultRON_train)

## 相对误差计算公式
RE_RONtest = abs(y_testRON - RON) / y_testRON
RE_RONtrain = abs(y_trainRON - RONtrain) / y_trainRON
 
plt.figure()
plt.plot(np.arange(len(RE_RONtrain)), RE_RONtrain, "ro-", label="RE value 总传热系数K  - train")
plt.plot(np.arange(len(RE_RONtrain), len(RE_RONtrain) + len(RE_RONtest)), RE_RONtest, "bo-", label="RE value 总传热系数K  - test")
plt.title('Relative Error') # 相对误差
plt.legend(loc="best")
plt.show()
 