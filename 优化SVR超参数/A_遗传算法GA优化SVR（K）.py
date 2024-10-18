# 最近一次修改时间：2024.10.16

import pandas as pd 
import math 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.datasets import make_regression 
from sklearn.model_selection import train_test_split 
import numpy as np 
from sklearn.metrics import mean_absolute_percentage_error, r2_score 
import matplotlib.pyplot as plt 
import matplotlib as mpl 

## 防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei'] 
mpl.rcParams['axes.unicode_minus'] = False 
# plt.rcParams['font.family'] = 'Times New Roman'


# df = pd.read_excel("C:\\Users\\Administrator\\Desktop\\CODE_repository\\python_code\\6in1out_K_96.xlsx")
# ## 分割数据集（6输入、1输出）
# X = df.drop(['K_pre'], axis = 1)
# y = df['K_pre']


data = pd.read_csv('C:\\Users\\Administrator\\Desktop\\CODE_repository\\python_code\\6in2out_96+96.csv') # data为结构体 


X = data[['p_suc','p_dis','T_suc','T_dis','M_suc','M_dis']] # 读取前六列 
# print(X)
y = data[['K_pre']] # 读取第七列
y = np.ravel(y)
# print(y) # Series用于存储一列有索引的数据，相当于只有一列的一维阵列。


## 将数据标准化
from sklearn.preprocessing import  StandardScaler 
from sklearn.preprocessing import  MinMaxScaler
from sklearn.model_selection import train_test_split

# 创建一个标准化器对象 
scaler = StandardScaler() 
# 输入参数的归一化标准化 
X = scaler.fit_transform(X) 


from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# 划分：训练集 + 测试集
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3, random_state=42 ) 
# df.head()

# 类型转换为数组（为后面使用r2_score做准备）
y_test = np.array( y_test )


##--------------------------------------------------------------------------------------------------


from sklearn.svm import SVR 
from hyperopt import fmin, tpe, hp 
from sko.GA import GA


## 定义超参数空间 
# 后面SVR传递三个超参数要求类型为float，于是先定义三个数组，后面随机取出
C_boundary = np.arange(0.1, 500, 0.1)
gamma_boundary = np.arange(0.001, 10, 0.001)
epsilon_boundary = np.arange(0.01, 1, 0.01)

for i in range(0, len(C_boundary)): # TypeError: 'int' object is not iterable
    C = C_boundary[i]

for i in range(0, len(gamma_boundary)):
    gamma = gamma_boundary[i]

for i in range(0, len(epsilon_boundary)):
    epsilon = epsilon_boundary[i]

# hyperparameter_space_SVR = {

#            'kernel': hp.choice('kernel', ['rbf']), 
#            'C': hp.uniform('C', 1, 500 ), 
#            'gamma': hp.uniform('gamma', 0.001, 10), # gamma过大，可能导致 
#            'epsilon': hp.uniform('epsilon', 0.01, 1) 

#                       }



## 初始化计数器
count = 0

 

## 定义优化目标函数 
def optfunc(hypers): 
    
    C, gamma, epsilon = hypers
    global count    
    count += 1    


    # 打印
    print(f"\nIteration {count}: Hyperparameters - {hypers}")

    # 创建SVR分类器，传递超参数    
    # model = SVR( **args ) # 报错：TypeError: sklearn.svm._classes.SVR() argument after ** must be a mapping, not numpy.ndarray
    model = SVR( C=C, gamma=gamma, epsilon=epsilon, kernel='rbf' ) 

    # 训练模型    
    model.fit( X_train, y_train ) 

    # 预测测试集    
    prediction = model.predict( X_test ) 
    prediction = np.array( prediction )

    # 计算准确率    
    Test_accuracy_score = r2_score( y_test, prediction ) # accuracy_score的输入要求，都是一维矩阵
        # 这里将accuracy_score改换为r2_score：
          # 回归问题用r2_score
          # 分类问题用accuracy_score

        # Python numpy ，shape为(x,) 和 shape为(x,1)的区别：
        #  (x,)意思是一维数组，数组中有2个元素
        #  (x,1)意思是一个x维数组，每行有1个元素


    # 打印
    print(f'Test accuracy (R^2): {Test_accuracy_score}') 


    # 由于 ga 函数默认是最小化目标函数，所以返回"负准确率"作为目标 
    return -Test_accuracy_score 

# C、gamma、epsilon的上下界范围
bound_low = [1, 0.001, 0.01 ]
bound_upper = [500, 10, 1 ]

# 使用ga算法进行超参数优化

best_group = GA( func = optfunc, n_dim = 3, size_pop = 100, max_iter = 100, lb = bound_low, ub = bound_upper ) # 输出 best_x ， best_y 
best_hypers , best_score = best_group.run() # best_hypers数组，best_score数值

# 输出最佳超参数组合
print( '\n Best hyperparameters: ', best_hypers )



##--------------------------------------------------------------------------------------------------



## 利用找到的最优超参数组合，创建最终的SVR模型
final_model = SVR(    
    
                      kernel='rbf',                    # 核函数类型，这里选择径向基函数（RBF）
                      C=best_hypers[0],                # C    
                      gamma=best_hypers[1],            # gamma
                      epsilon=best_hypers[2],          # epsilon
                      cache_size=5000,                 # 缓存大小，单位为MB
                      max_iter=200

                  )

## 使用"训练集"进行模型训练
final_model.fit( X_train, y_train )

## 预测
    # 使用模型在"训练集"上进行预测
y_pred_train = final_model.predict( X_train )
    # 使用模型在"测试集"上进行预测
y_pred = final_model.predict( X_test )

## 计算得分
    # 计算：模型在"测试集"上的得分 
score = final_model.score( X_test, y_test )             # R^2（决定系数）作为评估指标 
    # 计算：模型在"训练集"上的得分 
score_train = final_model.score( X_train, y_train ) 


# from sklearn.metrics import classification_report # 用于分类
from sklearn.metrics import explained_variance_score # 用于回归

# 输出模型报告， 查看评价指标
print( explained_variance_score( y_test, y_pred ) )


## 将各结果汇总储存
    # test集
repository_y_test = [] 
repository_y_pred = [] 
for i in range(len(y_pred)): 
    repository_y_test.append(y_test[i]) 
    repository_y_pred.append(y_pred[i]) 

    # train集
repository_y_train = [] 
repository_y_pred_train = [] 
for i in range(len(y_pred_train)): 
    repository_y_train.append(y_train[i]) 
    repository_y_pred_train.append(y_pred_train[i]) 

# list用len()
# array用.shape
# print(len(repository_y_train))
# print(len(repository_y_pred_train))
# print(len(repository_y_test))
# print(len(repository_y_pred))
 
 

# print("-----------------------------------------------------------------------------------------------------------------------") 
 

 
## 回归效果 可视化 
fig = plt.figure() 
fig.subplots_adjust(hspace=0.4) # hspace：子图间高度内边距，距离单位为子图平均高度的比例（小数） 
 
# train的对比：
#  y_train（直接获得）、y_pred_train（输入为X_train）
plt.subplot(2, 1, 1) 
plt.plot(np.arange(len(y_pred_train)), repository_y_train, "bo-", label="True value 总传热系数K - train") # 真实值                   # b：blue，o：circle 蓝色圆点
plt.plot(np.arange(len(y_pred_train)), repository_y_pred_train, "ro-", label="Predict value 总传热系数K - train") # 预测值           # r：red，o：circle 红色圆点
plt.title(f"train_SVR---R^2:{score_train}") # 图名 
plt.legend(loc="best") # 位置 

# test的对比： 
# y_test（直接获得）、y_pred（输入为X_test）
plt.subplot(2, 1, 2) 
plt.plot(np.arange(len(y_pred)), repository_y_test, "bo-", label="True value 总传热系数K - test") 
plt.plot(np.arange(len(y_pred)), repository_y_pred, "ro-", label="Predict value 总传热系数K - test") 
plt.title(f"test_SVR---R^2:{score}") 
plt.legend(loc="best") 

plt.show() 



##--------------------------------------------------------------------------------------------------------------------------------



## 标准误差 可视化 

# 预测结果 与 相对误差

    # test 
RON = np.array(repository_y_pred) # （输入为X_test）
RE_RONtest = abs(repository_y_test - RON) / repository_y_test 

    # train 
RONtrain = np.array(repository_y_pred_train) # （输入为X_train）
RE_RONtrain = abs(repository_y_train - RONtrain) / repository_y_train 

 
plt.figure()
plt.plot(np.arange(len(RE_RONtrain)), RE_RONtrain, "ro-", label="RE value 总传热系数K  - train")
plt.plot(np.arange(len(RE_RONtrain), len(RE_RONtrain) + len(RE_RONtest)), RE_RONtest, "bo-", label="RE value 总传热系数K  - test")
plt.title('Relative Error') 
plt.legend(loc="best")

plt.show()