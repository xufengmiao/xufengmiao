import numpy as np
from sklearn.svm import SVR
from matplotlib import pyplot as plt
import pyswarms as ps



np.random.seed(42) # 使用NumPy库中的随机数生成器时设置种子值为42 把seed()中的参数比喻成“堆”；eg. seed(42)：表示第42堆种子。
 
 
## 定义适应度函数
def fitness_function(params, X, y, xt, yt): # 训练集——X特征矩阵，y目标变量；测试集——xt，yt
    C, gamma = params
    svm_model = SVR(kernel='rbf', gamma=gamma, C=C) # 创建SVM模型：svm_model
    svm_model.fit(X, y) # 使用svm_model.fit()函数在训练集上训练SVM模型
    y_pred = svm_model.predict(xt)
    mse = np.mean((yt - y_pred) ** 2)
    return mse
 
 
## 定义优化函数
def optimize_svm(X, y, xt, yt, n_particles=100, n_iterations=100):
    def _fitness_function(params):
        fitness_values = []
        for p in params: # 遍历循环3个超参数
            fitness = fitness_function(p, X, y, xt, yt) # 调用定义的子函数fitness_function，返回mse值
            fitness_values.append(fitness) # append函数会在数组后加上相应的元素
        return fitness_values # 3个mse值
 
    ## 参数边界
    bounds = (np.array([0.1, 0.01]), np.array([50.0, 10.0]))
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

    ## 运行优化器进行参数优化 
        # PSO超参数调优采用的是 pyswarm 包中的 GlobalBestPSO()
        # GlobalBestPSO()的优化方法：objective_func (function)：自适应函数或损失函数；iters (int)：迭代次数；kwargs (dict) ：目标函数的参数
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=2, bounds=bounds, options=options) 
        # n_particles：整数，粒子群中的粒子数量 ；
        # dimension：整数，空间维度，或特征维度 ；
        # options：系数，字典形式 {‘c1’=2, ‘c2’=2, ‘w’=0.3} ；
        # bounds：数组，可选，行数为 2，第一行是边界的最小值，第二行是边界的最大值
    best_cost, best_params = optimizer.optimize(_fitness_function, iters=n_iterations)

    cost_history = np.zeros(n_iterations)
    ## 在每次迭代前，保存代价值
    for i, cost in enumerate(optimizer.cost_history):
        cost_history[i] = cost
 
 
    ## 根据优化结果建立最终的SVM模型
    C, gamma = best_params
    svm_model = SVR(kernel='rbf', gamma=gamma, C=C)
    svm_model.fit(X, y)
    print('最优参数：', best_params)
    
    ## 绘制代价值变化曲线
    plt.plot(range(n_iterations), cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function Evolution')
    plt.show()
 
    return svm_model