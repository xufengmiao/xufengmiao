#!/usr/bin/env python
# coding: utf-8


# 导入第三方库
from sys import ps1
from turtle import end_fill
import numpy as np  # 科学计算库
from numpy.random import rand  # 随机数
from sklearn.model_selection import train_test_split  # 数据集拆分工具
import matplotlib.pyplot as plt  # 数据可视化库
import pandas as pd  # 数据处理库
import warnings  # 告警库
from sklearn.svm import SVR  # 支持向量机回归模型
from sklearn.model_selection import cross_val_score  # 交叉验证
import sklearn.metrics as metrics  # 导入模型评估工具
import seaborn as sns  # 高级数据可视化库


warnings.filterwarnings(action='ignore')  # 忽略


# 获取最小适应度
def get_fitness(x, y, X_train, y_train, X_test, y_test):
    x = np.min(x)
    y = np.min(y)

    # 定义超参数的空间范围

    # if int(abs(x)) > 0:  # 判断取值
    #     C = int(abs(x) / 100) + 1  # 赋值
    # else:
    #     C = int(abs(x) + 1)  # 赋值

    # if int(abs(y)) > 0:  # 判断取值
    #     gamma = int(abs(y)) / 10  # 赋值
    # else:
    #     gamma = int(abs(y) + 1) / 10  # 赋值

    # if int(abs(y)) > 0:  # 判断取值
    #     epsilon = int(abs(y)) / 10  # 赋值
    # else:
    #     epsilon = int(abs(y) + 1) / 10  # 赋值

    
    C_array = np.linspace(1,100,400) # SVR（支持向量机）的C参数必须为正浮点数
    gamma_array = np.linspace(0.1,10,400)
    epsilon_array = np.linspace(0,1,400)

    for i in range(len(C_array)):
        C = C_array[i]
        gamma = gamma_array[i]
        epsilon = epsilon_array[i]

    svr_model = SVR(kernel='linear', C=C, gamma=gamma, epsilon=epsilon ).fit(X_train, y_train)  # 建模、拟合
    cv_accuracies = cross_val_score(svr_model, X_test, y_test, cv=3, scoring='r2')  # 交叉验证计算r方

    # 使错误率降到最低
    accuracies = cv_accuracies.mean()  # 取交叉验证均值

    # 使错误率降到最低
    fitness_value = (1 - accuracies)  # 错误率 赋值 适应度函数值

    return fitness_value  # 返回数据



##----------------------------解码DNA函数----------------------------
def translateDNA(pop):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    # 友情提示：pop是二维的哦~
    x_pop = pop[:, 1::2]  # 奇数列表示X
    y_pop = pop[:, ::2]  # 偶数列表示y

    # pop:(POP_SIZE,DNA_SIZE)*(DNA_SIZE,1) --> (POP_SIZE,1)     dot() 矩阵乘法计算
    x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]
    # 返回的是x与y染色体的实数值(解码值)
    return x, y


def crossover_and_mutation(pop, CROSSOVER_RATE=0.8):
    new_pop = []  # 定义空列表
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因
        if np.random.rand() < CROSSOVER_RATE:  # 产生一个0~1随机值，如果小于0.8则交叉
            mother = pop[np.random.randint(POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生交叉的点 [low, high)
            child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
        mutation(child)  # 后代变异
        new_pop.append(child)  # 加入下一代种群

    return new_pop  # 返回新一代种群



##----------------------------变异----------------------------
def mutation(child, MUTATION_RATE=0.003):
    if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生一个实数，代表要变异基因的位置
        child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转



##----------------------------选择----------------------------
def select(pop, fitness):
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True)  # 生成索引
    # '''
    # choice方法的参数：
    # numpy.random.choice(a, size=None, replace=True, p=None)
    # #从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
    # #replace:True表示可以取相同数字，False表示不可以取相同数字
    # #数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
    # 也就是说，从种群中根据适应度函数的大小为挑选概率，挑选POP_SIZE个元素作为下一代
    # '''
    return pop[idx]  # 返回数据


##----------------------------最优值数据输出----------------------------
def print_info(x, y, pop):
    fitness = get_fitness(x, y, X_train, y_train, X_test, y_test)  # 调用适用度函数
    # 获取适应度最小的下标索引
    min_fitness_index = np.argmin(fitness)  # 获取最优索引
    x, y = translateDNA(pop)  # 调用解码DNA函数

    return x[min_fitness_index], y[min_fitness_index]  # 返回数据


##----------------------------定义：遗传算法主函数----------------------------
def genetic_algorithm( X_train, y_train, X_test, y_test ):
    pop = np.random.randint( 2, size=( POP_SIZE, DNA_SIZE * 2 ) )  # 生成随机数的范围是[0，2) 矩阵大小为 rows: POP_SIZE cols:DNA_SIZE*2，因为两个自变量
    for _ in range( N_GENERATIONS ):  # 迭代N代
        x, y = translateDNA( pop )
        # 交叉变异，产生新一代种群
        pop = np.array( crossover_and_mutation( pop, CROSSOVER_RATE ) )
        # 计算适应度
        fitness = get_fitness( x, y, X_train, y_train, X_test, y_test )
        # 选择
        pop = select( pop, fitness )  # 选择生成新的种群
    # 打印最后的全局最优信息
    x, y = print_info( x, y, pop )

    return x, y  # 返回数据



##-----------------------------------------------------------------------------------------------------------------



if __name__ == "__main__":
    # 读取数据
    # df = pd.read_excel('C:\\Users\\Administrator\\Desktop\\CODE_repository\\python_code\\6in1out_K_96.xlsx')
    df = pd.read_excel('C:\\Users\\Administrator\\Desktop\\CODE_repository\\python_code\\7in1out_origin.xlsx')

    # 用Pandas工具查看数据
    print(df.head())

    # 查看数据集摘要
    print(df.info())

    # 数据描述性统计分析
    print(df.describe())

    # y变量分布直方图
    fig = plt.figure(figsize=(8, 5))  # 设置画布大小
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    # data_tmp = df['K_pre']  # 过滤出K_pre变量的样本
    data_tmp = df['y']  # 过滤出K_pre变量的样本
    # 绘制直方图  
        # bins：控制直方图中的区间个数 
        # auto为自动填充个数  
        # color：指定柱子的填充色
    plt.hist(data_tmp, bins='auto', color='g')
    # plt.xlabel('K_pre')  # 设置x轴名称
    plt.xlabel('y')  # 设置x轴名称
    plt.ylabel('数量')  # 设置y轴名称
    # plt.title('K_pre变量分布直方图')  # 设置标题名称
    plt.title('y变量分布直方图')  # 设置标题名称
    plt.show()  # 展示图片

    # 数据的相关性分析
    sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)  # 绘制热力图
    plt.title('相关性分析热力图')  # 设置标题名称
    plt.show()  # 展示图片



##-----------------------------------------------------------------------------------------------------------------



    # 提取特征变量和标签变量
    # y = df['K_pre']
    # X = df.drop('K_pre', axis=1)
    y = df['y']
    X = df.drop('y', axis=1)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=16, random_state=42)

    # 参数初始化
    # 染色体长度
    DNA_SIZE = 24
    # 种群规模
    POP_SIZE = 200
    # 交叉率 一般在0.6~0.9之间
    CROSSOVER_RATE = 0.8
    # 变异率 一般小于0.1
    MUTATION_RATE = 0.005
    # 种群迭代次数
    N_GENERATIONS = 100
    # 自变量x和y的范围
    X_BOUND = [-3, 3]
    Y_BOUND = [-3, 3]



##-----------------------------------------------------------------------------------------------------------------



    # 调用遗传算法主函数
    arg_x, arg_y = genetic_algorithm(X_train, y_train, X_test, y_test)

    if int(abs(arg_x)) > 0:  # 判断
        best_C = int(abs(arg_x) / 100) + 1  # 赋值
    else:
        best_C = int(abs(arg_x) + 1)  # 赋值

    if int(abs(arg_y)) > 0:  # 判断
        best_gamma = int(abs(arg_y)) / 10  # 赋值
    else:
        best_gamma = int(abs(arg_y) + 1) / 10  # 赋值

    if int(abs(arg_y)) > 0:  # 判断
        best_epsilon = int(abs(arg_y)) / 10  # 赋值
    else:
        best_epsilon = int(abs(arg_y) + 1) / 10  # 赋值



##-----------------------------------------------------------------------------------------------------------------



    print('----------------GA遗传算法优化支持向量机回归模型-最优结果展示-----------------')
    print("The best C is " + str(abs(best_C)))
    print("The best gamma is " + str(abs(best_gamma)))
    print("The best epsilon is " + str(abs(best_epsilon)))



    print('----------------应用优化后的最优参数值构建支持向量机回归模型-----------------')
    # 应用优化后的最优参数值构建支持向量机回归模型
    svr_model = SVR( kernel='linear', C=best_C, gamma=best_gamma, epsilon=best_epsilon )  # 建模
    svr_model.fit( X_train, y_train )  # 拟合
    y_pred = svr_model.predict( X_test )  # 预测



    print('----------------模型评估-----------------')
    # 模型评估
    print('**************************输出测试集的模型评估指标结果*******************************')

    print('SVR回归模型 - 最优参数 - R^2值：{}'.format(round(metrics.r2_score(y_test, y_pred), 4)))
    print('SVR回归模型 - 最优参数 - mse均方误差：{}'.format(round(metrics.mean_squared_error(y_test, y_pred), 4)))
    print('SVR回归模型 - 最优参数 - evc可解释方差值：{}'.format(round(metrics.explained_variance_score(y_test, y_pred), 4)))
    print('SVR回归模型 - 最优参数 - mape平均绝对误差：{}'.format(round(metrics.mean_absolute_error(y_test, y_pred), 4)))



    # 真实值与预测值比对图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.plot(range(len(y_test)), y_test, color="blue", linewidth=1.5, linestyle="-")  # 绘制折线图
    plt.plot(range(len(y_pred)), y_pred, color="red", linewidth=1.5, linestyle="-.")  # 绘制折线图
    plt.legend(['真实值', '预测值'])  # 设置图例
    plt.title("GA优化SVR模型 - 真实值与预测值比对图")  # 设置标题名称
    plt.show()  # 显示图片


