#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
K-means Clustering
=========================================================

The plots display firstly what a K-means algorithm would yield
using three clusters. It is then shown what the effect of a bad
initialization is on the classification process:
By setting n_init to only 1 (default is 10), the amount of
times that the algorithm will be run with different centroid
seeds is reduced.
The next plot displays what using eight clusters would deliver
and finally the ground truth.

"""

## 0. 显示上述说明
print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

## 1. =====================================================================
#  导入必要的算法包
#  scikit-learn依赖于NumPy， SciPy和 matplotlib。
#  (1)导入python Numpy包(主要用于数组及数组操作，常用矩阵运算)，并以np为别名
#  Numpy含各种子包,想了解，可以：dir(np)
#  (2)导入python matplotlib包(2D图像的绘图工具包,常用语数据可视化)的子包pyplot
#  用于数据的可视化，并以plt为别名
#  了解该子包内容，可以dir(plt)
import numpy as np
import matplotlib.pyplot as plt


#  (3) 导入 mpl_toolkits.mplot3d
#  该工具包提供了一些基本的3D绘图功能，其支持的图表类型包括散点图（scatter）、
#  曲面图（surf）、线图（line）和网格图（mesh）
#  Though the following import is not directly being used, it is required
#  for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D

## 2. 导入scikit-learn内置的 KMeans API接口模块
#   导入scikit-learn内置的datasets 数据集模块
from sklearn.cluster import KMeans
from sklearn import datasets

## 3. 设置随机数发生器种子值，以初始化随机数发生器状态
np.random.seed(5)

## 4. 加载iris数据集 ,得到数据集对象iris：3类别(0,1,2)，150个样本，4个特征
# 含5个成员：data,target,feature_names等
# 获取该数据集对象的data部分，以及类别标号

iris = datasets.load_iris()
X = iris.data
y = iris.target

## 5. 创建数组estimators，每个元素对应一个KMeans对象
estimators = [('k_means_iris_8', KMeans(n_clusters=8)),
              ('k_means_iris_3', KMeans(n_clusters=3)),
              ('k_means_iris_bad_init', KMeans(n_clusters=3, n_init=1,
                                               init='random'))]
# 例如 该数组的首个元素内容为：
# ('k_means_iris_8',KMeans(algorithm='auto', copy_x=True, init='k-means++', 
#  max_iter=300,n_clusters=8, n_init=10, n_jobs=1, precompute_distances='auto',
# random_state=None, tol=0.0001, verbose=0))         
# ======================================================================
# 该元素的具体说明KMeans()
# (1)algorithm=要使用的K-MEANS算法风格,可以是 “auto”, “full” 或者 “elkan”，默认值=”auto”
#   The classical EM-style algorithm is “full”.   
#   The “elkan” variation is more efficient by using the triangle inequality, but currently doesn’t support
#   sparse data. “auto” chooses “elkan” for dense data and “full” for sparse data
# (2)copy_x=为布尔值，默认为True
#   若为真，则直接使用原始样本数据；若为false，则在进行聚类之前，先计算样本中心，对整个数据集去中心，再聚类
#   然后再对聚类后的样本再加上中心.            
# (3)init=初始化的方式,默认值为'k-means++'
# 主要初始化方式为 {‘k-means++’, ‘random’ or an ndarray}
# 其中，方式‘k-means++’ :  以聪明方式初始化聚类中心，以加速收敛；See section Notes in k_init for more details.
# 方式‘random’: 随机选择数据集的k行初始化聚类中心
# 若提供的为数组ndarray，应当形如(n_clusters, n_features) 来作为聚类中心的初始化.                 
# (4)max_iter=默认值 300， K-均值聚类，动态聚类的最大迭代次数
# (5)n_clusters=聚类簇的数目，默认值为8
# (6)n_init=动态聚类初始化的次数，默认值10.K-均值算法将基于不同的初始化状态，产生
#     不同的聚类结果，最后返回其中一个最好的聚类结果。
# (7)n_jobs=默认值为1.该参数主要与n_init参数值相结合.K-均值聚类的每个初始化状态都对应一个聚类结果.
# 可以采用并行计算方式，采用不同CPU处理不同的初始化状态.n_jobs取值为-1，使用所有CPU；
# 若取值为1，则不使用并行计算，这样的话可方便调试；若参数取值小于-1，例如为-2，则并行计算时，所用
# 的CPU数目=总CPU数目+1+n_jobs
# (8)precompute_distances=标记是否需要预先进行有关距离的计算.若要预先计算距离，
#   聚类过程更快，但要耗费更多内存.默认值为'auto',
# 三种取值: {‘auto’, True, False}
# 若为‘auto’ 值，则根据情况自动选择。如果  n_samples  *  n_clusters  >  12  million，则不预先计算。
# True : always precompute distances
# False : never precompute distances
# (9)random_state=随机数发生器的状态，默认值None
# 该参数取值有三种情况
# 若提供的为一个整数值int，此时該值为随机数发生器使用的种子值.
# 若为RandomState instance,random_state is the random number generator
# 若为 None,the random number generator is the RandomState instance used by np.random       
# (10)tol=默认值0.0001,动态聚类过程终止的软条件.容忍的最小误差，当误差小于tol就会退出迭代。
# (11)verbose=用于标记是否需要输出聚类的详细信息。默认值0--不显示详细信息.
# ====================================================================              
## 6. 初始化图形窗口ID序号
fignum = 1

## 7. 生成数组，数组的每个元素对应一个图形的名称
titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']

## 8. estimators数组的元素为二元组，第1部分为name，第2部分为est.
#  以循环方式，顺次实现不同参数下的k-means聚类.该循环体执行三次，得到三个聚类结果.
for name, est in estimators:
    
    # 8-1 创建一个图形敞口(窗口ID号,窗口大小)
    # 参数说明:
    # (1)num=窗口ID号,默认值None，顺序增1
    # (2)figsize 以二元组形式描述的窗口大小英寸(宽度，高度)
    fig = plt.figure(fignum, figsize=(15, 10))
    
    # 8-2 在当前图形窗口中绘制3D图形,
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    
    # 8-3 调用指定参数的K-MEANS对象模块，对数据集X进行聚类
    est.fit(X)
    
    # 8-4 返回该聚类结果中的各个样本的聚类簇标号,从0开始,150个样本
    labels = est.labels_

    #8-5 获取样本的其中3个特征，在3D空间绘制样本散点图,默认为"o"，颜色
    ax.scatter(X[:, 3], X[:, 0], X[:, 2],
               c=labels.astype(np.float), edgecolor='k')

    # 8-6 显示坐标轴刻度
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    
    # 8-7 设置坐标轴名称、图的名称
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title(titles[fignum - 1])
    
    # 8-8 设置坐标轴刻度间隔
    ax.dist = 10
    
    # 8-9 图形窗口ID号增1
    fignum = fignum + 1

## 9. Plot the ground truth
fig = plt.figure(fignum, figsize=(8, 6))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

## 10.二元组形式的数组
for name, label in [('Setosa', 0),
                    ('Versicolour', 1),
                    ('Virginica', 2)]:
    
    # 10-1 在相应类别样本的平均位置，显示该类的名称
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean(),
              X[y == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.1, edgecolor='w', facecolor='w'))
    
## 11. Reorder the labels to have colors matching the cluster results
# 建立颜色映射表，绘制散点图
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor='k')

# 12显示坐标轴刻度
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

# 13设置坐标轴名称、图的名称
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title('Ground Truth')
ax.dist = 5


# 14显示所有图形窗口
fig.show()

## 作业：选择一个最好的k值(在2-8之间)，要求绘制不同k值对应的损失函数值曲线