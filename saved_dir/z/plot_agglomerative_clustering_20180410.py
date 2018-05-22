"""
Agglomerative clustering  without structure
===================================================

This example shows the effect of imposing a connectivity graph to capture
local structure in the data. The graph is simply the graph of 20 nearest
neighbors.

Two consequences of imposing a connectivity can be seen. First clustering
with a connectivity matrix is much faster.

Second, when using a connectivity matrix, average and complete linkage are
unstable and tend to create a few clusters that grow very quickly. Indeed,
average and complete linkage fight this percolation behavior by considering all
the distances between two clusters when merging them. The connectivity
graph breaks this mechanism. This effect is more pronounced for very
sparse graphs (try decreasing the number of neighbors in
kneighbors_graph) and with complete linkage. In particular, having a very
small number of neighbors in the graph, imposes a geometry that is
close to that of single linkage, which is well known to have this
percolation instability.
"""
# Authors: Gael Varoquaux, Nelle Varoquaux
# License: BSD 3 clause


## 1.  导入必要的算法包
#  (1)time 时间模块
#  (2)scikit-learn依赖于NumPy和 matplotlib。
#  导入python Numpy包(主要用于数组及数组操作，常用矩阵运算)，并以np为别名
#  Numpy含各种子包,想了解，可以：dir(np)
#  (3)导入python matplotlib包(2D图像的绘图工具包,常用语数据可视化)的子包pyplot
#  用于数据的可视化，并以plt为别名
#  了解该子包内容，可以dir(plt)

import time
import matplotlib.pyplot as plt
import numpy as np

## 2.导入scikit-learn内置的 AgglomerativeClustering API接口模块
from sklearn.cluster import AgglomerativeClustering


## 3.  利用随机数发生器构造1500个样本.

# 3-1 样本数
n_samples = 1500

# 3-2 初始化随机数发生器状态，种子值0
np.random.seed(0)

# 3-3 产生1*1500个均匀分布随机数，取值范围：1.5pi*[1,4]
t = 1.5 * np.pi * (1 + 3 * np.random.rand(1, n_samples))

# 3-4 基于上述随机数生成新的数组x,y
x = t * np.cos(t)
y = t * np.sin(t)

# 3-5 X: 2*1500
X = np.concatenate((x, y))
X += .7 * np.random.randn(2, n_samples)

# 3-6 转置： 1500*2
X = X.T


## 4 分层聚类，聚类数目=30,10,3
for n_clusters in (30, 10, 3, 2):
    
    # 4-1 生成一个指定大小的图形窗口
    plt.figure(figsize=(10, 4))
    
    # 4-2 分别按照指定的距离度量聚类
    for index, linkage in enumerate(('average', 'complete', 'ward')):
        
        # 4-3 图像窗口1*3
        plt.subplot(1, 3, index + 1)
        
        # 4-4 聚集式分层聚类，创建聚类实例
        model = AgglomerativeClustering(linkage=linkage,
                                        n_clusters=n_clusters)
        # (1)ward--minimizes the variance of the clusters being merged.
        # (2)average-- uses the average of the distances of each observation of
        # the two sets.
        # (3)complete or maximum linkage uses the maximum distances between 
        # all observations of the two sets.
        
        # 4-5 获取当前时间
        t0 = time.time()
        
        # 4-6 生成聚类结果
        model.fit(X)
        
        # 4-7 统计时间
        elapsed_time = time.time() - t0
        
        # 4-8 绘制散点图，按照聚类簇的标号，以相应颜色标记
        plt.scatter(X[:, 0], X[:, 1], c=model.labels_,
                    cmap=plt.cm.spectral)
        
        # 4-9 图的名称，坐标轴等刻度，不显示坐标轴
        plt.title('linkage=%s (time %.2fs)' % (linkage, elapsed_time),
                  fontdict=dict(verticalalignment='top'))
        plt.axis('equal')
        plt.axis('off')
 
        # 4-10 各个子窗口布局设置
        plt.subplots_adjust(bottom=0, top=.89, wspace=0,
                            left=0, right=1)
        # (1)left  -- the left side of the subplots of the figure
        # (2)right -- the right side of the subplots of the figure
        # (3)bottom -- the bottom of the subplots of the figure
        # (4)top -- the top of the subplots of the figure
        # (5)wspace -- the amount of width reserved for blank space between subplots,
        #    expressed as a fraction of the average axis width
        # (6)hspace --the amount of height reserved for white space between subplots,
        #    expressed as a fraction of the average axis height
    
        
        # 4-11 图形窗口的名称
        plt.suptitle('n_cluster=%i' %(n_clusters), size=17)
        

## 5 显示所有图形窗口
plt.show()
