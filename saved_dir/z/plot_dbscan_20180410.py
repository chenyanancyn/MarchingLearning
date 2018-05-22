# -*- coding: utf-8 -*-
"""
===================================
Demo of DBSCAN clustering algorithm
===================================

Finds core samples of high density and expands clusters from them.

"""
## 0. 显示上述说明
print(__doc__)

##############################################################################
## 1.导入必要的算法包
#  scikit-learn依赖于NumPy和 matplotlib。
#  (1)导入python Numpy包(主要用于数组及数组操作，常用矩阵运算)，并以np为别名
#  Numpy含各种子包,想了解，可以：dir(np)
#  (2)导入python matplotlib包(2D图像的绘图工具包,常用语数据可视化)的子包pyplot
#  用于数据的可视化，并以plt为别名
#  了解该子包内容，可以dir(plt)
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
## 2. (1)导入scikit-learn内置的cluster包的 DBSCAN API接口模块
from sklearn.cluster import DBSCAN

#(2)The sklearn.metrics module includes score functions,
#   performance metrics and pairwise metrics and distance computations.
from sklearn import metrics

#(3)Generate isotropic Gaussian blobs for clustering.
#  scikit-learn includes various random sample generators that can be used to 
#  build artificial datasets of controlled size and complexity.
#  
from sklearn.datasets.samples_generator import make_blobs

#(4) Standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler

##############################################################################
## 3.按照指定的高斯分布(期望向量，分布方差)生成指定数量的样本集 
# (1)三个高斯成分的中心
centers = [[1, 1], [-1, -1], [1, -1]]
#(2)make_blobs provides greater control regarding the centers and standard deviations
#  of each cluster, and is used to demonstrate clustering.
# 有关参数说明
# 
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)

X = StandardScaler().fit_transform(X)

##############################################################################
## 4. Compute DBSCAN
#(1)输入全局输入参数(邻域半径，核心对象邻域内最小样本数)，聚类，返回结果db
db = DBSCAN(eps=0.3, min_samples=10).fit(X)

#(2)生成一个形状与db.labels_一致的bool型的全零数组--实际上数组的元素值都是false
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)

#(3)对上述数组中，对应核心点的样本，赋值为true--这样的话，相当于得到了一个关于核心对象的
#  模板数组，核心对象元素值为true，非核心对象元素值为false。
core_samples_mask[db.core_sample_indices_] = True

#(4)得到聚类后每个样本的标记值:噪声点标记值-1；其它标志值0,1,2分别对应三个输出的聚类簇标号
labels = db.labels_

#(5)统计最终的聚类簇数目-- Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

## 5.以格式控制的字符串形式，输出聚类结果的有关信息
#(1)聚类簇的数目
print('Estimated number of clusters: %d' % n_clusters_)

#(2)有关聚类的评价指标(参见scikit-learn文档的5.21.5 Clustering metrics)
# Homogeneity metric of a cluster labeling given a ground truth.
# A clustering result satisfies homogeneity if all of its clusters contain 
#  only data points which are members of a single class
# 返回值[0,1]. 1.0 stands for perfectly homogeneous labeling
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))

#(3)A clustering result satisfies completeness if all the data points that 
#   are members of a given class are elements of the same cluster.
# 返回值[0,1]. 1.0 stands for perfectly complete labeling
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))

#(4)The V-measure is the harmonic mean between homogeneity and completeness
# 取值[0,1]，越大越好
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))

#(5)The Rand Index computes a similarity measure between two clusterings by 
#   considering all pairs of samples and counting pairs that are assigned 
#   in the same or different clusters in the predicted and true clusterings
#   Similarity score between -1.0 and 1.0. 
#   Random labelings have an ARI close to 0.0. 1.0 stands for perfect match
#   取值越接近1，越好.
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))

#(6)The AMI returns a value of 1 when the two partitions are identical (ie perfectly
#   matched). Random partitions (independent labellings) have an expected AMI around
#   0 on average hence can be negative.
#  取值[0,1]，越大越好

print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))

#(7)轮廓系数,取值[-1,1],取值越大越好
# The best value is 1 and the worst value is -1. 
# Values near 0 indicate overlapping clusters. 
# Negative values generally indicate that a sample has been assigned to the 
# wrong cluster, as a different cluster is more similar

print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

##############################################################################
## 6.Plot result
# Black removed and is used for noise instead.
# (1) {-1,0,1,2}
unique_labels = set(labels)

#(2)基于指定的颜色映射表，生成四种标记值对应的颜色数组。
# 注意：dir(plt.cm) 可查看各种颜色映射表
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

#(3)针对聚类结果样本的不同标记值，以不同颜色绘制这些点
plt.figure(figsize=(15, 15))
for k, col in zip(unique_labels, colors):
    
    # 若为噪声点，以黑色标记
    if k == -1:
        # Black used for noise.对于噪声点，指定相应颜色为黑色
        col = [0, 0, 0, 1]

    # 否则获取相应聚类簇的标记模板
    class_member_mask = (labels == k)

    # 得到每个聚类簇的核心样本点
    xy = X[class_member_mask & core_samples_mask]
    
    # 对该聚类簇的核心样本点，以相同颜色标记，但标记尺寸比较大
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

    # 对该聚类簇的非核心样本点，以较小的尺寸的标记块标记
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], '*', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

#(4)图的名称
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Estimated number of clusters: %d' % n_clusters_)

#(5)显示结果
plt.show()
