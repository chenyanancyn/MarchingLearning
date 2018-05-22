#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
===============
GMM covariances
===============

Demonstration of several covariances types for Gaussian mixture models.

See :ref:`gmm` for more information on the estimator.

Although GMM are often used for clustering, we can compare the obtained
clusters with the actual classes from the dataset. We initialize the means
of the Gaussians with the means of the classes from the training set to make
this comparison valid.

We plot predicted labels on both training and held out test data using a
variety of GMM covariance types on the iris dataset.
We compare GMMs with spherical, diagonal, full, and tied covariance
matrices in increasing order of performance. Although one would
expect full covariance to perform best in general, it is prone to
overfitting on small datasets and does not generalize well to held out
test data.

On the plots, train data is shown as dots, while test data is shown as
crosses. The iris dataset is four-dimensional. Only the first two
dimensions are shown here, and thus some points are separated in other
dimensions.

"""

# Author: Ron Weiss <ronweiss@gmail.com>, Gael Varoquaux
# Modified by Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

## 0. 显示上述说明
print(__doc__)
# 多元正太分布的协方差为对角矩阵  *******
#-------------------------------------------------------------------------
## 1.导入必要的算法包
#  scikit-learn依赖于NumPy和 matplotlib。
#  (1)导入python Numpy包(主要用于数组及数组操作，常用矩阵运算)，并以np为别名
#  Numpy含各种子包,想了解，dir(np)
#  (2)导入python matplotlib包(2D图像的绘图工具包,常用语数据可视化)及其子包pyplot
#  用于数据的可视化，并分别以mpl，plt为别名
#  了解该子包内容，dir(plt)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#-------------------------------------------------------------------------
## 2. 导入scikit-learn内置的 GaussianMixture API接口模块
#    导入scikit-learn内置的datasets 数据集模块
#   导入scikit-learn内置的 StratifiedKFold API接口模块

from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

#  见文档 5.23 sklearn.model_selection: Model Selection
#-------------------------------------------------------------------------
## 3. 定义颜色数组，用于可视化
colors = ['navy', 'turquoise', 'darkorange']

#-------------------------------------------------------------------------
## 4. 定义函数
def make_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        
        #(1)返回协方差矩阵的本征值(降序)、本征列向量
        v, w = np.linalg.eigh(covariances)
        
        #(2)获取第1大本征值对应的本征向量的单位化向量-第一主轴
        u = w[0] / np.linalg.norm(w[0])
        
        #(3)获取这个主轴方向角-弧度
        angle = np.arctan2(u[1], u[0])
        
        #(4)转成度
        angle = 180 * angle / np.pi  # convert to degrees
        
        #(5)获取两个主轴方向样本分布标准差的2*sqrt(2)倍值
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        
        #(6)绘制椭圆(椭圆中心，长轴，短轴，角度，颜色)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        
#-------------------------------------------------------------------------
## 5. 加载iris数据集 ,得到数据集对象iris：3类别(0,1,2)，150个样本，4个特征
# 含5个成员：data,target,feature_names等
# 获取该数据集对象的data部分，以及类别标号
iris = datasets.load_iris()

#-------------------------------------------------------------------------
## 6. 初始化数据集分层类别实例
#  Break up the dataset into non-overlapping training (75%) and testing
# (25%) sets.
skf = StratifiedKFold(n_splits=4)

## 7.基于该划分实例，将给定样本集划分为训练集与测试集，返回对应样本序号 Only take the first fold.
train_index, test_index = next(iter(skf.split(iris.data, iris.target)))

##8. 提取训练集、测试集
X_train = iris.data[train_index]
y_train = iris.target[train_index]

X_test = iris.data[test_index]
y_test = iris.target[test_index]

## 9 获取类别数:首先生成一个包含类别标签的集合，集合元素数目就是类别数
n_classes = len(np.unique(y_train))

## 10.创建字典类实例对象，得到数组estimators
#    Try GMMs using different types of covariances.
#   数组中每个元素对应一个字典类实例,它是一个二元组
#   该元素的第1个成分为cov_type值，第2个成分为1个GaussianMixture实例
estimators = dict((cov_type, GaussianMixture(n_components=n_classes,
                   covariance_type=cov_type, max_iter=20, random_state=0))
                  for cov_type in ['spherical', 'diag', 'tied', 'full'])
# GaussianMixture有关参数设置
#(1)n_components--单高斯成分的数目,或者聚类数目,
#(2)covariance_type--各个高斯成分的协方差矩阵的类型，四种类型
# 'full' -- each component has its own general covariance matrix
# 'tied' -- all components share the same general covariance matrix
# 'diag' -- each component has its own diagonal covariance matrix 
# 'spherical' --each component has its own single variance     
#(3)max_iter=20 The number of EM iterations to perform 
#(4)random_state 随时数发生器的状态设置   
#(5)init_params : {‘kmeans’, ‘random’}, defaults to ‘kmeans’.
#   The method used to initialize the weights, the means and the precisions. 
#   Must be one of:
#   'kmeans' : responsibilities are initialized using kmeans.
#   'random' : responsibilities are initialized randomly.

## 11 获取实例对象数组的元素数目
n_estimators = len(estimators)

## 12. 创建1个指定大小的图形窗口10*10
plt.figure(figsize=(6 * n_estimators // 2, 12))

## 13. 使该窗口内各个子窗口按照指定要求布局设置
plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                    left=.01, right=.99)
        # (1)left  -- the left side of the subplots of the figure
        # (2)right -- the right side of the subplots of the figure
        # (3)bottom -- the bottom of the subplots of the figure
        # (4)top -- the top of the subplots of the figure
        # (5)wspace -- the amount of width reserved for blank space between subplots,
        #    expressed as a fraction of the average axis width
        # (6)hspace --the amount of height reserved for white space between subplots,
        #    expressed as a fraction of the average axis height


## 14. 循环，生成不同风格协方差矩阵的高斯聚类结果.
for index, (name, estimator) in enumerate(estimators.items()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    
    ## 14-1 初始化各个高斯成分的中心
    estimator.means_init = np.array([X_train[y_train == i].mean(axis=0)
                                    for i in range(n_classes)])

    ## 14-2  Train the other parameters using the EM algorithm.
    estimator.fit(X_train)

    ##14-3 获取相应子窗口位置及坐标轴
    h = plt.subplot(2, n_estimators // 2, index + 1)
    
    ## 14-4 在该子窗口内绘制各个高斯成分对应的置信区间图
    make_ellipses(estimator, h)

    ## 14-5 绘制真实类别样本各个样本散点图
    for n, color in enumerate(colors):
        data = iris.data[iris.target == n]
        plt.scatter(data[:, 0], data[:, 1], s=0.8, color=color,
                    label=iris.target_names[n])
        
    ## 14-6 绘制测试集散点图 Plot the test data with crosses
    for n, color in enumerate(colors):
        data = X_test[y_test == n]
        plt.scatter(data[:, 0], data[:, 1], marker='x', color=color)

    ## 14-7 对训练样本所在聚类簇预测；估计预测正确率；显示字符串
    y_train_pred = estimator.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    plt.text(0.05, 0.95, 'Train accuracy: %.1f' % train_accuracy,
             transform=h.transAxes)
    
    ## 14-8 对测试样本所在聚类簇预测；估计预测正确率；显示字符串
    y_test_pred = estimator.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    plt.text(0.05, 0.90, 'Test accuracy: %.1f' % test_accuracy,
             transform=h.transAxes)

    ## 14-9显示坐标轴刻度、图的名称
    plt.xticks(())
    plt.yticks(())
    plt.title(name)
    
    ## 14-10.显示图例,注意这句话位置的不同，结果也不一样。
    plt.legend(scatterpoints=1, loc='upper right', prop=dict(size=12))

## 15. 显示所有图形
plt.show()
