import pandas as pd
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=30, oob_score=True)

train_df=pd.read_csv('sonar.csv',header=None)


label_df=train_df.pop(60).map({'M':0,'R':1})
print(train_df.head())
train=train_df.values
label=label_df.values

# print(label[:10])

# rf.fit(train,label)

# print(rf.oob_score_)


from sklearn.utils import shuffle
train,label=shuffle(train,label,random_state=5)

train_fea=train[:160];train_label=label[:160]
test_fea=train[160:];test_label=label[160:]

rf.fit(train_fea,train_label)
# print(rf.oob_score_)
pred_label=rf.predict(test_fea)
pred_prob_label=rf.predict_proba(test_fea)
print(pred_prob_label)
print(rf.predict(test_fea))

from sklearn.metrics import roc_auc_score
print(roc_auc_score(test_label,rf.predict_proba(test_fea)[:,1]))
# print(pred_label)
# # print('real labels')
# print(test_label)

# # print([pred_label==test_label].count(True)/len(test_label))

# from sklearn.metrics import accuracy_score
# print(accuracy_score(test_label,pred_label))

####第一种方法，利用交叉验证方法对变量逐一进行调整
from sklearn.model_selection import cross_val_score

def rmse_cv(model,X_train,y):
    rmse = cross_val_score(model, X_train, y, 
                                    scoring='accuracy',#"roc_auc",
                                   cv = 6)
    return rmse

NE=range(1,61,5)
MSE=range(2,10,2)
score=rmse_cv(RandomForestClassifier(n_estimators=10),train,label)
print(score)

score=[rmse_cv(RandomForestClassifier(n_estimators=20,min_samples_split=mse)\
	,train,label).mean() for mse in MSE]
print(score)
import matplotlib.pyplot as plt
plt.plot(MSE,score)
plt.show()