import pandas as pd 
import numpy as np
import CART_Classifier
import treePlotter_cart
sonar_df=pd.read_csv('sonar.csv',header=None,prefix='V')

print(sonar_df.head())
print(sonar_df.V60.groupby(sonar_df.V60).count())
sonar=sonar_df.values.tolist()
np.random.shuffle(sonar)
train=sonar[:160]
test_data=[d[:-1] for d in sonar[160:]];test_label=[d[-1] for d in sonar[160:]]

tree=CART_Classifier.fit(train)
pred_label=[CART_Classifier.predict(tree,data) for data in test_data]
print(pred_label)

# treePlotter_cart.createPlot(tree)