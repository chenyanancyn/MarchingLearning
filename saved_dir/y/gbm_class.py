from CART_regression_self import *

def GBDT(data, n_trees):
    trees = {}
    tree = fit(data, max_depth=3)
    trees[0] = tree
    data_features = [d[:-1] for d in data]
    for i in range(1, n_trees):
        pred_label = [d[-1] for d in data]
        # for j, d in enumerate(data):
        #     d[-1] = d[-1] - predict(tree, data_features[j])

        for d_fea, d in zip(data_features, data):
            d[-1] = d[-1] - predict(tree, d_fea)

        tree = fit(data)
        trees[i] = tree

        pred_label = [prdict(tree, d) for d in data_features]
        label = np.concatenate([label, np.array(pred_label)])
        print(np.sum(label.rehshape((i+1, -1)), axis=1))

    return trees


