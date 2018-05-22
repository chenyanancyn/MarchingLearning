# 降低错误剪枝REP实现过程

# 计算重点：
#  利用tree对test_data进行划分
#  计算剪枝前的错误率：（左右子树错误率加权和）
#  计算剪枝后的错误率：
#            求剪枝后的标签
# 如何剪枝：（塌陷处理）
#   如果测试集为空：无论是不是叶节点直接塌陷
#   如果测试集不为空：从叶节点开始进行塌陷

# 降低错误剪枝REP实现过程
# 1. 判断obj是否是树，是否到达叶子节点
def isTree(obj):
    return isinstance(obj, dict)

# 2.根据树桩位置的决策点得到塌陷后的标签
def get_label(tree):
    label = []
    groups = tree['groups']
    for group in groups:
        for d in group:
            label.append(d[-1])

    label = max(label, key=label.count)

    return label

# 3.根据树桩的决策将data进行划分
def data_split(tree, data):
    left = []
    right = []

    index = tree['index']
    value = tree['value']

    for d in data:
        if d[index] < value:
            left.append(d)
        else:
            right.append(d)
    
    return left, right

# 4.对树进行塌陷处理
def purning_CART(tree, test_data):
    if not test_data:   # 数据集为空
        return get_label(tree)
    # 儿子有子树
    if isTree(tree['left']) or isTree(tree['right']):
        left_test_data, right_test_data = data_split(tree, test_data)
        if isTree(tree['left']):
            tree['left'] = purning_CART(tree['left'], left_test_data)
        if isTree(tree['right']):
            tree['right'] = purning_CART(tree['right'], right_test_data)
    # 儿子无子树
    if not isTree(tree['left']) and not isTree(tree['right']):
        left_test_data, right_test_data = data_split(tree, test_data)
        if len(left_test_data) == 0 or len(right_test_data) == 0:
            return get_label(tree)

        # 剪枝前情况
        left_test_label = [d[-1] for d in left_test_data]
        right_test_label = [d[-1] for d in right_test_data]
        test_label = [d[-1] for d in test_data]

        left_err_radio = 1 - left_test_label.count(tree['left']) / len(left_test_label)
        right_err_radio = 1 - right_test_label.count(tree['right']) / len(right_test_label)
        err = len(left_test_label) / len(test_label) * left_err_radio + len(right_test_label) / len(test_label) * right_err_radio
        
        # 剪枝之后
        purning_label = get_label(tree)
        pruning_err = 1 - test_label.count(purning_label) / len(test_label)

        if pruning_err < err:
            return purning_label
        else:
            return tree
    
    return tree
        