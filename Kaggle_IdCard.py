'''
Kaggle：构建信用卡反欺诈预测模型
1、数据准备和探索
2、特征工程（包含特选择）
3、建模：训练和评估模型（包含训练集样本不平衡、多指标评估）
4、模型网格搜索调优
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import missingno as msno
import itertools
sns.set_style('whitegrid')
plt.style.use('ggplot')
np.set_printoptions(precision=2)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.float_format', lambda x: '%.4f' % x)
'''正负样本占比的可视化图'''
def draw_all_class_radio(data):
    count_classes = pd.value_counts(data['Class'], sort=True).sort_index()
    N = np.sum(count_classes.values)
    non_fraud, fraud = count_classes.values
    print('正常交易：{}笔，占比为：{:.2%}'.format(non_fraud, non_fraud / N))
    print('欺诈交易：{}笔，占比为：{:.2%}'.format(fraud, fraud / N))
    plt.bar(x=['正常交易', '欺诈交易'], height=count_classes.values)
    plt.title('正负样本占比')
    plt.show()
'''数据缺省值检测的可视化图'''
def draw_has_none(data):
    msno.matrix(data)
    plt.show()
'''单一特征在所有class分布的可视化图'''
def draw_one_feature_class_dist(data, feature):
    f, (ax1, ax2) = plt.subplots(2, 1, sharex='row', figsize=(12, 6))
    bins = 50
    ax1.hist(data[feature][data.Class == 1], bins=bins)
    ax1.set_title('欺诈交易直方图', fontsize=15)
    ax1.set_ylabel('欺诈交易次数', fontsize=12)
    ax2.hist(data[feature][data.Class == 0], bins=bins)
    ax2.set_title('正常交易直方图', fontsize=15)
    ax2.set_ylabel('正常交易次数', fontsize=12)
    plt.xlabel('金额(单位:$)', fontsize=12)
    plt.yscale('log')
    plt.show()
'''各个特征在所有class分布的可视化图'''
def draw_all_feature_class_dist(data):
    v_feature = list(data.columns.drop(['Class']))
    plt.figure(figsize=(12, 28 * 4))
    gs = gridspec.GridSpec(len(v_feature), 1)  # 调整非对称子图
    for i, feature in enumerate(v_feature):
        ax = plt.subplot(gs[i])
        sns.distplot(data[feature][data['Class'] == 1], bins=50, color='red', label='欺诈')
        sns.distplot(data[feature][data['Class'] == 0], bins=50, color='green', label='正常')
        ax.set_xlabel('')
        ax.set_title('{}直方图'.format(feature))
        plt.legend()
    plt.savefig('./data/各个特征与class的关系.png', transparent=False, bbox_inches='tight')
'''彼此特征相关性系数矩阵的可视化图'''
def draw_each_feature_Pearson(data):
    Xfraud = data.loc[data['Class'] == 1]  # 正常交易类数据样本
    XnonFraud = data.loc[data['Class'] == 0]  # 欺诈交易类数据样本
    correlationNonFraud = XnonFraud.loc[:, data.columns != 'Class'].corr()  # 正常交易类相关系数
    correlationFraud = Xfraud.loc[:, data.columns != 'Class'].corr()  # 欺诈交易类相关系数
    mask = np.zeros_like(correlationNonFraud)
    indices = np.triu_indices_from(correlationNonFraud)  # 求上三角矩阵元素坐标
    mask[indices] = True
    grid_kws = {'width_ratios': (.9, .9, .05), 'wspace': 0.2}
    f, (ax1, ax2, cbar_ax) = plt.subplots(1, 3, gridspec_kw=grid_kws, figsize=(14, 9))
    cmap = sns.diverging_palette(220, 8, as_cmap=True)
    ax1 = sns.heatmap(correlationNonFraud, ax=ax1, vmin=-1, vmax=1, cmap=cmap, square=False, linewidths=0.5, mask=mask, cbar=False)
    ax1.set_xticklabels(ax1.get_xticklabels(), size=16)
    ax1.set_yticklabels(ax1.get_yticklabels(), size=16)
    ax1.set_title('正常交易', size=20)
    ax2 = sns.heatmap(correlationFraud, vmin=-1, vmax=1, cmap=cmap, ax=ax2, square=False, linewidths=0.5, mask=mask,
                      yticklabels=False, cbar_ax=cbar_ax, cbar_kws={'orientation': 'vertical', 'ticks': [-1, -0.5, 0, 0.5, 1]})
    ax2.set_xticklabels(ax2.get_xticklabels(), size=16)
    ax2.set_title('欺诈交易', size=20)
    cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(), size=14)
    plt.show()
'''高维特征空间映射到低维可视化'''
def draw_high_to_low_feature(X, y):
    from sklearn.decomposition import PCA
    new_X = PCA(n_components=2).fit_transform(X)
    plt.scatter(new_X[y[y == 0].index, 0], new_X[y[y == 0].index, 1], color='#1E90FF', marker='o', label='正常交易', s=100)
    plt.scatter(new_X[y[y == 1].index, 0], new_X[y[y == 1].index, 1], color='#FF4500', marker='o', label='欺诈交易', s=100)
    plt.legend()
    plt.show()
'''基于GDBT的特征选择重要度可视化'''
def draw_GDBT_feature_importances(select_feature, select_importances):
    plt.bar(select_feature, select_importances, color='lightblue', align='center', label='特征重要度')
    plt.step(select_feature, np.cumsum(select_importances), where='mid', color='m', label='重要度累加')
    plt.title('基于GDBT的特征选择')
    plt.legend()
    plt.show()
'''计算混淆矩阵的二级和三级统计量'''
def calc_confusion_matrix(cnf_matrix):
    TP, TN = cnf_matrix[0, 0], cnf_matrix[1, 1]
    FP, FN = cnf_matrix[1, 0], cnf_matrix[0, 1]
    Accuracy = (TP + TN) / (TP + TN + FP + FN)  # 准确率
    Precision = TP / (TP + FP)  # 精确率
    Recall = TP / (TP + FN)  # 召回率
    Specificity = TN / (TN + FP)  # 特异度
    f1 = (2 * Precision * Recall) / (Precision + Recall)  # f1分值
    print('召回率：{:.2%}，f1分值：{:.2%}'.format(Recall, f1))
'''混淆矩阵的可视化图'''
def draw_confusion_matrix(cm, classes, cmap='Blues'):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('真实类型')
    plt.xlabel('预测类型')
    plt.title('混淆矩阵')
    plt.show()
if __name__ == '__main__':
    '''1、数据探索EDA'''
    data = pd.read_csv('./data/creditcard.csv')
    print('数据加载完毕...')
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    # draw_all_class_radio(data=data)  # 正负样本占比的可视化图
    # draw_has_none(data=data)  # 数据缺省值检测的可视化图
    # draw_one_feature_class_dist(data=data, feature='Amount')  # 某个特征在所有class分布的可视化图
    # draw_all_feature_class_dist(data=data)  # 各个特征在所有class分布的可视化图
    # draw_each_feature_Pearson(data=data)  # 两两特征相关性系数矩阵的可视化图
    # draw_high_to_low_feature(X=X, y=y)  # 高维特征空间映射到低维可视化

    '''2、特征工程'''
    from sklearn.preprocessing import StandardScaler
    col = ['Amount', 'Time']
    X[col] = StandardScaler().fit_transform(X[col])  # 特征标准化
    # # 特征选择：GBDT选择
    # from sklearn.ensemble import GradientBoostingClassifier
    # gdbt_model = GradientBoostingClassifier(random_state=123)
    # gdbt_model.fit(X, y)
    # indices = np.argsort(gdbt_model.feature_importances_)[::-1]  # 特征重要性排序
    # k = 18  # 选择特征的数量
    # select_feature = X.columns[indices[:k]]  # 前k个重要性大的特征
    # select_importances = gdbt_model.feature_importances_[indices[:k]]  # 前k个重要性大的特征重要性
    select_feature = ['V17', 'V14', 'V26', 'V8', 'V10', 'Time', 'V12', 'V7', 'V20', 'V27',
       'V4', 'V16', 'V18', 'V21', 'V23', 'V24', 'V9', 'V6']
    # select_importances = [3.80e-01, 3.49e-01, 9.76e-02, 7.04e-02, 3.13e-02, 2.52e-02, 1.93e-02, 7.78e-03, 5.95e-03,
    #                       5.05e-03, 2.50e-03, 1.79e-03, 1.28e-03, 1.05e-03, 6.96e-04, 3.15e-04, 1.42e-04, 9.99e-05]
    # draw_GDBT_feature_importances(select_feature=select_feature, select_importances=select_importances)  # 基于GDBT的特征选择重要度可视化
    new_X = X[select_feature]  # 选择筛选后的特征列

    # 数据降维
    # print('数据降维...')
    # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    # new_X = LinearDiscriminantAnalysis(n_components=1).fit_transform(new_X, y)

    '''3、建模：训练和评估模型'''
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.3, random_state=123)
    # 处理训练集样本不平衡：过采样
    from imblearn.over_sampling import SMOTE
    X_train, y_train = SMOTE(random_state=123).fit_sample(X_train, y_train)  # 生成过采样训练样本
    train_all_class, test_all_class = pd.value_counts(y_train).values, pd.value_counts(y_test).values
    n_train_sample, n_train_pos_sample, n_train_neg_sample = train_all_class.sum(), train_all_class[0], train_all_class[1]
    n_test_sample, n_test_pos_sample, n_test_neg_sample = test_all_class.sum(), test_all_class[0], test_all_class[1]
    print('训练集样本过采样...')
    print('训练集样本总{}个; 正样本{}个，占{:.2%}; 负样本{}个，占{:.2%}'.format(n_train_sample, n_train_pos_sample, n_train_pos_sample / n_train_sample, n_train_neg_sample, n_train_neg_sample / n_train_sample))
    print('测试样本总{}个; 正样本{}个，占{:.2%}; 负样本{}个，占{:.2%}'.format(n_test_sample, n_test_pos_sample, n_test_pos_sample / n_test_sample, n_test_neg_sample, n_test_neg_sample / n_test_sample))
    penalty, C = 'l1', 0.01  # 模型超参数
    lr_model = LogisticRegression(penalty=penalty, C=C, solver='liblinear')  # 定义模型
    lr_model.fit(X_train, y_train)  # 训练模型
    y_pred = lr_model.predict(X_test)  # 模型在测试集上的预测结果
    print('参与模型训练的特征数量：{}个，特征组合：{}'.format(len(select_feature), select_feature))
    print('模型超参数为：C={}，penalty={}'.format(C, penalty))
    print('逻辑模型测试集准确度为：{:.2%}，ROC曲线下面积为：{:.2%}'.format(accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_pred)))
    from sklearn.metrics import confusion_matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)  # 混淆矩阵
    print('混淆矩阵为：\n{}'.format(cnf_matrix))
    calc_confusion_matrix(cnf_matrix=cnf_matrix)  # 计算混淆矩阵的二级和三级统计量
    # draw_confusion_matrix(cm=cnf_matrix, classes=['正常交易', '欺诈交易'])  # 混淆矩阵的可视化图

    '''4、模型网格搜索调优'''
    # from sklearn.model_selection import GridSearchCV
    # param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2']}  # 模型参数组合
    # grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=3)  # 3折交叉验证
    # grid_search.fit(X_train, y_train)  # 不同参数在训练集上的效果
    # print('参与模型训练的特征数量：{}个，特征组合：{}'.format(len(select_feature), select_feature))
    # print('最优参数组合: {}'.format(grid_search.best_params_))
    # print('最优参数组合训练集准确率: {:.2%}'.format(grid_search.best_score_))
