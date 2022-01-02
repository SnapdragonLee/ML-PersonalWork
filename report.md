## 机器学习个人作业文档

本人通过个人实践、代码编写、优化调参，完成了一份评分为 0.7133916302483058，排名第 3 的代码。总体来说，首先尝试了逻辑回归的模型，但是分类后发现效果并不是很好，后来评测方式从 `logloss `改了以后发现效果可能更不好了，所以尝试使用了 `xgboost `包中的并行 `sklearn`，里面的分类器 `XGBClassifier`。这已经不是我第一次使用了，所以直接进行了一波改造，然后对学习率、最大深度、分裂数等参数进行了参数调整。





### 数据预处理

根据不同的维度，筛选有用的信息，去除无用的信息，并采取二值化的方式进行单属性分类。数据读取使用 pandas 库进行操作。





### xgboost 与 XGBClassifier

逻辑回归的部分我们就不提了，主要是效果着实不令人满意。由于之前也使用过其他的模型，所以直接就转手扔掉了自己的逻辑回归代码。

`xgboost` 是 `Boosting` 算法的其中一种，`Boosting` 算法的思想是将许多弱分类器集成，形成一个强分类器。 ``xgboost` 是一种提升树模型，所以它是将许多树模型集成在一起，形成一个强分类器。而所用到的树模型则是 `CART` 回归树模型。XGBoost 应用了更好的正则化技术来减少过度拟合，这是梯度提升的区别之一。

另外 `Xgboost` 是在GBDT的基础上进行改进，使之更强大，适用于更大范围。

在使用的时候，`Xgboost` 一般和 `sklearn` 一起使用，但是由于 `sklearn` 中没有集成 `xgboost`，所以才使用了 `xgboost.sklearn`。

对于 `XGBClassifier` 是一个用于分类的 `scikit-learn` API，即是众多分类器中的其中一种。



其实说到底，`xgboost` 的基模型还是决策树模型，只是取决于不同的优化罢了。`xgboost `在使用的时候，我自己的电脑 16 核可以全部占满，由此可以看到 `xgboost` 的优化之到位。





### 参数调整

对于 `XGBClassifier` 来说，有很多的参数可以调整，毕竟它的底层继承了父类很多的属性。

我调的参数大概如下：



**learning_rate**：指定学习率。默认值为 **0.3**，我调整为 0.01。

**max_depth**：指定树的深度，小心过拟合，默认值为 **6**，我这里调整为 7。

**n_estimators：**总共迭代的次数，我调整为 550。

**subsample**：训练每棵树时，使用的数据占全部训练集的比例，这个也要小心过拟合，默认是1，这里我调低了变成 0.5。

**colsample_bytree**：训练每棵树时，使用的特征占全部特征的比例，这个依然要小心过拟合，默认是1，这里我调低了变成 0.5。

**eval_metric**：校验数据所需要的评价指标，我这里选用的还是 `logloss`。





### 具体代码：

###### 完整代码如下：

```python
import numpy as np
import pandas as pd
import matplotlib as plt

from sklearn.linear_model import LogisticRegression
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelBinarizer

from xgboost.sklearn import XGBClassifier

plt.use('Agg')

dataDir = "./DataSet/"
trainData = "Train.csv"
testData = "Test.csv"


def mapperBiuld(df):
    x_mapper = DataFrameMapper([
        (u'action_type', LabelBinarizer()),
        (u'combined_shot_type', LabelBinarizer()),
        (u'loc_x', None),
        (u'loc_y', None),
        (u'minutes_remaining', None),
        (u'period', LabelBinarizer()),
        (u'playoffs', LabelBinarizer()),
        (u'seconds_remaining', None),
        (u'shot_distance', None),
        (u'shot_type', LabelBinarizer()),
        (u'shot_zone_area', LabelBinarizer()),
        (u'shot_zone_basic', LabelBinarizer()),
        (u'shot_zone_range', LabelBinarizer()),
        (u'time_remaining', None),
        (u'opponent_num', LabelBinarizer()),
        (u'game_id_num', LabelBinarizer()),

        """
        (u'game_id', )
        (u'season',)
        (u'shot_made_flag', )
        (u'opponent',)
        (u'action_type_num', )
        (u'combined_shot_type_num',)
        """
    ])

    x_mapper.fit(df)
    y_mapper = DataFrameMapper([(u'shot_made_flag', None), ])

    y_mapper.fit(df)
    return x_mapper, y_mapper


def xgboost_mappedvec(df_train, df_test):
    x_mapper, y_mapper = mapperBiuld(df_train)
    train_x_vec = x_mapper.transform(df_train.copy())
    train_y_vec = y_mapper.transform(df_train.copy())
    test_x_vec = x_mapper.transform(df_test.copy())

    clf = XGBClassifier(max_depth=7, learning_rate=0.01, n_estimators=550, subsample=0.5, colsample_bytree=0.5,
                        eval_metric=['mseloss', 'auc', 'error'])

    clf.fit(train_x_vec, train_y_vec)
    test_y_vec = clf.predict_proba(test_x_vec)[:, 1]
    return test_y_vec


def preproc(df):
    df["time_remaining"] = df["minutes_remaining"] * 60 + df["seconds_remaining"]

    action_type_list = list(set(df["action_type"].tolist()))
    df["action_type_num"] = pd.Series([action_type_list.index(df["action_type"][i]) for i in range(0, len(df))])

    combined_shot_type_list = list(set(df["combined_shot_type"].tolist()))
    df["combined_shot_type_num"] = pd.Series(
        [combined_shot_type_list.index(df["combined_shot_type"][i]) for i in range(0, len(df))])

    opponent_list = list(set(df["opponent"].tolist()))
    df["opponent_num"] = pd.Series([opponent_list.index(df["opponent"][i]) for i in range(0, len(df))])

    game_id_list = list(set(df["game_id"].tolist()))
    df["game_id_num"] = pd.Series([game_id_list.index(df["game_id"][i]) for i in range(0, len(df))])

    del df["game_event_id"], df["lat"], df["lon"]
    return df


def makeSubmission(predict_y, savename):
    '''
        submit_df = pd.read_csv(dataDir + "sample_submission.csv")
        submit_df["shot_made_flag"] = predict_y
        submit_df = submit_df.fillna(np.nanmean(predict_y))
        submit_df.to_csv(savename, index=False)
    '''

    np.savetxt(X=predict_y, fname=savename, fmt='%.09f')
    pass


def mean_game(train_df, test_df):
    game_id_list = list(set(df["game_id"].tolist()))
    success_rate_game = np.array([train_df["shot_made_flag"][train_df["game_id"] == game_id_list[i]].mean() for i in
                                  range(0, len(game_id_list))])

    predict_y = success_rate_game[test_df["game_id_num"]]
    return predict_y



def mean_all(train_df, test_df):
    return train_df["shot_made_flag"].mean()


if __name__ == "__main__":
    df_trn = pd.read_csv(dataDir + trainData)
    df_trn = preproc(df_trn)

    df_tst = pd.read_csv(dataDir + testData)
    df_tst = preproc(df_tst)

    # logistic
    
    # xgboost
    predict_y = xgboost_mappedvec(df_trn, df_tst)
    makeSubmission(predict_y, savename="submission_xgboost.txt")

    """
    loc = train_df[["loc_x", "loc_y"]].as_matrix()
    flag = train_df["shot_made_flag"].as_matrix()
    heatmap_all, xedges, yedges = np.histogram2d(loc[:, 0], loc[:, 1], bins=100)
    heatmap_success, xedges, yedges = np.histogram2d(loc[flag==1, 0], loc[flag==1, 1], bins=100)
    sns.heatmap(heatmap_all, vmax=50)
    success_rate = heatmap_success / heatmap_all
    success_rate[np.is]
    namelist = ["action_type", "combined_shot_type", "game_event_id",
                "game_id", "lat", "loc_x", "loc_y", "lon", "minutes_remaining",
                "period", "playoffs", "season", "seconds_remaining", "shot_distance",
                "shot_made_flag", "shot_type", "shot_zone_area", "shot_zone_range", 
                "team_id", "team_name", "opponent", "shot_id"]
    y_pred = clf.predict(train_x)
    print("Number of mislabeled points out of a total %d points : %d"  % (train_x.shape[0],(train_y != y_pred).sum()))
    def logloss(act, pred):
        epsilon = 1e-15
        pred = sp.maximum(epsilon, pred)
        pred = sp.minimum(1-epsilon, pred)
        ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
        ll = ll * -1.0/len(act)
        print(ll)
        return ll
    
    logloss(train_y,clf.predict_proba(train_x)[:,1])
    """
```



###### 项目结构：

```
│  main.py
│  submission_wf.txt
│  submission_xgboost.txt
│
└─DataSet
        Test.csv
        Train.csv
```





### 感想：

可以说这个题目中规中矩，数据的处理与输出都比较常规，但是对于个人来说如果可以加入更多的训练策略可能会让模型变得更好，比如调整参数的时候加入更多的候选参数，例如 `colsample_bylevel` 等，这样可能会让结果更丰富，产生意想不到的后果。加入所有的参数之后，可以制作一个贝叶斯自动调参器，在一个空间内自动找出最佳参数区域。模型构造上可以尝试尝试优化一下 `Dataframe`，可以尝试一下不同的预处理，感觉可以做的还有很多，还可以尝试更多。

暂时没有了解到 `xgboost` 包是如何做到决策树并行的，什么时候接触底层代码的时候可以细细研究一下。