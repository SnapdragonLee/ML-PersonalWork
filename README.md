# 2021 Autumn ML-Personalwork

Personalwork Code, Opensource with GPLv2 License



### 个人作业 投篮命中预测

根据投篮的位置，投篮手段，比赛时间等因素，预测单次投篮是否会命中。







### **Author**:

SnapdragonLee



### Data：

20558 条训练数据， 5139 条测试数据，共有 20 条属性，目标标签为 `shot_made_flag`。

**Training Set**：`DataSet/Train.csv`

**Test Set**：`DataSet/Test.csv`



### Evaluation：

**roc_auc_score**



### Submit：

Submission.zip (submission.txt) 以测试集数据为顺序，每一行记录模型预测出的命中概率  



### Score：

 0.7133916302483058, Rank 3/22.



### Methods

基于 `XGBClassifier`，具体可以查看 **report.pdf** 。

