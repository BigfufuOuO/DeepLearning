from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from dataload import get_data
import pandas as pd
import pdb

train_data, test_data, goal, test_CUST_ID = get_data()
print('Data loaded')
# pdb.set_trace()

X_train, X_val, y_train, y_val = train_test_split(
    train_data, goal, test_size=0.2
)

XGBtree = XGBClassifier(
    learning_rate=0.075,
    n_estimators=200,
    max_depth=5,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.5,
    device='gpu',
    early_stopping_rounds=10
)

# score = cross_val_score(XGBtree, train_data, goal, cv=5).mean()
# print('Train | Cross validation score:', score)

# train
model = XGBtree.fit(X_train, y_train, 
                    eval_set=[(X_val, y_val)],
                    )
print('Trained | Best score:', model.best_score)

# validation
y_pred = model.predict(X_val)
cols_when_model_builds = model.get_booster().feature_names
print('Val | Accuracy:', accuracy_score(y_val, y_pred))
print('Val | Precision:', precision_score(y_val, y_pred))
print('Val | Recall:', recall_score(y_val, y_pred))
print('Val | F1:', f1_score(y_val, y_pred))

# test
test_data = test_data[cols_when_model_builds]
test_pred = model.predict(test_data)
test_pred = pd.DataFrame(test_pred, columns=['bad_good'])
result = pd.concat([test_CUST_ID, test_pred], axis=1)
# pdb.set_trace()
# 求macro f1
macro_F1 = f1_score(y_val, y_pred, average='macro')
print('Test | macro F1:', macro_F1)
# record result
result.to_csv(f'result_{macro_F1}.csv', index=False)