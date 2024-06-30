import pandas as pd
import numpy as np
import category_encoders as ce
import pdb

def get_data():
    train_data = pd.read_csv('data/train-466844.csv', on_bad_lines='skip')
    test_data = pd.read_csv('data/test-439524.csv')
    test_CUST_ID = test_data['CUST_ID']

    drop_columns = [
        'OPEN_ORG_NUM', # 3, 开户机构
        'IDF_TYP_CD', # 4, 证件类型
        'GENDER', # 5, 性别
        'CUST_EUP_ACCT_FLAG', # 30, 是否欧元账户,
        'CUST_AU_ACCT_FLAG', # 31, 是否澳元账户
        'CUST_DOLLER_FLAG', # 35, 是否美元账户
        'CUST_INTERNATIONAL_GOLD_FLAG', # 36, 是否国际金卡
        'CUST_INTERNATIONAL_COMMON_FLAG', # 37, 是否国际普卡
        'CUST_INTERNATIONAL_SIL_FLAG', # 38, 是否国际银卡
        'CUST_INTERNATIONAL_DIAMOND_FLAG', # 39, 是否国际钻石卡
        'CUST_GOLD_COMMON_FLAG', # 40, 是否金卡
        'CUST_STAD_PLATINUM_FLAG', # 41, 是否标准白金卡
        'CUST_LUXURY_PLATINUM_FLAG', # 42, 是否豪华白金卡
        'CUST_PLATINUM_FINANCIAL_FLAG', # 43, 是否白金理财卡
        'CUST_DIAMOND_FLAG', # 44, 是否钻石卡
        'CUST_INFINIT_FLAG', # 45, 是否无限卡
        'CUST_BUSINESS_FLAG', # 46, 是否商务卡
    ]

    train_data.drop(drop_columns, axis=1, inplace=True)
    test_data.drop(drop_columns, axis=1, inplace=True)
    
    # train_data = train_data.convert_dtypes()

    train_data = train_data.drop_duplicates(keep='first') # 去除重复行
    train_data.dropna(inplace=True) # 去除空值的行

    def onehot(data):
        encoder = ce.OneHotEncoder(use_cat_names=True)
        number_column = data.select_dtypes(include=[np.number])
        nan_column = data.select_dtypes(exclude=[np.number])
        #pdb.set_trace()
        onehot_encoded = encoder.fit_transform(nan_column)
        data = pd.concat([number_column, onehot_encoded], axis=1)
        return data

    train_data = train_data.apply(pd.to_numeric, errors='ignore')
    test_data = test_data.apply(pd.to_numeric, errors='ignore')
    train_data = onehot(train_data)
    test_data = onehot(test_data)
    
    goal = train_data['bad_good']
    train_data.drop(['bad_good'], axis=1, inplace=True)
    
    # pdb.set_trace()
    
    return train_data, test_data, goal, test_CUST_ID
