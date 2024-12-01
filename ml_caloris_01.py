#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
import warnings

warnings.filterwarnings('ignore')


# In[2]:


# Seed 고정
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)


# In[3]:


# 데이터 불러오기
train = pd.read_csv('cal.train.csv')
test = pd.read_csv('cal.test.csv')
sample_submission = pd.read_csv('cal.sample_submission.csv', index_col=0)


# In[4]:


# 데이터 전처리
ordinal_features = ['Weight_Status', 'Gender']
for feature in ordinal_features:
    le = LabelEncoder()
    le.fit(train[feature])
    train[feature] = le.transform(train[feature])
    for label in np.unique(test[feature]):
        if label not in le.classes_:
            le.classes_ = np.append(le.classes_, label)
    test[feature] = le.transform(test[feature])


# In[5]:


# Feature Selection 및 PolynomialFeatures 적용
train_x = train.drop(['ID', 'Calories_Burned', 'Weight_Status', 'Height(Remainder_Inches)', 'Height(Feet)'], axis=1)
train_y = train['Calories_Burned']
test_x = test.drop(['ID', 'Weight_Status', 'Height(Remainder_Inches)', 'Height(Feet)'], axis=1)

poly = PolynomialFeatures(degree=3)  # 모든 고차항 생성
train_poly = poly.fit_transform(train_x)
test_poly = poly.transform(test_x)

# Feature Selection
selector = SelectKBest(score_func=f_regression, k=40)
train_poly = selector.fit_transform(train_poly, train_y)
test_poly = selector.transform(test_poly)


# In[6]:


# 스태킹 모델 구성
base_models = [
    ('linear', LinearRegression()),
    ('ridge', Ridge()),
    ('rf', RandomForestRegressor(n_estimators=50, random_state=42))  # RandomForest 추가
]

stacking = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(alpha=1.0)  # 최종 추정기로 Ridge 사용
)


# In[7]:


# 교차 검증
cv_scores = np.sqrt(-cross_val_score(stacking, train_poly, train_y, cv=5, scoring='neg_mean_squared_error'))
print(f"Cross-validation RMSE: {cv_scores.mean()}")


# In[8]:


# 학습 및 검증
X_train, X_val, y_train, y_val = train_test_split(train_poly, train_y, test_size=0.1, random_state=42, shuffle=True)
stacking.fit(X_train, y_train)


# In[9]:


# 검증 성능 평가
y_pred = stacking.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f"Validation RMSE: {rmse}")


# In[10]:


# 테스트 데이터 예측 및 제출 파일 생성
test_preds = stacking.predict(test_poly)
sample_submission['Calories_Burned'] = np.round(test_preds)
sample_submission.to_csv('submission_01.csv', index=True)


# In[ ]:




