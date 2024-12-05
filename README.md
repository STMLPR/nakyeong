# 칼로리 소모량 예측모델
운동 데이터를 바탕으로 사용자의 운동 중 소모된 칼로리를 예측하는 모델

## 주요 모델링 기법
### 데이터 전처리
범주형 변수 인코딩: 'Weight_Status', 'Gender'와 같은 범주형 변수를 수치형 데이터로 변환하기 위해 LabelEncoder를 사용. 모델이 범주형 피처를 처리할 수 있도록 준비

특성 선택 및 다항식 변환: PolynomialFeatures(degree=3)은 다항식 특성을 생성하여 원래의 특성을 바탕으로 비선형성을 표현할 수 있는 추가적인 피처 생성함
                        SelectKBest은 모델의 복잡도를 줄이고 계산 효율성을 높이기 위해 F-검정(f_regression)을 통해 가장 유의미한 40개의 피처를 선택하여 특성 공간을 줄임
### 스태킹 앙상블 모델 구성
기초 모델(Base Models): LinearRegression, Ridge, RandomForestRegressor를 기초 모델로 사용하여 각 모델의 예측을 결합. 선형 및 비선형 관계를 학습할 수 있는 특성을 지님
최종 추정기(Final Estimator): 기초 모델들의 예측값을 다시 학습하여 보다 정확한 결과를 도출하기 위해 Ridge를 최종 추정기로 사용하여 기초 모델들의 예측 결과를 종합하여 최종 예측을 수행

### 모델 평가 및 교차 검증
교차 검증: cross_val_score를 사용하여 5-Fold 교차 검증을 수행하고 평균 RMSE 값을 출력함으로써 모델의 일반화 성능 평가
훈련 및 검증 데이터 분할: 데이터를 90:10 비율로 훈련 및 검증 데이터로 나누어 모델을 학습하고, 검증 데이터에서 성능 확인

## 모델 결과 및 분석
### 결과

### 분석 

## 모델 성능 개선을 위한 기법 제안
### 비선형 모델 도입 강화
Gradient Boosting 모델: 스태킹 모델의 기초 모델로는 선형 회귀와 랜덤 포레스트가 사용되었는데, XGBoost, LightGBM, 또는 CatBoost와 같은 Gradient Boosting 모델을 추가하여 복잡한 비선형 패턴을 더 잘 학습할 수 있음
트리 기반 앙상블 모델 추가: 현재 랜덤 포레스트만 사용되고 있으므로, 다양한 트리 기반 모델을 추가하거나 배깅/부스팅 앙상블을 통해 성능을 높일 수 있음

### 하이퍼파라미터 튜닝
RandomizedSearchCV나 GridSearchCV를 사용하여 Ridge, RandomForestRegressor 등의 하이퍼파라미터를 최적화하여 모델의 성능을 높일 수 있음. ex) RandomForest의 n_estimators, max_depth, min_samples_split 등의 파라미터를 최적화
최종 추정기(Ridge)의 하이퍼파라미터인 alpha를 최적화

### Feature Engineering 개선
다양한 피처 변환 기법 도입: Polynomial Features 외에도 로그 변환, 제곱근 변환, 비닝(binning) 등을 사용하여 특성을 변환하고, 비선형성을 모델이 더 잘 학습할 수 있음
새로운 피처 추가: 데이터셋 내 다른 피처 간의 조합을 통해 추가적인 특성을 생성할 수 있음. ex) 키와 몸무게를 이용한 BMI 계산 또는 운동 시간에 기반한 운동 강도 등의 피처가 예측 성능에 도움됨

### Feature Selection 개선
SelectKBest로 40개의 피처를 선택하고 있지만, L1 정규화 회귀나 Recursive Feature Elimination (RFE) 등을 사용해 더 유의미한 피처를 선택하고, 모델의 해석력을 높일 수 있음

### 교차 검증 전략 강화
StratifiedKFold 교차 검증을 사용하여 데이터의 클래스 불균형을 반영한 교차 검증을 수행할 수 있음
또한 폴드의 수를 늘리거나 Repeated K-Fold 교차 검증을 사용하여 더 많은 검증을 수행함으로써 모델의 성능에 대한 신뢰도를 높일 수 있음

### 앙상블 기법의 개선
기초 모델들만 사용하여  구현한 단순한 스태킹 모델을 확장해 Blending이나 Bagging 기법을 추가적으로 적용할 수 있음
Weighted Voting: 스태킹 대신 기초 모델들의 예측에 가중치를 부여하는 방법으로 앙상블 가능. 모델의 검증 성능에 따라 각 모델의 가중치를 조정하면 예측 성능을 높일 수 있음.

### Regularization 기법 도입
Lasso나 ElasticNet과 같은 정규화 모델을 추가하여 피처에 대한 제약을 가함으로써 모델의 과적합을 방지하고 일반화 성능을 높일 수 있음

