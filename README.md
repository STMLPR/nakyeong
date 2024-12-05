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
Cross-validation RMSE: 0.2905
Validation RMSE: 0.2891

### 분석 
Cross-validation RMSE(0.2905)와 Validation RMSE(0.2891)는 비슷한 수준으로, 모델의 일반화 성능이 비교적 안정적임. 그러나 모델의 성능을 높이기 위해 Voting 기법을 추가하거나 하이퍼파라미터 튜닝, 피처 엔지니어링 등을 적용하면 더욱 성능을 향상시킬 수 있음

### Voting 기법 추가 사용 결과 및 분석
#### 결과
Voting Regressor Cross-validation RMSE: 0.9399324847927749

Voting Regressor Validation RMSE: 0.8995721450451558

#### 분석
Voting Regressor의 RMSE가 Stacking Regressor보다 크게 증가: RMSE 값이 크게 증가했다는 것은 모델이 데이터를 제대로 학습하지 못했거나, Voting 기법의 조합이 데이터에 적합하지 않았다는 것을 나타냄

#### Voting 기법의 한계 및 데이터 특성
일반적으로 Voting 기법은 여러 모델의 예측을 결합하여 오차를 줄이고 성능을 높이는 데 사용되지만, 이번 학습에서는 다른 결과가 나타남 

모델 간 성능 편차: Voting 기법은 개별 모델의 성능이 균일할 때 좋은 성능을 발휘하는 경향이 있음. 하지만 Voting Regressor에서 사용된 모델(Linear Regression, Ridge, RandomForest) 간에 예측 성능 차이가 크다면, 성능이 낮은 모델의 영향으로 전체적인 성능이 감소했을 가능성 존재

비선형성에 대한 대응 부족: Voting Regressor는 개별 모델의 예측 결과를 단순 평균하여 결합하는데, 데이터가 매우 복잡하고 비선형적인 경우에는 개별 선형 모델(예: Linear Regression, Ridge)이 충분히 데이터를 설명하지 못해 전체 성능이 저하될 수 있음

과적합 문제: Voting Regressor는 각 모델의 예측을 평균화함으로써 과적합을 방지하는 장점이 있음. 그러나 과적합된 모델이 포함되어 있는 경우, Voting의 평균화된 결과도 부정적인 영향을 받을 수 있음

#### Stacking 모델과 Voting 모델의 차이점
Stacking 모델은 각 기초 모델의 예측을 다시 학습하여 더 복잡한 관계를 학습할 수 있있음 특히, 최종 추정기로 Ridge를 사용하여 기초 모델들이 놓친 패턴을 보완 가능

Voting 모델은 단순히 각 모델의 결과를 평균화하거나 가중 평균을 하여 결합하는 방식. 복잡한 패턴을 학습하기보다는 모든 모델의 예측에 균일하게 의존하기 때문에, 모델들이 충분히 다양한 패턴을 잘 학습하지 못하면 성능이 떨어질 수 있음

#### Voting 모델 개선 방안
Voting 모델에 적합한 모델 선택: 현재 Voting Regressor에 사용된 모델들(Linear Regression, Ridge, RandomForest) 중 일부는 데이터의 비선형성을 잘 표현하지 못할 수 있음. 비선형성을 잘 학습할 수 있는 Gradient Boosting 모델(XGBoost, LightGBM, CatBoost)을 추가하여 다양한 유형의 모델을 Voting에 포함시키는 것이 도움이 될 수 있음

가중 Voting 기법 도입: 각 모델의 성능이 균일하지 않기 때문에, 가중 Voting 기법을 사용하여 성능이 좋은 모델에 더 높은 가중치를 부여해 성능이 낮은 모델의 영향을 줄이고 Voting의 성능을 높일 수 있음

하이퍼파라미터 최적화: Voting에 포함된 모델들(특히 랜덤 포레스트나 Ridge)의 하이퍼파라미터를 최적화하여 각 모델이 개별적으로 더 좋은 성능을 낼 수 있도록 해서 전체 앙상블의 성능 향상

더 복잡한 Stacking 앙상블 고려: Voting 앙상블이 잘 작동하지 않을 경우, 이전의 스태킹 모델에 더 다양한 모델을 추가하거나, 스태킹의 스택 레벨을 깊게 설정하여 성능 개선. Stacking 앙상블은 각 모델의 예측을 최종 추정기가 학습하여 다시 조정할 수 있기 때문에 데이터의 복잡한 특성을 더 잘 반영 가능함

모델 성능 확인 후 선택: Cross-validation 결과와 Validation 결과가 모두 Voting에서 낮았기 때문에, Voting 모델 대신 Stacking Regressor를 사용하거나, Voting과 Stacking의 성능을 비교하여 더 나은 모델을 선택


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

