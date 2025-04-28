# 📈 Henry Hub Natural Gas Price Prediction

미국 헨리 허브(Henry Hub) 천연가스 가격 예측을 위한 관계 분석 및 머신러닝 모델링 프로젝트입니다.

---

## 📊 Correlation Analysis (`Correlation Analysis.ipynb`)

- **목표**: Henry Hub 가격과 주요 경제 지표 간 상관관계 분석
- **주요 작업**:
  - 상관계수 기반으로 주요 변수 선정 (ex: 원유 지수, 가스 굴착기수, 세계경기지수 등)
  - 시계열 패턴 유사성 비교 및 인과성 탐색
  - 단변량 회귀분석(OLS)으로 유의미한 변수 식별
  - 변수별 산점도 시각화

- **주요 인사이트**:
  - 공급 측면 지표(굴착기 수, 원유 지수 등)가 가격 변동에 가장 큰 영향
  - 세계 경기지수, 달러 인덱스 등은 보조적 역할

---

## 🛠️ Modeling (`Modeling.ipynb`)

- **목표**: Henry Hub 가격을 예측하는 머신러닝 모델 최적화 및 성능 비교
- **주요 작업**:
  - 결측치 다중 보간법(6가지 방식 앙상블) 적용
  - XGBoost, RandomForest, CatBoost 모델 학습
  - Optuna를 통한 하이퍼파라미터 최적화
  - 개별 모델 및 앙상블(평균, 가중 평균) 결과 비교

- **모델 성능 요약**:

| Model              | MAPE  | R² Score |
|--------------------|-------|----------|
| XGBoost            | 0.0733 | 0.9442   |
| RandomForest       | 0.0831 | 0.8806   |
| CatBoost           | 0.0719 | 0.9146   |
| Ensemble (Mean)    | 0.0726 | 0.9221   |
| Ensemble (Weighted)| 0.0723 | 0.9261   |

- **주요 특징**:
  - 앙상블(Weighted)이 가장 높은 예측 성능 달성
  - 결측치 문제를 다양한 보간법으로 해결하여 데이터 손실 최소화
  - Optuna를 활용해 각 모델을 최적 성능으로 튜닝

---

## 📝 사용 기술

- Python (Pandas, Numpy, Matplotlib, Seaborn, Plotly)
- Scikit-learn
- XGBoost, RandomForest, CatBoost
- Optuna (하이퍼파라미터 최적화)
- Statsmodels (회귀분석)

---

## 📂 파일 구조

├── Correlation Analysis.ipynb # 관계 분석 및 EDA ├── Modeling.ipynb # 모델링 및 최적화 ├── data_train.csv # 훈련 데이터 ├── data_test.csv # 테스트 데이터 ├── submission_example.csv # 제출 포맷 예시 ├── optuna/ # Optuna study 저장 폴더 ├── submission.csv # 최종 제출 파일
