# 🏘️ Real Estate Price Prediction via Machine Learning: A Case Study of Pakistan's Cities 🏙️

![Real Estate](https://img.shields.io/badge/Domain-Real%20Estate-blue)
![ML Project](https://img.shields.io/badge/Project-Machine%20Learning-brightgreen)
![Python](https://img.shields.io/badge/Language-Python-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📊 Project Overview

This research project explores the application of machine learning algorithms to predict real estate prices in major Pakistani cities. Traditional methods for forecasting house and flat prices often fail to capture the complex factors affecting price variations. This project aims to:

- 🔍 Provide valuable insights into Pakistan's real estate market
- 💰 Develop accurate price prediction models for properties
- 🧪 Compare various machine learning algorithms for house price prediction
- 🤝 Help buyers negotiate fair prices with real estate agents

## 🌆 The Dataset

The analysis utilizes a comprehensive secondary dataset obtained from Pakistan's open real estate portal (Zameen.com). The dataset contains property listings from 2018-2019 for major metropolitan areas:

- 🏙️ Islamabad
- 🏙️ Lahore
- 🏙️ Karachi
- 🏙️ Rawalpindi
- 🏙️ Faisalabad

### 📋 Dataset Features

The original dataset includes the following columns:
- property_id
- location_id
- page_url
- property_type
- price
- location
- city
- province_name
- latitude
- longitude
- baths
- area
- purpose
- bedrooms
- date_added
- agency
- agent
- Area Type
- Area Size
- Area Category

## 🧹 Data Preprocessing

### Cleaning Steps:
1. **Removed Unnecessary Columns**:
   - location_id
   - page_url
   - agency
   - agent
   - date_added
   - location (housing societies and street names)

2. **Filtered Property Purposes**:
   - Focused only on "For Sale" properties (71.6% of the dataset)
   - Removed "For Rent" properties (28.4%)

3. **Refined Property Types**:
   - Kept the two main property types: Houses (87,976) and Flats (28,118)
   - Removed minority property types: Upper Portion, Lower Portion, Farm House, Penthouse, and Room

4. **Handled Missing Values**:
   - No missing values were found in the final dataset

### 📊 Data Normalization

Applied Min-Max scaling to ensure all features are on the same scale using the formula:

$$X_{normalized} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

This step ensures that machine learning algorithms are not biased toward features with larger magnitudes.

## 📈 Exploratory Data Analysis

Several key insights were discovered during exploratory data analysis:

- Price increases with the number of bedrooms and bathrooms
- Property size category significantly affects price
- Price variations exist across different cities in Pakistan
- Location (city and province) plays a crucial role in determining property prices

## 🧠 Machine Learning Models

The project implemented and compared ten machine learning algorithms, all hyperparameter-tuned using GridSearchCV:

### 1️⃣ Stochastic Gradient Descent (SGD)
- Optimizes model coefficients by iteratively adjusting them to minimize mean square error
- Hyperparameters: alpha=0.04, eta0=0.08, learning_rate='invscaling', max_iter=213, penalty='l1'

### 2️⃣ Decision Tree
- Constructs a decision tree to model relationships in data
- Hyperparameters: max_features='auto', min_samples_leaf=7, min_samples_split=5, max_depth=12

### 3️⃣ Gradient Boosting Regressor
- Uses boosting ensemble methods
- Hyperparameters: learning_rate=0.099, max_depth=9, max_features='sqrt', min_samples_leaf=3, min_samples_split=4, n_estimators=280, warm_start=False

### 4️⃣ XGBRegressor
- Provides parallel tree boosting
- Hyperparameters: gamma=0, learning_rate=0.089, max_depth=16, min_child_weight=25, reg_lambda=12

### 5️⃣ AdaBoost Regressor
- Combines multiple weak learners into a strong learner
- Hyperparameters: learning_rate=0.01, n_estimators=214, base_estimator=DecisionTreeRegressor(max_depth=14), loss='exponential'

### 6️⃣ Random Forest Regressor
- Creates multiple decision trees and combines their predictions
- Hyperparameters: bootstrap=True, n_estimators=205, min_samples_split=6, min_samples_leaf=1, max_features='sqrt', oob_score=True

### 7️⃣ CatBoost Regressor
- Gradient boosting on decision trees
- Hyperparameters: depth=9, iterations=180, l2_leaf_reg=0.287, learning_rate=0.0999, model_size_reg=0.1

### 8️⃣ KNeighbors Regressor
- Memorizes training data and predicts based on k-nearest neighbors
- Hyperparameters: n_neighbors=6

### 9️⃣ LightGBM Regressor
- Gradient boosting framework using tree-based learning
- Hyperparameters: learning_rate=0.2, max_depth=12, min_child_samples=9, n_estimators=300, num_leaves=14, reg_lambda=0.503

### 🔟 Stacking Regressor
- Combines strong heterogeneous learners
- Base models: RandomForestRegressor, GradientBoostingRegressor, KNeighborsRegressor
- Meta-model: LinearRegression

## 📊 Model Evaluation

The models were evaluated using two metrics:
1. **R² Score**: Measures how well the model explains variance in the dependent variable
2. **Mean Absolute Percentage Error (MAPE)**: Calculates the mean percentage difference between predicted and actual values

Two data splitting strategies were tested:
- 75% training / 25% testing
- 85% training / 15% testing

### Results with 85% training / 15% testing:

| Algorithm | Test MAPE | Test R² | Train MAPE | Train R² |
|-----------|-----------|---------|------------|----------|
| SGD Regressor | 0.7088 | 0.7099 | 0.7007 | 0.7119 |
| Decision Tree | 0.2551 | 0.8719 | 0.2403 | 0.8965 |
| Gradient Boosting Regressor | 0.2042 | 0.917 | 0.1735 | 0.9683 |
| XGB Regressor | 0.199 | 0.9033 | 0.1764 | 0.9104 |
| AdaBoost Regressor | 0.2109 | 0.9103 | 0.1797 | 0.9709 |
| Random Forest Regressor | 0.1878 | 0.9106 | 0.1115 | 0.9555 |
| CatBoost Regressor | 0.2541 | 0.9059 | 0.2486 | 0.9311 |
| Light GBM Regressor | 0.2437 | 0.911 | 0.2394 | 0.9346 |
| Stacking Regressor | 0.1877 | 0.9157 | 0.1355 | 0.9696 |
| KNN Regressor | 0.3205 | 0.7996 | 0.2665 | 0.8415 |

### Results with 75% training / 25% testing:

| Algorithm | Test MAPE | Test R² | Train MAPE | Train R² |
|-----------|-----------|---------|------------|----------|
| SGD Regressor | 0.6891 | 0.7245 | 0.6839 | 0.7026 |
| Decision Tree | 0.2529 | 0.8707 | 0.2348 | 0.8917 |
| Gradient Boosting Regressor | 0.2029 | 0.9108 | 0.1691 | 0.9701 |
| XGB Regressor | 0.1972 | 0.9019 | 0.1703 | 0.9094 |
| Ada Boost Regressor | 0.209 | 0.9075 | 0.1752 | 0.9722 |
| Random Forest Regressor | 0.1868 | 0.9076 | 0.1106 | 0.9517 |
| Cat Boost Regressor | 0.2558 | 0.9053 | 0.2469 | 0.9323 |
| Light GBM Regressor | 0.2428 | 0.914 | 0.2348 | 0.9341 |
| Stacking Regressor | 0.1868 | 0.9153 | 0.1388 | 0.9696 |
| KNN Regressor | 0.322 | 0.8033 | 0.2653 | 0.8327 |

## 🏆 Key Findings

1. **Ensemble Methods Outperform Single Models**: Ensemble techniques (Stacking, Random Forest, Gradient Boosting) achieved superior performance compared to individual learners.

2. **Best Overall Model**: The Stacking Regressor demonstrated exceptional performance with:
   - Lowest MAPE (18.77% with 15% test set)
   - Highest R² score (0.9157 with 15% test set)

3. **Strong Performers**: Random Forest, XGBoost, and Gradient Boosting Regressors also showed excellent results, achieving R² scores above 0.90 and MAPE values below 21%.

4. **Weaker Performers**: SGD Regressor and KNN showed the lowest performance metrics.

5. **Model Stability**: The models showed consistent performance across both train-test split ratios, indicating robust learning.

## 🎯 Conclusion

This research demonstrates the effectiveness of machine learning algorithms in real estate price prediction for the Pakistani market. The stacking ensemble method achieved the highest evaluation metrics, followed closely by Random Forest.

These models can provide valuable assistance to:
- 🏠 Homebuyers seeking fair property prices
- 💼 Real estate agents for market analysis
- 📈 Investors making informed decisions
- 📊 Policymakers understanding housing market trends

## 👨‍💻 Author

This project was created by Abuzar as part of a Master's in Data Science program (1st semester).

## 🔗 References

- Original article: [Real Estate Price Prediction via Machine Learning: A Case Study of Pakistan's Cities](https://www.paradigmshift.com.pk/real-estate-machine-learning/)
- Dataset source: Zameen.com (Pakistan's open real estate portal) (I used this secondary dataset, taken from KAGGLE)

## 📜 License

This project is available for academic and research purposes.
