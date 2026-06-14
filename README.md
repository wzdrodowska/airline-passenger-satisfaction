# Airline Passenger Satisfaction Analysis and Modeling

## 1. Introduction

The motivation for choosing the **Airline Passenger Satisfaction** dataset was to understand which elements of air travel are most important in shaping passengers' final satisfaction assessment.

The goal of this project is to analyze the influence of different travel, passenger, and service-quality factors on airline passenger satisfaction, and to build a predictive model for the target variable: passenger satisfaction level.

## 2. Dataset Description

The dataset comes from Kaggle:  
https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

The data is already split into two subsets:

- `train.csv` - used for analysis, data cleaning, feature engineering, and model training
- `test.csv` - used only for the final evaluation of the trained models

Dataset size:

| Subset | Rows | Columns |
| --- | ---: | ---: |
| Training set | 103,904 | 25 |
| Test set | 25,976 | 25 |

### Passenger Variables

- `Gender` - passenger gender (`Female`, `Male`)
- `Customer Type` - customer type (`Loyal Customer`, `disloyal Customer`)
- `Age` - passenger age

### Travel Variables

- `Type of Travel` - travel purpose (`Personal Travel`, `Business travel`)
- `Class` - travel class (`Business`, `Eco`, `Eco Plus`)
- `Flight Distance` - flight distance

### Service Quality Variables

Most service-quality variables are ratings on a scale from 1 to 5. In the case of `Inflight wifi service`, a value of 0 means that the service was not available.

- `Inflight wifi service`
- `Departure/Arrival time convenient`
- `Ease of Online booking`
- `Gate location`
- `Food and drink`
- `Online boarding`
- `Seat comfort`
- `Inflight entertainment`
- `On-board service`
- `Leg room service`
- `Baggage handling`
- `Checkin service`
- `Inflight service`
- `Cleanliness`
- `Departure Delay in Minutes`
- `Arrival Delay in Minutes`

### Target Variable

- `satisfaction` - passenger satisfaction level (`satisfied`, `neutral or dissatisfied`)

## 3. Exploratory Data Analysis

The first step of the exploratory data analysis was to examine the distribution of the target variable.

| Satisfaction level | Share |
| --- | ---: |
| neutral or dissatisfied | 56.67% |
| satisfied | 43.33% |

The target variable does not show extreme class imbalance, so additional class-balancing techniques were not required before modeling.

The analysis of categorical variables showed visible differences in satisfaction depending on:

- travel class
- customer type
- type of travel

These variables carry useful predictive information and were included in later modeling stages.

The analysis of service-quality ratings using boxplots showed that satisfied passengers generally gave higher and more consistent ratings for important service elements such as:

- seat comfort
- inflight service
- cleanliness

This confirmed that service-quality ratings are strongly related to passenger satisfaction and are useful for prediction.

The correlation analysis showed moderate positive relationships between several service-quality variables and a very strong correlation between the delay-related variables. Other variables showed weaker dependencies, suggesting a limited risk of feature redundancy.

## 4. Missing Values and Outliers

Missing values appeared only in the `Arrival Delay in Minutes` column:

| Subset | Missing values |
| --- | ---: |
| Training set | 310 |
| Test set | 83 |

These missing values were assumed to correspond to cases with no arrival delay and were filled with `0`.

The strongest outliers were observed in the delay variables. Most flights had no delay or only a small delay, while a small number of cases had very large delays. To reduce the influence of extreme values, `Departure Delay in Minutes` and `Arrival Delay in Minutes` were clipped to the interval defined by the 1st and 99th percentiles, using boundaries calculated on the training set.

Calculated clipping boundaries:

| Variable | Lower bound | Upper bound |
| --- | ---: | ---: |
| Departure Delay in Minutes | 0.00 | 181.97 |
| Arrival Delay in Minutes | 0.00 | 183.00 |

After clipping, the maximum delay values were reduced from 1,592 and 1,584 minutes to approximately 182 and 183 minutes without disrupting the overall data structure.

The columns `Unnamed: 0` and `id` were removed because they did not provide meaningful predictive information.

## 5. Feature Engineering

The dataset was split into:

- `X_train`, `X_test` - explanatory variables
- `y_train`, `y_test` - target variable

Categorical variables:

- `Gender`
- `Customer Type`
- `Type of Travel`
- `Class`

Numerical variables:

- `Age`
- `Flight Distance`
- service-quality ratings
- delay variables

The preprocessing pipeline used:

- `StandardScaler` for numerical variables
- `OneHotEncoder` for categorical variables
- `ColumnTransformer` to combine both preprocessing paths

Numerical variables were standardized to improve the performance of scale-sensitive models, while categorical variables were encoded using one-hot encoding.

## 6. Model Training

The modeling stage started with a baseline classifier. The baseline always predicts the majority class: `neutral or dissatisfied`.

Baseline accuracy:

```text
0.5610
```

Three machine learning models were then trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

Each model was built as a scikit-learn pipeline combining preprocessing and classification.

## 7. Model Evaluation

The models were evaluated on the test set using:

- Accuracy - overall classification performance
- Precision - quality of predictions for the `satisfied` class
- Recall - ability to identify satisfied passengers
- F1-score - balance between precision and recall

Final evaluation results:

| Model | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.561018 | 0.000000 | 0.000000 | 0.000000 |
| Logistic Regression | 0.871920 | 0.869510 | 0.833289 | 0.851014 |
| Decision Tree | 0.947759 | 0.939842 | 0.941244 | 0.940542 |
| Random Forest | 0.963120 | 0.972667 | 0.942471 | 0.957331 |

The baseline model achieved moderate accuracy only because it predicted the majority class, but it completely failed to identify satisfied passengers. All trained machine learning models significantly outperformed the baseline.

The confusion matrices confirmed the metric-based evaluation: as model complexity increased, the number of incorrect classifications decreased. Among all tested models, **Random Forest achieved the best results**.

## 8. Summary

Based on the final evaluation, the **Random Forest** model was selected as the best model. It achieved the highest F1-score and the lowest number of incorrect classifications among the evaluated models.

The exploratory analysis identified key factors influencing passenger satisfaction, especially service-quality ratings and flight delays. The trained models substantially outperformed the baseline, confirming that machine learning methods are appropriate for this classification problem.

From a business perspective, this model can support airline passenger satisfaction analysis by identifying service areas that have the strongest impact on customer satisfaction and pointing to elements that may require improvement.

Further improvements could include:

- hyperparameter tuning
- additional explanatory variables
- testing other classification algorithms
- deeper class-imbalance analysis
- decision-threshold optimization

## 9. Project Structure

```text
.
+-- Raport z analizy i modelowania zbioru danych Airline Passenger Satisfaction.ipynb
+-- train.csv
+-- test.csv
+-- README.md
```

## 10. How to Run the Notebook

Install the required Python libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Then open and run the notebook:

```text
Raport z analizy i modelowania zbioru danych Airline Passenger Satisfaction.ipynb
```

The notebook expects `train.csv` and `test.csv` to be located in the project root directory.
