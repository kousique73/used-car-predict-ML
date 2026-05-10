# Used Car Price Prediction using Machine Learning

## Overview

This project aims to predict the prices of used cars based on various features such as the car's age, mileage, engine size, brand, and more. It utilizes a Machine Learning pipeline to clean the data, explore patterns, engineer new features, and build a robust predictive model.

The project is structured into three main phases, each detailed in its respective directory:
1. **Exploratory Data Analysis (EDA)**
2. **Feature Engineering**
3. **Model Building**

---

## Repository Structure

```
├── data/                       # Contains the original and cleaned CSV datasets
├── Exploratory Data Analysis/  # Notebooks and documentation for EDA
├── Feature Engineering/        # Notebooks and documentation for feature transformations
├── Model Building/             # Notebooks and documentation for training the ML model
├── EDA, Feature Engineering.ipynb  # Combined notebook for preprocessing
├── Final.ipynb                 # Final consolidated execution notebook
└── README.md                   # Project overview
```

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
**Directory: `Exploratory Data Analysis/`**

In this phase, we dive deep into the data to understand the underlying distributions and relationships:
- **Univariate Analysis**: Histograms and boxplots to visualize distributions (e.g., `car_age`, `engine`, `price`).
- **Bivariate Analysis**: Examining the relationship between predictor variables and the target (`price`).
- **Outlier Detection**: Using IQR boundaries to identify and understand extreme values.
- **Correlation**: Heatmaps to identify highly correlated numerical features.

### 2. Feature Engineering
**Directory: `Feature Engineering/`**

Data is transformed and prepared for modeling:
- **Feature Extraction**: Splitting the raw `name` column into `car_make`, `car_model`, and `car_spec`.
- **Derived Features**: Calculating `car_age` from the `year` of manufacture.
- **Data Standardization**: Standardizing the `mileage` values (handling both `kmpl` and `km/kg`).
- **Encoding**: Applying One-Hot Encoding to categorical features (`fuel_type`, `transmission`, `owner_type`, etc.).
- **Outlier Treatment**: Winsorizing (capping) extreme outliers to the upper and lower whiskers to reduce noise.

### 3. Model Building
**Directory: `Model Building/`**

We build and evaluate the predictive models:
- **Algorithm**: Ordinary Least Squares (OLS) Linear Regression using `statsmodels`.
- **Transformation**: Applying log transformations to highly skewed features like `engine`, `kilometers_driven`, and `price`.
- **Model Evaluation**: Using metrics like R-squared ($R^2$), RMSE (Root Mean Squared Error), and MAE (Mean Absolute Error). The OLS model achieved an excellent Adjusted $R^2$ score (~0.907).
- **Assumptions Testing**: 
  - Validating Variance Inflation Factor (VIF) scores for multicollinearity.
  - Checking for heteroscedasticity and normality of errors.

---

## Getting Started

### Prerequisites
To run the notebooks locally, ensure you have Python installed along with the following libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`
- `scikit-learn`
- `statsmodels`
- `jupyter`

### Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/kousique73/used-car-predict-ML.git
cd used-car-predict-ML
pip install -r requirements.txt # (If a requirements.txt is provided, otherwise install libraries manually)
```

### Usage
1. Start the Jupyter Notebook server:
   ```bash
   jupyter notebook
   ```
2. Navigate to the `Final.ipynb` notebook to see the complete end-to-end pipeline.
3. You can also dive into the specific phases by exploring the notebooks located within the `Exploratory Data Analysis`, `Feature Engineering`, and `Model Building` folders.

## Results
The trained OLS regression model demonstrated strong predictive capabilities, capturing the majority of variance in used car prices based on features like brand category (e.g., Luxury vs Mid-Range), transmission type, car age, and engine capacity.

---