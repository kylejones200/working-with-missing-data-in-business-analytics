---
author: "Kyle Jones"
date_published: "May 4, 2025"
date_exported_from_medium: "November 10, 2025"
canonical_link: "https://medium.com/@kyle-t-jones/working-with-missing-data-in-business-analytics-74b1f5f0efd2"
---

# Working with Missing Data in Business Analytics Missing data is more than a technical inconvenience --- it's a window
into how your data was collected, what went wrong, and what kind of...

### Working with Missing Data in Business Analytics
Missing data is more than a technical inconvenience --- it's a window into how your data was collected, what went wrong, and what kind of assumptions you're willing to make. Before running any model, you need to ask: why is this value missing, and what does that tell me?

Every dataset has gaps. These missing values could come from user errors, system failures, skipped survey questions, or unreported transactions. The first step is to identify the *mechanism* behind the missingness. Statisticians usually describe this in three categories:

- Missing Completely at Random (MCAR): The missingness has no pattern or relationship with other data. This is the cleanest type to handle.
- Missing at Random (MAR): The missingness depends on other observed variables. For instance, older customers might be less likely to complete online surveys.
- Missing Not at Random (MNAR): The missingness depends on the unobserved value itself. For example, people with low incomes may choose not to report their income.

Knowing which of these applies isn't always obvious. But it matters. If you assume data is missing at random when it's not, your imputations can be biased and introduce misleading patterns.

The Python library `missingno` offers simple but effective visualizations. Here's how to explore a dataset for missing values:

```python
# pip install missingno

import pandas as pd
import missingno as msno

df = pd.read_csv('your_dataset.csv')
# Simple bar chart showing how many missing values per column
msno.bar(df)
# Matrix showing the pattern of missingness across records
msno.matrix(df)
# Heatmap showing correlations in missingness between columns
msno.heatmap(df)
```


<figcaption>Example of a dataset with some missing values</figcaption>


These plots help you see whether some columns are always missing together, or whether missingness is concentrated in certain rows. This informs your decision about whether to drop, impute, or flag the data.

### Strategies for Handling Missing Values
Once you've identified the structure of missingness, you can decide how to handle it. There's no universal solution, but these are the common techniques:

#### a. Dropping Rows or Columns
If the missingness is small in scale or appears completely at random, you might drop rows or columns:

``` 
df.dropna(inplace=True)
```

This is a blunt instrument. Use it only when the data you're dropping is negligible or not critical to your analysis.

#### b. Simple Imputation
You can replace missing values with a constant (like 0 or -1), the mean, median, or mode of the column:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
df['revenue'] = imputer.fit_transform(df[['revenue']])
```

This works well when the variable is approximately symmetric and the missingness is random. If the distribution is skewed, use the median instead.

#### c. Group-Based Imputation
You can impute based on grouped characteristics. For example, fill missing revenue values with the median revenue from the same product category:

``` 
df['revenue'] = df.groupby('product_category')['revenue'].transform(lambda x: x.fillna(x.median()))
```

This preserves within-group patterns and is better aligned with business structure.

#### d. Multivariate Imputation
Sometimes, you can infer a missing value based on other variables using models. Scikit-learn's `IterativeImputer` uses chained regression:

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imp = IterativeImputer()
df_imputed = pd.DataFrame(imp.fit_transform(df), columns=df.columns)
```

This allows you to impute revenue using relationships with quantity sold, customer rating, or time of year. It's slower but more statistically principled.

#### e. Indicator Variables
Another approach is to flag where data is missing:

``` 
df['revenue_missing'] = df['revenue'].isna().astype(int)
```

This binary flag can be included in your model to capture the signal from the missingness itself.

Imputation isn't just about filling gaps. It changes the behavior of your model. Suppose you're running a customer churn prediction, and you fill missing income with the mean. That imputes low-income customers with an artificially high value, making your model overly optimistic about retention.

Here's what's at stake:

- Bias: Poor imputation can push your coefficients in the wrong direction.
- Variance: Dropping too many rows can shrink your dataset and reduce predictive power.
- Interpretability: Some imputation methods (like multivariate ones) obscure what values were real and what were guessed.
- Business insight: The fact that a value was missing may tell you something important. Never assume the absence of data is meaningless.

Missing data forces you to make assumptions. Visualizing it helps you understand the structure of what's gone. Imputing values makes those assumptions explicit. Don't treat missingness as a nuisance. It's a clue, and your handling of it shapes every step downstream in your analytics workflow.

missing_data_analysis.py

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(42)

def save_fig(path: str):
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def plot_missingness(df: pd.DataFrame):
    msno.bar(df)
    plt.title("Missing Values per Column")
    save_fig("missing_bar.png")
    msno.matrix(df)
    plt.title("Missingness Matrix")
    save_fig("missing_matrix.png")
    msno.heatmap(df)
    plt.title("Missingness Correlation Heatmap")
    save_fig("missing_heatmap.png")

def run_leakage_free_cv(df_in: pd.DataFrame):
    df_eval = df_in.copy()
    df_eval['Month_num'] = pd.to_datetime(df_eval['Month']).dt.month
    X_all = df_eval[['Passengers', 'Month_num']].values
    y_all = df_eval['Revenue'].values
    tscv = TimeSeriesSplit(n_splits=5)
    simple_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy='mean')),
        ("model", LinearRegression()),
    ])
    iter_pipe = Pipeline([
        ("imputer", IterativeImputer(random_state=0)) ,
        ("model", LinearRegression()),
    ])
    results = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all), start=1):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        train_mask = ~np.isnan(y_train)
        test_mask = ~np.isnan(y_test)
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        X_test, y_test = X_test[test_mask], y_test[test_mask]
        simple_pipe.fit(X_train, y_train)
        y_pred_simple = simple_pipe.predict(X_test)
        results.append({'fold': fold, 'method': 'SimpleImputer+LR', 'mae': float(mean_absolute_error(y_test, y_pred_simple)), 'r2': float(r2_score(y_test, y_pred_simple))})
        iter_pipe.fit(X_train, y_train)
        y_pred_iter = iter_pipe.predict(X_test)
        results.append({'fold': fold, 'method': 'IterativeImputer+LR', 'mae': float(mean_absolute_error(y_test, y_pred_iter)), 'r2': float(r2_score(y_test, y_pred_iter))})
    return results

def summarize(results, method: str):
    rows = [r for r in results if r['method'] == method]
    return {
        'method': method,
        'mae_mean': float(np.mean([r['mae'] for r in rows])),
        'mae_std': float(np.std([r['mae'] for r in rows])),
        'r2_mean': float(np.mean([r['r2'] for r in rows])),
        'r2_std': float(np.std([r['r2'] for r in rows])),
    }

def print_summary(results):
    print("\nSummary:")
    print(summarize(results, 'SimpleImputer+LR'))
    print(summarize(results, 'IterativeImputer+LR'))

def print_results(results):
    print("\nLeakage-free CV results:")
    for row in results:
        print(row)

def main():
    df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv")
    df.columns = ['Month', 'Passengers']  # Rename for clarity
    df.loc[10:15, 'Passengers'] = np.nan
    df['Revenue'] = df['Passengers'] * 10 + np.random.normal(0, 50, size=len(df))
    df.loc[5:10, 'Revenue'] = np.nan
    plot_missingness(df)
    print("Missing values summary:\n", df.isnull().sum())
    df_dropped = df.dropna()
    print(f"After dropping, shape: {df_dropped.shape}")
    simple_imputer = SimpleImputer(strategy='mean')
    df['Passengers_mean'] = simple_imputer.fit_transform(df[['Passengers']])
    df['Revenue_mean'] = simple_imputer.fit_transform(df[['Revenue']])
    df['Month_num'] = pd.to_datetime(df['Month']).dt.month
    df['Revenue_median_group'] = df.groupby('Month_num')['Revenue'].transform(lambda x: x.fillna(x.median()))
    multi_imputer = IterativeImputer(random_state=0)
    df[['Passengers_iter', 'Revenue_iter']] = multi_imputer.fit_transform(df[['Passengers', 'Revenue']])
    df['Revenue_missing'] = df['Revenue'].isna().astype(int)
    df['Passengers_missing'] = df['Passengers'].isna().astype(int)
    print("\nImputed Data Preview:")
    print(df[['Month', 'Passengers', 'Passengers_mean', 'Passengers_iter',
              'Revenue', 'Revenue_mean', 'Revenue_median_group', 'Revenue_iter',
              'Revenue_missing', 'Passengers_missing']].head(12))
    df.to_csv("imputed_output.csv", index=False)
    results = run_leakage_free_cv(df)
    print_results(results)
    print_summary(results)

if __name__ == "__main__":
    main()
```

This script:

- Injects missing values for demonstration
- Visualizes the missing patterns
- Applies five different strategies for handling missingness
- Outputs an enriched DataFrame with imputed columns and missingness flags
