# Working with Missing Data in Business Analytics

This project demonstrates techniques for handling missing data in business analytics.

## Business context

Missing data is more than a technical inconvenience --- it's a window into how your data was collected, what went wrong, and what kind of assumptions you're willing to make. Before running any model, you need to ask: why is this value missing, and what does that tell me?

Every dataset has gaps. These missing values could come from user errors, system failures, skipped survey questions, or unreported transactions. The first step is to identify the *mechanism* behind the missingness. Statisticians usually describe this in three categories:

- Missing Completely at Random (MCAR): The missingness has no pattern or relationship with other data. This is the cleanest type to handle. - Missing at Random (MAR): The missingness depends on other observed variables. For instance, older customers might be less likely to complete online surveys. - Missing Not at Random (MNAR): The missingness depends on the unobserved value itself. For example, people with low incomes may choose not to report their income.

## Article

Medium article: [Working with Missing Data in Business Analytics](https://medium.com/@kylejones_47003/working-with-missing-data-in-business-analytics-74b1f5f0efd2)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Missing data analysis functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source (URL, file path, or synthetic generation)
- Imputation strategy (mean, median, iterative)
- Output settings

## Missing Data Strategies

Imputation methods:
- Mean/Median: Simple univariate imputation
- Iterative: Multivariate imputation using other features
- Group-based: Imputation within groups
- Indicator variables: Track missingness patterns

## Caveats

- By default, generates synthetic data with artificial missingness.
- Imputation strategy should match data characteristics.
- Missingness patterns may indicate data quality issues.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).