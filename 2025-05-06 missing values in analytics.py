"""Generated from Jupyter notebook: 2025-05-06 missing values in analytics

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
from sklearn.impute import IterativeImputer, SimpleImputer


def main():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    )
    df.columns = ["Month", "Passengers"]
    df.loc[10:15, "Passengers"] = np.nan
    df["Revenue"] = df["Passengers"] * 10 + np.random.normal(0, 50, size=len(df))
    df.loc[5:10, "Revenue"] = np.nan
    msno.bar(df)
    plt.title("Missing Values per Column")
    plt.savefig("missing_bar.png")
    plt.show()
    msno.matrix(df)
    plt.title("Missingness Matrix")
    plt.savefig("missing_matrix.png")
    plt.show()
    msno.heatmap(df)
    plt.title("Missingness Correlation Heatmap")
    plt.savefig("missing_heatmap.png")
    plt.show()
    print("Missing values summary:\n", df.isnull().sum())
    df_dropped = df.dropna()
    print(f"After dropping, shape: {df_dropped.shape}")
    simple_imputer = SimpleImputer(strategy="mean")
    df["Passengers_mean"] = simple_imputer.fit_transform(df[["Passengers"]])
    df["Revenue_mean"] = simple_imputer.fit_transform(df[["Revenue"]])
    df["Month_num"] = pd.to_datetime(df["Month"]).dt.month
    df["Revenue_median_group"] = df.groupby("Month_num")["Revenue"].transform(
        lambda x: x.fillna(x.median())
    )
    multi_imputer = IterativeImputer(random_state=0)
    df[["Passengers_iter", "Revenue_iter"]] = multi_imputer.fit_transform(
        df[["Passengers", "Revenue"]]
    )
    df["Revenue_missing"] = df["Revenue"].isna().astype(int)
    df["Passengers_missing"] = df["Passengers"].isna().astype(int)
    print("\nImputed Data Preview:")
    print(
        df[
            [
                "Month",
                "Passengers",
                "Passengers_mean",
                "Passengers_iter",
                "Revenue",
                "Revenue_mean",
                "Revenue_median_group",
                "Revenue_iter",
                "Revenue_missing",
                "Passengers_missing",
            ]
        ].head(12)
    )
    df.to_csv("imputed_output.csv", index=False)


def main() -> None:
    main()


if __name__ == "__main__":
    main()
