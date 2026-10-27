"""
eda_plots.py - Originally was named plotting.py, renamed appropriately. 
File is responsible for creating and saving exploratory data visualizations.
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


DEFAULT_CONTINUOUS_FEATURES = {
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
}


def create_eda_plots(
    X,
    y,
    output_directory="plots",
    continuous_features=None,
    categorical_features=None,
):
    """
    Creates standard exploratory data analysis plots.

    Continuous features receive histograms and box plots grouped by target.
    Categorical features receive grouped bar plots. A correlation heatmap and
    target-count bar plot are also saved.
    """
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows")

    feature_names = list(X.columns)
    if continuous_features is None:
        continuous_features = [
            name for name in feature_names
            if name in DEFAULT_CONTINUOUS_FEATURES
        ]
    else:
        continuous_features = list(continuous_features)

    if categorical_features is None:
        categorical_features = [
            name for name in feature_names
            if name not in continuous_features
        ]
    else:
        categorical_features = list(categorical_features)

    unknown_features = (
        set(continuous_features + categorical_features) - set(feature_names)
    )
    if unknown_features:
        raise ValueError(
            f"Unknown feature(s): {sorted(unknown_features)}"
        )

    if set(continuous_features) & set(categorical_features):
        raise ValueError(
            "A feature cannot be both continuous and categorical"
        )

    output_directory = Path(output_directory)
    histogram_directory = output_directory / "histograms"
    boxplot_directory = output_directory / "boxplots"
    categorical_directory = output_directory / "categorical"
    for directory in (
        histogram_directory,
        boxplot_directory,
        categorical_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    target_series = pd.Series(
        np.asarray(y).reshape(-1),
        index=X.index,
        name=y.columns[0] if hasattr(y, "columns") else "Target",
    )

    # Histograms for continuous features.
    for feature_name in continuous_features:
        values = pd.to_numeric(X[feature_name], errors="coerce")
        figure, axis = plt.subplots(figsize=(9, 6))
        axis.hist(values.dropna(), bins=15, edgecolor="black")
        axis.set_xlabel(feature_name)
        axis.set_ylabel("Count")
        axis.set_title(f"Distribution of {feature_name}")
        axis.grid(axis="y", alpha=0.3)
        figure.tight_layout()
        figure.savefig(
            histogram_directory / f"{feature_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)

    # Box plots for continuous features, separated by target class.
    for feature_name in continuous_features:
        values = pd.to_numeric(X[feature_name], errors="coerce")
        grouped_values = [
            values[target_series == target_value].dropna()
            for target_value in sorted(target_series.dropna().unique())
        ]
        target_values = sorted(target_series.dropna().unique())

        figure, axis = plt.subplots(figsize=(9, 6))
        axis.boxplot(grouped_values, tick_labels=target_values)
        axis.set_xlabel(target_series.name)
        axis.set_ylabel(feature_name)
        axis.set_title(f"{feature_name} by {target_series.name}")
        axis.grid(axis="y", alpha=0.3)
        figure.tight_layout()
        figure.savefig(
            boxplot_directory / f"{feature_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)

    # Grouped bar plots for categorical features.
    for feature_name in categorical_features:
        counts = pd.crosstab(X[feature_name], target_series)
        axis = counts.plot(kind="bar", figsize=(9, 6))
        axis.set_xlabel(feature_name)
        axis.set_ylabel("Count")
        axis.set_title(f"{feature_name} by {target_series.name}")
        axis.legend(title=target_series.name)
        axis.grid(axis="y", alpha=0.3)
        figure = axis.get_figure()
        figure.tight_layout()
        figure.savefig(
            categorical_directory / f"{feature_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)

    # Correlation heatmap for numeric features and the target.
    correlation_data = pd.concat(
        [X.apply(pd.to_numeric, errors="coerce"), target_series],
        axis=1,
    )
    correlations = correlation_data.corr()
    figure, axis = plt.subplots(
        figsize=(max(10, len(correlations.columns) * 0.8), 8)
    )
    image = axis.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlations.columns)))
    axis.set_yticks(range(len(correlations.columns)))
    axis.set_xticklabels(correlations.columns, rotation=45, ha="right")
    axis.set_yticklabels(correlations.columns)
    figure.colorbar(image, ax=axis, label="Correlation")
    axis.set_title("Feature and Target Correlations")
    figure.tight_layout()
    figure.savefig(
        output_directory / "correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    # Target class counts.
    target_counts = target_series.value_counts(dropna=False).sort_index()
    figure, axis = plt.subplots(figsize=(8, 6))
    target_counts.plot(kind="bar", ax=axis)
    axis.set_xlabel(target_series.name)
    axis.set_ylabel("Count")
    axis.set_title("Target Class Distribution")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(
        output_directory / "target_counts.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def graph_with_multiple_x_values(
    X,
    y,
    feature_names=None,
    output_directory="plots",
    continuous_features=None,
    categorical_features=None,
):
    """
    Creates cumulative and individual plots for features of X
    
    @Param X - feature data, stored as pandas dataframe
    @Param y - target data, also stored as pandas dataframe
    @Param feature_names - names for columns in X
    @Param output_directory - folder where graph folders and plots will be created
    
    """
    
    
    if feature_names is None:
        feature_names = list(X.columns)

    if len(feature_names) != X.shape[1]:
        raise ValueError(
            "There must be one feature name for each X column"
        )

    output_directory = Path(output_directory)
    cumulative_directory = output_directory / "cumulative"
    individual_directory = output_directory / "individual"
    cumulative_directory.mkdir(parents=True, exist_ok=True)
    individual_directory.mkdir(parents=True, exist_ok=True)

    target = np.asarray(y).reshape(-1)
    target_name = y.columns[0] if hasattr(y, "columns") else "Target"
    number_of_features = X.shape[1]

    # Save one graph containing the first 1, 2, ..., n features.
    for feature_count in range(1, number_of_features + 1):
        figure, axis = plt.subplots(figsize=(12, 6))

        for feature_index in range(feature_count):
            feature_values = X.iloc[:, feature_index]
            axis.scatter(
                feature_values,
                target,
                alpha=0.6,
                label=feature_names[feature_index]
            )

        axis.set_xlabel("Feature value")
        axis.set_ylabel(target_name)
        axis.set_title(
            f"First {feature_count} feature(s) vs. {target_name}"
        )
        axis.legend()
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(
            cumulative_directory / f"{feature_count}FeatureGraph.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close(figure)

    # Save one separate feature-vs-target graph for every feature. 
    for feature_index, feature_name in enumerate(feature_names):
        figure, axis = plt.subplots(figsize=(9, 6))
        axis.scatter(
            X.iloc[:, feature_index],
            target,
            alpha=0.6
        )
        axis.set_xlabel(feature_name)
        axis.set_ylabel(target_name)
        axis.set_title(f"{feature_name} vs. {target_name}")
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(
            individual_directory / f"{feature_index + 1}_{feature_name}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close(figure)

    create_eda_plots(
        X,
        y,
        output_directory=output_directory,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
    )
