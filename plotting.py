import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def graph_with_multiple_x_values(
    X,
    y,
    feature_names=None,
    output_directory="plots"
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