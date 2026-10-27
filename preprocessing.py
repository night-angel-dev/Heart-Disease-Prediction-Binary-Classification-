"""
preprocessing.py
Contains transformations applied before modeling.

"""


import pandas as pd


def make_binary_target(target: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the Cleveland heart-disease severity values to 0/1 labels, 
    needed for binary classification problems.

    The Cleveland target uses 0 for no disease and 1-4 for disease
    severity. The returned DataFrame keeps one target column so it can be
    passed to the existing plotting and modeling code.
    
    
    @Param target - Takes in a DataFrame object
    @returns a DataFrame object of only 0s and 1's. 
    """
    
    # y (target) should be 1 column
    if target.shape[1] != 1:
        raise ValueError("The target must contain exactly one column")

    # Extracts single column by pos
    target_values = target.iloc[:, 0]
    
    # Refuse lables NaN
    if target_values.isna().any():
        raise ValueError("The target contains missing values")

    # Keep anyting not 0-4
    unexpected_values = set(target_values.unique()) - {0, 1, 2, 3, 4}
    
    # If unexpected values present
    if unexpected_values:
        raise ValueError(
            f"Unexpected target value(s): {sorted(unexpected_values)}"
        )

    # Return Data frame of 0 and 1's via comparison. 
    return (target_values > 0).astype("int64").to_frame(name="heart_disease")
