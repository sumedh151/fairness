import pandas as pd
from sklearn.model_selection import train_test_split

def perform_train_test_split(data, num_non_sensitive, num_sensitive, test_size=0.2, random_state=None):
    """
    Splits the dataset into training and testing sets based on specified number of non-sensitive and sensitive feature columns.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the dataset with features and target variable.
    - num_non_sensitive (int): The number of non-sensitive feature columns to include in the split. 
      Assumes feature columns are named 'X1', 'X2', ..., 'Xn' where n is the number of non-sensitive features.
    - num_sensitive (int): The number of sensitive feature columns to include in the split.
      Assumes sensitive feature columns are named 'S1', 'S2', ..., 'Sn' where n is the number of sensitive features.
    - test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
    - random_state (int): Seed for the random number generator to ensure reproducibility (default is 42).

    Returns:
    - data_train (pd.DataFrame): Training set containing both features and target.
    - data_test (pd.DataFrame): Testing set containing both features and target.

    Notes:
    - The function first constructs the DataFrame with feature and target columns based on the given `num_non_sensitive` 
      and `num_sensitive` parameters, and then splits the data into training and testing sets.
    - Ensure that the `data` DataFrame includes columns with names conforming to the expected patterns 
      ('X1', 'X2', ..., 'Xn', 'S1', 'S2', ..., 'Sn', and 'Y').

    Example:
    >>> data = pd.DataFrame({
    ...     'X1': [0.807, -0.156, 0.482, 0.402, 0.970],
    ...     'X2': [0.550, -0.216, 0.105, -0.294, 0.453],
    ...     'X3': [0.144, -0.027, 0.047, 0.131, 0.358],
    ...     'X4': [0.449, 0.289, 0.391, 0.381, 0.316],
    ...     'S1': [1, 1, 1, 1, 1],
    ...     'Y': [1, 0, 1, 1, 1]
    ... })
    >>> data_train, data_test = perform_train_test_split(data, num_non_sensitive=4, num_sensitive=1)
    >>> print(data_train.shape, data_test.shape)
    """
    
    columns = [f'X{i+1}' for i in range(num_non_sensitive)] + [f'S{i+1}' for i in range(num_sensitive)] + ['Y']

    data_selected = data[columns]
    
    data_train, data_test = train_test_split(data_selected, test_size=test_size, random_state=random_state)
    
    return data_train, data_test