import pandas as pd

def get_demographic_parity(y_true, y_pred, data, num_sensitive):
    """
    Calculate demographic parity for different sensitive groups based on predictions.

    Args:
    - y_true (numpy array): True labels.
    - y_pred (numpy array): Binary predictions.
    - data (pandas DataFrame): DataFrame containing sensitive attributes.
    - num_sensitive (int): Number of sensitive attributes.

    Returns:
    - all_group_stats (dict): Dictionary with group statistics for each sensitive attribute.
    - all_demographic_parity (dict): Dictionary with demographic parity for each sensitive attribute.
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    }, index=data.index)
    
    for i in range(num_sensitive):
        df[f'S{i+1}'] = data[f'S{i+1}']

    sensitive_cols = [f'S{i+1}' for i in range(num_sensitive)]

    all_group_stats = {}
    all_demographic_parity = {}

    for sensitive_col in sensitive_cols:
        group_stats = df.groupby(sensitive_col).agg(
            total_predictions=('y_pred', 'size'),
            positive_predictions=('y_pred', 'sum')
        )
        group_stats['positive_prediction_rate'] = group_stats['positive_predictions'] / group_stats['total_predictions']
        demographic_parity = group_stats['positive_prediction_rate'].max() - group_stats['positive_prediction_rate'].min()

        all_group_stats[sensitive_col] = group_stats
        all_demographic_parity[sensitive_col] = demographic_parity

    return all_group_stats, all_demographic_parity