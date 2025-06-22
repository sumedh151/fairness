import torch
import pandas as pd

def get_transformed_features(test_dataloader, data_test, num_non_sensitive, num_sensitive, encoder):
    """
    Get a DataFrame combining the transformed features from the encoder and the test data.

    This function evaluates the encoder on the test data, creates a DataFrame from the encoder outputs,
    combines it with the test data, and returns the resulting DataFrame.

    Parameters:
    - test_dataloader (DataLoader): DataLoader instance for the test dataset.
    - data_test (DataFrame): The test dataset containing sensitive and outcome variables.
    - num_non_sensitive (int): The number of non-sensitive features in the encoder outputs.
    - num_sensitive (int): The number of sensitive features in the test data.
    - encoder (nn.Module): The encoder model to be evaluated.

    Returns:
    - pd.DataFrame: A DataFrame combining encoder outputs and test data.
    """
    
    encoder_outputs = []
    encoder.eval()
    
    with torch.no_grad():
        for batch_X, _ in test_dataloader:
            output = encoder(batch_X)
            encoder_outputs.append(output)
    
    encoder_outputs = torch.cat(encoder_outputs, dim=0)
    encoder_outputs_np = encoder_outputs.numpy()
    
    df_X = pd.DataFrame(encoder_outputs_np, columns=[f'X{i+1}' for i in range(num_non_sensitive)])    
    df_sens_Y = data_test.copy()[[f'S{i+1}' for i in range(num_sensitive)] + ['Y']].reset_index(drop=True)
    df_combined = pd.concat([df_X, df_sens_Y], axis=1)
    return df_combined


def get_predictions(data_test, num_non_sensitive, num_sensitive, encoder, predictor, threshold=0.6):
    """
    Get predictions for a given test dataset using a specified encoder and predictor model.
    
    Args:
    - data_test (pandas DataFrame): Test data containing feature columns and true labels.
    - num_non_sensitive (int): Number of non-sensitive features.
    - num_sensitive (int): Number of sensitive features.
    - encoder (torch.nn.Module): Encoder model to transform input data.
    - predictor (torch.nn.Module): Predictor model to get logits from the encoded data.
    - threshold (float): Threshold for converting logits to binary predictions. Default is 0.6.
    
    Returns:
    - y_true (numpy array): True labels from the test dataset.
    - y_logits (numpy array): Predicted logits from the model.
    - y_pred (numpy array): Binary predictions based on the logits and threshold.
    """
    # Extract true labels
    y_true = data_test['Y'].values
    
    # Extract feature columns based on the number of non-sensitive and sensitive features
    feature_cols = [f'X{i+1}' for i in range(num_non_sensitive)] + [f'S{i+1}' for i in range(num_sensitive)]
    features = data_test[feature_cols].values
    
    # Convert features to tensor and pass through encoder and predictor
    with torch.no_grad():
        features_tensor = torch.tensor(features, dtype=torch.float32)
        encoded_features = encoder(features_tensor)
        y_logits = predictor(encoded_features).detach().numpy().reshape(-1)
    
    # Convert logits to binary predictions based on the threshold
    y_pred = (y_logits > threshold).astype(int)
    
    return y_true, y_logits, y_pred