import xgboost as xgb

def train_xgb_classifier(X_train, y_train):
    """
    Trains an XGBoost classifier on the provided training data and returns the trained model.
    
    Parameters:
    X_train (array-like or DataFrame): The feature matrix for training data.
    y_train (array-like or Series): The target values for training data.
    
    Returns:
    xgb.XGBClassifier: The trained XGBoost classifier model.
    """
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    
    return model
