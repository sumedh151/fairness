from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_classifier(y_true, y_pred, y_logits, threshold=0.6):
    """
    Evaluate the classifier's performance based on true labels, predicted labels, and predicted logits.
    
    Args:
    - y_true (numpy array): True labels
    - y_pred (numpy array): Predicted class labels (binary)
    - y_logits (numpy array): Predicted logits from the model
    - threshold (float): Threshold for converting logits to class predictions (not used in this version)
    
    Returns:
    - metrics (dict): Dictionary containing accuracy, precision, recall, f1 score, and ROC AUC score
    """

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_logits)
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC Score': roc_auc
    }
    
    return metrics