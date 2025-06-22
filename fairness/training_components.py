import torch.optim as optim
import torch.nn as nn

def initialize_training_components(encoder, predictor, adversaries, learning_rate=0.01):
    """
    Initialize the training components including loss functions and optimizers for the given models.

    Args:
    - encoder (torch.nn.Module): The encoder model.
    - predictor (torch.nn.Module): The predictor model.
    - adversaries (list of torch.nn.Module): List of adversary models.
    - learning_rate (float, optional): Learning rate for the optimizers. Default is 0.01.

    Returns:
    - criterion_enc (torch.nn.Module): Loss function for the encoder (Mean Squared Error).
    - criterion_pred (torch.nn.Module): Loss function for the predictor (Binary Cross-Entropy).
    - criterion_adv (list of torch.nn.Module): List of loss functions for the adversaries (Binary Cross-Entropy).
    - optimizer_enc (torch.optim.Optimizer): Optimizer for the encoder model.
    - optimizer_pred (torch.optim.Optimizer): Optimizer for the predictor model.
    - optimizer_adv (list of torch.optim.Optimizer): List of optimizers for the adversary models.
    """
    criterion_enc = nn.MSELoss()
    criterion_pred = nn.BCELoss()
    criterion_adv = [nn.BCELoss() for _ in range(len(adversaries))]
    
    optimizer_enc = optim.Adam(encoder.parameters(), lr=learning_rate)
    optimizer_pred = optim.Adam(predictor.parameters(), lr=learning_rate)
    optimizer_adv = [optim.Adam(adversaries[i].parameters(), lr=learning_rate) for i in range(len(adversaries))]
    
    return criterion_enc, criterion_pred, criterion_adv, optimizer_enc, optimizer_pred, optimizer_adv
