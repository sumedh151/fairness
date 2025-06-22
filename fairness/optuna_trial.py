from fairness.data_simulation import DataGenerator
from fairness.split_data import perform_train_test_split
from fairness.models import Encoder, Predictor, Adversary
from fairness.init_networks import initialize_networks
from fairness.data_loaders import create_dataloaders
from fairness.training_components import initialize_training_components
from fairness.train import train_model
from fairness.plots import plot_lines,plot_comparison_heatmaps
from fairness.test import get_transformed_features, get_predictions
from fairness.eval import evaluate_classifier
from fairness.fairness_metrics import get_demographic_parity

num_non_sensitive = 4
num_sensitive = 1
generator = DataGenerator(num_non_sensitive = num_non_sensitive, num_sensitive = num_sensitive, seed=None)
data = generator.generate_simulated_data()
data_train, data_test = perform_train_test_split(data, num_non_sensitive=num_non_sensitive, num_sensitive=num_sensitive, test_size=0.15)

def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    alpha = trial.suggest_float('alpha', 1, 50)
    max_norm = trial.suggest_float('max_norm', 0.1, 5.0)

    encoder, predictor, adversaries = initialize_networks(Encoder, Predictor, Adversary, num_non_sensitive=num_non_sensitive, num_sensitive=num_sensitive)
    criterion_enc, criterion_pred, criterion_adversaries, optimizer_enc, optimizer_pred, optimizer_adversaries = initialize_training_components(encoder, predictor, adversaries, learning_rate=learning_rate)
    train_loader, test_loader = create_dataloaders(data_train, data_test, batch_size=256, shuffle_train=True)
    
    num_epochs = 500
    encoder, predictor, adversaries, curr_epochs_total, curr_gradient_norms_enc, 
    curr_gradient_norms_pred, curr_gradient_norms_adversaries, curr_loss_enc, 
    curr_loss_pred, curr_loss_adversaries, curr_loss_comb = train_model(encoder=encoder, predictor=predictor, adversaries=adversaries,
                                                                        num_epochs=num_epochs, num_non_sensitive=num_non_sensitive,
                                                                        num_sensitive=num_sensitive, train_dataloader=train_loader,
                                                                        criterion_enc=criterion_enc, criterion_pred=criterion_pred,
                                                                        criterion_adversaries=criterion_adversaries,
                                                                        optimizer_enc=optimizer_enc, optimizer_pred=optimizer_pred,
                                                                        optimizer_adversaries=optimizer_adversaries, alpha=alpha,
                                                                        max_norm=max_norm)

    validation_loss = get_validation_loss(valid_loader, num_non_sensitive, num_sensitive, encoder, predictor, adversaries, 
                                          alpha, criterion_enc, criterion_pred, criterion_adversaries)
    return validation_loss