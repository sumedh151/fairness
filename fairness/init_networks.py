def initialize_networks(encoder_class, predictor_class, adversary_class, num_non_sensitive, num_sensitive):
    """
    Initialize the Encoder, Predictor, and Adversaries for the model using specified classes.

    Args:
    - encoder_class (type): The class for the Encoder network.
    - predictor_class (type): The class for the Predictor network.
    - adversary_class (type): The class for the Adversary network.
    - num_non_sensitive (int): Number of non-sensitive features.
    - num_sensitive (int): Number of sensitive features.

    Returns:
    - encoder (nn.Module): Initialized Encoder network.
    - predictor (nn.Module): Initialized Predictor network.
    - adversaries (list of nn.Module): List of initialized Adversary networks.
    """
    encoder = encoder_class(input_dim=num_non_sensitive + num_sensitive)
    predictor = predictor_class(input_dim=num_non_sensitive)
    adversaries = [adversary_class(input_dim=num_non_sensitive) for _ in range(num_sensitive)]
    return encoder, predictor, adversaries
