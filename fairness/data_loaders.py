import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.X = torch.tensor(dataframe.drop(columns=['Y']).values, dtype=torch.float32)
        self.Y = torch.tensor(dataframe[['Y']].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    
def create_dataloaders(train_df, test_df, batch_size=256, shuffle_train=True):
    """
    Create DataLoader instances for training and testing datasets.

    Args:
    - train_df (pandas DataFrame): DataFrame containing training data. Should be compatible with the CustomDataset.
    - test_df (pandas DataFrame): DataFrame containing testing data. Should be compatible with the CustomDataset.
    - batch_size (int, optional): The number of samples per batch. Default is 256.
    - shuffle_train (bool, optional): Whether to shuffle the training data. Default is True. Shuffling is typically done for training to ensure better generalization.

    Returns:
    - train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    - test_loader (torch.utils.data.DataLoader): DataLoader for the testing dataset. Test data is typically not shuffled.
    """
    train_dataset = CustomDataset(train_df)
    test_dataset = CustomDataset(test_df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # Test data is typically not shuffled
    
    return train_loader, test_loader
