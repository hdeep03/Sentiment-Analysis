from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch

class BatchDataLoader():
    """
    Base class for data loaders
        * split: float between 0 and 1, the proportion of the dataset to include in the train split
        * batch_size: int, the size of the batches to be returned by the dataloader
    """
    def __init__(self, split, batch_size):
        self.split = split
        self.batch_size = batch_size
    def get_train_data(self):
        raise NotImplementedError
    def get_test_data(self):
        raise NotImplementedError

class CSVDataLoader(BatchDataLoader):
    """
    Data loader for CSV files
        * file_path: str, path to the CSV file
        * split: float between 0 and 1, the proportion of the dataset to include in the train split
        * embedder: Embedder object, the embedder to use to embed the text
        * batch_size: int, the size of the batches to be returned by the dataloader
    """
    def __init__(self, file_path, split, embedder, batch_size=64):
        super().__init__(split, batch_size)
        dataset = pd.read_csv(file_path, index_col=0)
        X = [embedder.embed(x) for x in tqdm(dataset['text'])]
        y = np.array([(1 if (x=='positive') else 0) for x in tqdm(dataset['airline_sentiment'])])
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(np.array(X), y, test_size=split, random_state=0)

    def get_train_dataloader(self):
        """
        Returns a DataLoader with the training loader
        """
        train_data=TensorDataset(torch.FloatTensor(self.X_train), torch.FloatTensor(self.y_train))
        train_loader=DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        return train_loader
    
    def get_test_dataloader(self):
        """
        Returns a DataLoader with the testing loader
        """
        test_data=TensorDataset(torch.FloatTensor(self.X_test), torch.FloatTensor(self.y_test))
        test_loader=DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
        return test_loader
    
