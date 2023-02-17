from torch import nn, optim
import torch
from sklearn.metrics import roc_auc_score

class Handler():
    """
    Handler which handles the training and evaluation for a model
    """
    def __init__(self, model):
        """
        Initialize Trainer for the model. 
            * model: nn.Module model to be trained
            * train_data_loader: DataLoader for data to be used
                                on training
            * test_data_loader: DataLoader with data to be used
                                on testing 
        """
        self.model = model

    
    def train(self, train_data_loader, num_epochs, loss_fn = nn.BCELoss()):
        """
        Train model on data from DataLoader
            * train_data_Loader: DataLoader with training samples
            * num_epochs: Number of epochs to train the model
            * loss_fn: The loss function to use during training
        
        Modifies model weights in-place. Returns training loss at each epoch
        """
        self.model.train()
        optimizer = optim.Adam(self.model.parameters())
        loss_values = []
        for epoch in range(num_epochs):
            for X, y in train_data_loader:
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                pred = self.model(X)

                # compute loss
                loss = loss_fn(pred, y.unsqueeze(-1))
                loss_values.append(loss.item())

                # backward + optimize
                loss.backward()
                optimizer.step()
        return loss_values

    def evaluate(self, test_data_loader, loss_fn = nn.BCELoss()):
        """
        Evaluate model on data from DataLoader
            * loss_fn: The loss function to use during evaluation
        
        Returns loss and accuracy
        """
        self.model.eval()
        correct = 0
        total = 0
        loss_values = []
        with torch.no_grad():
            for X, y in test_data_loader:
                pred = self.model(X)
                loss = loss_fn(pred, y.unsqueeze(-1))
                loss_values.append(loss.item())
                pred = torch.round(pred)
                total += y.size(0)
                correct += (pred == y.unsqueeze(-1)).sum().item()
        return sum(loss_values), correct/total

    def predict(self, X, embedding_fn = None):
        """
        Predict on X
            *  X: torch.Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        self.model.eval()
        with torch.no_grad():
            if embedding_fn:
                X = torch.from_numpy(embedding_fn(X))
            raw_pred = self.model(X)
            pred = torch.round(raw_pred)
        return raw_pred, pred

    def auroc(self, test_data_loader):
        """
        Compute AUROC on test data
        
        Returns AUROC
        """
        data_loader = test_data_loader
        self.model.eval()
        with torch.no_grad():
            y_pred = []
            y_true = []
            for X, y in data_loader:
                raw_pred = self.model(X)
                pred = torch.round(raw_pred)
                y_pred.extend(raw_pred.squeeze().tolist())
                y_true.extend(y.tolist())
        return roc_auc_score(y_true, y_pred)

    def save(self, path):
        """
        Save model to path
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """
        Load model from path
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval() # set model to evaluation mode



        





