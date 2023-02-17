from data_loader.dataloader import CSVDataLoader
from models.embedders import SentenceTransformerEmbedder
from models.neuralnetwork import NeuralNetwork
from util.handlers import Handler
import argparse

def train(path, hidden_size, split, epochs):
    """
    Trains a model for sentiment analysis
        * path: path to csv file
        * hidden_size: hidden layer size
        * split: fraction of data to use for training
        * epochs: number of epochs to train for
    """
    data_loader = CSVDataLoader(path, 0.8, embedder=SentenceTransformerEmbedder())
    train_loader = data_loader.get_train_dataloader()
    test_loader = data_loader.get_test_dataloader()
    model = NeuralNetwork(384, hidden_size)
    handler = Handler(model)
    handler.train(train_loader, epochs)
    print("Accuracy: {}".format(handler.evaluate(test_loader)[1]))
    print(f"AUROC: {handler.auroc(test_loader)}")
    handler.save('./model.pt')


if __name__ == '__main__':
    """
    Usage: python train.py <path to csv file> <fraction of data to use for training> <hidden layer size> <num epochs>
    """
    parser = argparse.ArgumentParser(
                    prog = 'Sentiment Analysis Trainer',
                    description = 'Trains a model for sentiment analysis')
    parser.add_argument('filename')           # positional argument
    parser.add_argument('fraction_train')     # Fraction of data to use for training
    parser.add_argument('hidden_size')        # hidden layer size
    parser.add_argument('epochs')             # num epochs
    args = parser.parse_args()
    train(args.filename, int(args.hidden_size), float(args.fraction_train), int(args.epochs))
