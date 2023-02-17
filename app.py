from fastapi import FastAPI
import pydantic
import uvicorn
from models.neuralnetwork import NeuralNetwork
from util.handlers import Handler
from models.embedders import SentenceTransformerEmbedder
import argparse


class Input(pydantic.BaseModel):
    """
    Type model for the sentiment analysis inference api
    """
    text: str

class Response(pydantic.BaseModel):
    """
    Type model for the sentiment analysis inference api return
    """
    sentiment: str


def main(filepath, port, host, hidden_size):
    """
    Starts a fastapi server for sentiment analysis
        * filepath: path to model
        * port: port to run server on
        * host: host to run server on
        * hidden_size: hidden layer size
    """
    app = FastAPI(
                    title="Sentiment Analyzer",
                    description="Analyzes the sentiment of a given text",
                    version="0.0.1",)
    model = NeuralNetwork(384, 50)
    handler = Handler(model)
    try:
        handler.load('./saved_models/model.pt')
    except:
        print("No model found, please train one first")
        exit(1)
    @app.post("/api/v0/inference", response_model=Response, responses={200: {"description": "Sentiment of text"}, 400: {"description": "Invalid input"}, 500: {"description": "Internal server error"}})
    def predict(Input: Input):
        text = Input.text
        pred = handler.predict(text, embedding_fn=SentenceTransformerEmbedder().embed)
        return {"sentiment": ('positive' if pred[1] == 1 else 'negative')}
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    """
    Usage: python app.py -f <path to model> -p <port> -o <host> -s <hidden layer size>
    Starts a fastapi server for sentiment analysis
    """
    parser = argparse.ArgumentParser(
                    prog = 'Start API for sentiment analysis',
                    description = 'A fastapi sentiment analysis API')
    parser.add_argument('-f', '--model_path', default='./saved_models/model.pt') # path to model
    parser.add_argument('-p', '--port', default="8000")                        # port to run on
    parser.add_argument('-o', '--host', default='127.0.0.1')                   # host to run on
    parser.add_argument('-s', '--hidden_size', default="50")                   # hidden size for model
    args = parser.parse_args()
    main(args.model_path, int(args.port), args.host, int(args.hidden_size))




    







