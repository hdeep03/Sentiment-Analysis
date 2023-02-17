from sentence_transformers import SentenceTransformer

class Embedder:
    """
    Abstract class for embedding text
        * output_shape: The shape of the output of the embedder
    """
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def embed(self, text):
        raise NotImplementedError
    
    def get_output_shape(self):
        return self.output_shape

class SentenceTransformerEmbedder(Embedder):
    """
    Embedder using SentenceTransformer
        * model_name: The name of the model to use
    Output shape is (384,)
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        super().__init__((384,))
        self.embedding_model = SentenceTransformer(model_name)

    def embed(self, text):
        """
        Embeds text using SentenceTransformer
            * text: The text to embed
            * returns: The embedding of the text w/ dimension (384,)
        """
        return self.embedding_model.encode(text)
