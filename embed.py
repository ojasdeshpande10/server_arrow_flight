from typing import List

from transformers import AutoModel, AutoTokenizer
import torch

class Embedder:
    def __init__(self, model_name_or_path:str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)

        if torch.cuda.is_available():
            print("GPU available")
            self.model.to("cuda")
    
    def embed(self, text:List[str]):
        # Fill in code to generate embeddings
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt",max_length=512)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state

        pooled_embeddings = embeddings.mean(dim=1)

        pooled_embeddings = pooled_embeddings.cpu().tolist()

        return pooled_embeddings