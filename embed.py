from typing import List

from transformers import AutoModel, AutoTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize the text
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return {key: val.squeeze(0) for key, val in inputs.items()}
class Embedder:
    def __init__(self, model_name_or_path:str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        if torch.cuda.is_available():
            print("GPU available")
            self.model.to("cuda")
    
    def embed(self, texts:List[str]):
        # Fill in code to generate embeddings
        # inputs = self.tokenizer(text, padding="longest", truncation=True, return_tensors="pt",max_length=512)
        dataset = TextDataset(texts, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        all_embeddings=[]
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to("cuda") for k, v in batch.items()}
                outputs = self.model(**batch)
                embeddings = outputs.last_hidden_state
                pooled_embeddings = embeddings.mean(dim=1)
                pooled_embeddings = pooled_embeddings.cpu().tolist()
                all_embeddings.extend(pooled_embeddings)
        return all_embeddings