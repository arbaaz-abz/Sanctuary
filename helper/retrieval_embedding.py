import gc
import torch
from sentence_transformers import SentenceTransformer
from helper.config import RETRIEVAL_MODEL_NAME, RETRIEVAL_BATCH_SIZE

class Retrieval_Embed:
    def __init__(self, device=None, show_progress=False, batch_size=RETRIEVAL_BATCH_SIZE):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using:",RETRIEVAL_MODEL_NAME)
        self.device = device
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.model = SentenceTransformer(
            RETRIEVAL_MODEL_NAME,
            trust_remote_code=True,
            model_kwargs={'attn_implementation': 'eager'}
        ).to(self.device)
    
    def clear_gpu_cache(self):
        gc.collect()
        torch.cuda.empty_cache()
    
    def encode_documents(self, text_list):
        self.clear_gpu_cache()
        doc_embeddings = self.model.encode(
            text_list, 
            batch_size=self.batch_size, 
            show_progress_bar=self.show_progress,
            normalize_embeddings=True,
        )
        return doc_embeddings
                                        
    def encode_queries(self, queries):
        self.clear_gpu_cache()
        queries_embeddings = self.model.encode(
            queries, 
            batch_size=10, 
            show_progress_bar=self.show_progress, 
            normalize_embeddings=True,
            prompt_name="query"
        )
        return queries_embeddings