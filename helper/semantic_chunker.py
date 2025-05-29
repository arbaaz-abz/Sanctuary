from chonkie import SemanticChunker #, SDPMChunker
from helper.config import SEMANTIC_CHUNKER_MODEL_NAME, MIN_CHUNK_SIZE, MAX_CHUNK_SIZE, CHUNKER_MIN_SENTENCES, SIMILARITY_WINDOW
import gc
import torch

class SemChunker:
    def __init__(self): 
        print("Using:",SEMANTIC_CHUNKER_MODEL_NAME)
        # self.chunker = SDPMChunker(
        #     embedding_model=SEMANTIC_CHUNKER_MODEL_NAME,
        #     threshold="auto",
        #     trust_remote_code=True,
        #     min_chunk_size=MIN_CHUNK_SIZE,
        #     chunk_size=MAX_CHUNK_SIZE, # Maximum tokens allowed per chunk
        #     min_sentences=CHUNKER_MIN_SENTENCES, # Minimum number of sentences per chunk
        #     skip_window=CHUNKER_WINDOW_OVERLAP,
        #     similarity_window=SIMILARITY_WINDOW
        # )

        self.chunker = SemanticChunker(
            embedding_model=SEMANTIC_CHUNKER_MODEL_NAME,
            threshold="auto",
            trust_remote_code=True,
            min_chunk_size=MIN_CHUNK_SIZE,
            chunk_size=MAX_CHUNK_SIZE, # Maximum tokens allowed per chunk
            min_sentences=CHUNKER_MIN_SENTENCES, # Minimum number of sentences per chunk
            similarity_window=SIMILARITY_WINDOW
        )

    def clear_gpu_cache(self):
        gc.collect()
        torch.cuda.empty_cache()
        
    def chunk_documents(self, documents):
        self.clear_gpu_cache()
        return self.chunker.chunk_batch(documents)