from helper.semantic_chunker import SemChunker
from helper.retrieval_embedding import Retrieval_Embed
from helper.config import RETRIEVAL_SIMILARITY_THRESHOLD, EVIDENCE_SIMILARITY_THRESHOLD, EVIDENCES_PER_QUERY
from tqdm import *
import numpy as np
import torch
import torch.nn.functional as F
import os
import json
import time
import shutil
import argparse

print("RETRIEVAL_SIMILARITY_THRESHOLD:", RETRIEVAL_SIMILARITY_THRESHOLD)
print("EVIDENCES_PER_QUERY", EVIDENCES_PER_QUERY)
print("EVIDENCE_SIMILARITY_THRESHOLD", EVIDENCE_SIMILARITY_THRESHOLD)

def initialize_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Cleared existing directory: {path}")
    os.makedirs(path, exist_ok=True)
    print(f"Directory ready: {path}")

def filter_similar_evidences(evidences, embeddings, threshold=0.9):
    if not evidences:
        return []

    emb_np = np.asarray(embeddings, dtype=np.float32)  # one memcpy, contiguous
    emb = torch.from_numpy(emb_np).to(device)
    emb = F.normalize(emb, dim=1) 
    kept_idx   : list[int]   = []
    kept_vecs  : list[torch.Tensor] = []

    for i in range(len(evidences)):
        if not kept_vecs:
            kept_idx.append(i)
            kept_vecs.append(emb[i])
            continue

        # stack *once* per loop
        sims = F.cosine_similarity(emb[i].unsqueeze(0), torch.stack(kept_vecs), dim=1)

        if sims.max() < threshold:
            kept_idx.append(i)
            kept_vecs.append(emb[i])

    return [evidences[i] for i in kept_idx]

def process_claim(file_path, index_idx, sem_chunker, retrieval_emb, qg_data, output_dir):    
    # Load bm25 filtered docs
    with open(file_path, encoding='utf-8') as fp:
        bm25_results = json.load(fp)
        
    # Load queries
    queries = [q for q, _ in qg_data[index_idx].items()]

    document_urls = []
    documents = []
    for url, document_obj in bm25_results.items():
        document_urls.append(url)
        documents.append(document_obj['document'])
    
    # Break paragraphs into chunks
    start = time.time()
    batch_chunks = sem_chunker.chunk_documents(documents)
    
    # Extract chunks
    all_chunks = []
    all_chunks_urls = []
    all_chunks_embeds = []
    for i, doc_chunks in enumerate(batch_chunks):
        curr_url = document_urls[i]
        for chunk in doc_chunks:
            # No point saving these chunks, too big for our use-case
            if chunk.token_count > 160:
                continue

            # Calculate mean of all sentences that make up the chunk; Use inbuilt function of the library
            chunk_mean_embed = sem_chunker.chunker._compute_group_embedding(chunk.sentences)
            
            all_chunks_embeds.append(chunk_mean_embed)
            all_chunks.append(chunk.text)
            all_chunks_urls.append(curr_url)
    sem_chunking_time = time.time() - start
    
    # Encode chunks and queries for retrieval
    start = time.time()
    evidence_embeddings = retrieval_emb.encode_documents(all_chunks)
    queries_embeddings = retrieval_emb.encode_queries(queries)    

    # Calculate similarity between chunks and queries
    similarity_matrix = retrieval_emb.model.similarity(queries_embeddings, evidence_embeddings)

    # Gather evidence and save to file
    queries_evidence_map = {}
    num_evidences = evidence_embeddings.shape[0]
    for q_id, q in enumerate(queries):
        similarities = similarity_matrix[q_id]
        values, indices = torch.topk(similarities, k=min(num_evidences, EVIDENCES_PER_QUERY), largest=True)

        evidences = []
        evidences_embeds = []
        for val, idx in zip(values, indices):
            score = float(val)
            if score < RETRIEVAL_SIMILARITY_THRESHOLD:
                continue            
            evidences.append({
                "evidence": all_chunks[idx],
                "score": score,
                "source_url": all_chunks_urls[idx]
            })
            evidences_embeds.append(all_chunks_embeds[idx])
        queries_evidence_map[q] = filter_similar_evidences(evidences, evidences_embeds, threshold=EVIDENCE_SIMILARITY_THRESHOLD)
    encoding_sim_time = time.time() - start

    with open(os.path.join(output_dir, f'{index_idx}.json'), 'w', encoding="utf-8") as fp:
        json.dump(queries_evidence_map, fp, indent=4)
        
    return {
        "total_urls": len(document_urls),
        "total_chunks": len(all_chunks), 
        "sem_chunking_time": sem_chunking_time,
        "encoding_sim_time": encoding_sim_time
    }


if __name__ == "__main__":
    # python semantic_filtering.py --CLAIMS_PATH ./data_store/averitec/dev.json
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--CLAIMS_PATH', default='./data_store/averitec/dev.json')
    args = parser.parse_args()

    with open(args.CLAIMS_PATH, "r", encoding="utf-8") as fp:
        claims_dataset = json.load(fp)
        claims_count = len(claims_dataset)
    
    root_dir = "output"

    # Load bm25 results
    bm25_results_dir = f'{root_dir}/bm25_results'
    bm25_result_files = [os.path.join(bm25_results_dir, f) for f in os.listdir(bm25_results_dir) if f.endswith('.json')]

    # Load HyDE questions
    with open(f'{root_dir}/qg_data.json', 'r', encoding='utf-8') as fp:
        qg_data = json.load(fp)

    # Define the save directory path
    output_dir = f'{root_dir}/semantic_results'
    initialize_dir(output_dir)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize the chunker and retrieval task embedding models
    sem_chunker = SemChunker()
    retrieval_emb = Retrieval_Embed(device=device)

    # Process all claims and collect stats
    chunk_embed_stats = {}
    for index_idx in tqdm(range(0, claims_count)):
        bm25_file_path = os.path.join(bm25_results_dir, f'{index_idx}.json')

        try:
            chunk_embed_stats[index_idx] = process_claim(bm25_file_path, 
                                                         str(index_idx), 
                                                         sem_chunker, 
                                                         retrieval_emb, 
                                                         qg_data, 
                                                         output_dir)
        except Exception as e:
            print(f"Error processing Index {index_idx}: {e}")
            
    # Save statistics
    with open(f'{root_dir}/chunk_embed_stats.json', 'w', encoding='utf-8') as fp:
        json.dump(chunk_embed_stats, fp, indent=4)