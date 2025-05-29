import time
import re
import pandas as pd
import json
import nltk
import heapq
import shutil
import os
import argparse
from collections import defaultdict
from rank_bm25 import BM25Okapi
from multiprocessing import Pool
from tqdm import *
from functools import partial
from helper.config import TOKENIZATION_WORKERS, MAX_SENTENCES, OVERLAP, TOP_N_PER_QUERY

print(f"TOP_N_PER_QUERY: {TOP_N_PER_QUERY}")
print(f"MAX_SENTENCES: {MAX_SENTENCES}")
print(f"TOKENIZATION_WORKERS: {TOKENIZATION_WORKERS}")

def download_nltk_data(package_name, download_dir='nltk_data'):
    # Ensure the download directory exists
    os.makedirs(download_dir, exist_ok=True)
    
    # Set NLTK data path
    nltk.data.path.append(download_dir)
    
    try:
        # Try to find the resource
        nltk.data.find(f'tokenizers/{package_name}')
        print(f"Package '{package_name}' is already downloaded")
    except LookupError:
        # If resource isn't found, download it
        print(f"Downloading {package_name}...")
        nltk.download(package_name, download_dir=download_dir)
        print(f"Successfully downloaded {package_name}")

def remove_duplicates(sentences, urls):
    df = pd.DataFrame({"document_in_sentences":sentences, "sentence_urls":urls})
    df['sentences'] = df['document_in_sentences'].str.strip().str.lower()
    df = df.drop_duplicates(subset="sentences").reset_index()
    return df['document_in_sentences'].tolist(), df['sentence_urls'].tolist()

def get_token_count(content, chars_per_token=4.0):
    content = re.sub(r'\s{2,}', ' ', content)
    content = content.strip()
    if not content:
        return 0
    token_count = int(len(content) / chars_per_token)
    return token_count

def initialize_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Cleared existing directory: {path}")
    os.makedirs(path, exist_ok=True)
    print(f"Directory ready: {path}")

def make_paragraphs_for_url(sentences, target_tokens_per_paragraph=500):    
    paragraphs = []
    current_paragraph_sentences = []
    current_token_count = 0
    sentence_index = 0
    
    while sentence_index < len(sentences):
        sentence = sentences[sentence_index]
        s_tokens = get_token_count(sentence)
        
        # Add sentence to current paragraph if it's the first sentence
        if not current_paragraph_sentences:
            current_paragraph_sentences.append(sentence)
            current_token_count += s_tokens                      
            # print(f"Starting PARAGRAPH with S{sentence_index}; Token Count: {current_token_count}")
            sentence_index += 1
        else:
            # If adding the sentence doesn't increase the global paragraph token target
            if current_token_count + s_tokens <= target_tokens_per_paragraph:
                current_paragraph_sentences.append(sentence)
                current_token_count += s_tokens                       
                # print(f"Adding S{sentence_index}; Token Count: {current_token_count}")
                sentence_index += 1
            else:
                # Add sentence if it doesn't cause the paragraph to become too big!
                if current_token_count + s_tokens <= 1.25 * target_tokens_per_paragraph:
                    current_paragraph_sentences.append(sentence)
                    current_token_count += s_tokens
                    # print(f"Within acceptable range, Adding S{sentence_index}; Token Count: {current_token_count}")
                    sentence_index += 1

                # Wrap up the paragraph
                paragraph = " ".join(current_paragraph_sentences)
                if paragraph:
                    paragraphs.append(paragraph)
                        
                # Start a new paragraph
                current_paragraph_sentences = []
                current_token_count = 0
                    
    if current_paragraph_sentences:
        paragraph = " ".join(current_paragraph_sentences)
        if paragraph:
            paragraphs.append(paragraph)
    return paragraphs
    
def chunk_into_paragraphs_parallel(file_path, pool, avg_tokens_per_sentence=42, target_sentences_per_para=12):
    evidence_docs = []
    urls = []

    # target_tokens = avg_tokens_per_sentence * target_sentences_per_para
    # print(f"Target tokens per paragraph: {avg_tokens_per_sentence} x {target_sentences_per_para}")

    all_sentences_lists = []
    all_urls = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for evidence_idx, line in enumerate(f):
            json_obj = json.loads(line)
            if not json_obj:
                continue

            url = json_obj.get("url", f"fallback_url_{evidence_idx}") 
            sentences = json_obj["url2text"]  
            if not sentences:
                continue

            all_sentences_lists.append(sentences)
            all_urls.append(url)

    # make_paragraphs_partial = partial(make_paragraphs_for_url, target_tokens_per_paragraph=target_tokens)
    # results = pool.map(make_paragraphs_partial, all_sentences_lists)
    results = pool.map(make_paragraphs_for_url, all_sentences_lists)
    for url, paragraphs_for_url in zip(all_urls, results):
        evidence_docs.extend(paragraphs_for_url)
        urls.extend([url] * len(paragraphs_for_url))

    return evidence_docs, urls

def tokenize_paragraph(paragraph):
    return nltk.word_tokenize(paragraph)

def tokenize_paragraphs_parallel(paragraphs, pool):
    """Tokenise a list of paragraphs in parallel with an existing Pool."""
    return pool.map(tokenize_paragraph, paragraphs)

def build_bm25_index_parallel(paragraphs, pool):
    tokenized_paras = tokenize_paragraphs_parallel(paragraphs, pool)
    return BM25Okapi(tokenized_paras)

def bm25_retrieval(queries, bm25, pool, top_n_per_query):
    top_bm25_paras = set()    
    tokenized_queries = tokenize_paragraphs_parallel(queries, pool)
    for tokenized_query in tokenized_queries:
        scores = bm25.get_scores(tokenized_query)
        ranked_para_ids = heapq.nlargest(top_n_per_query, range(len(scores)), key=scores.__getitem__)
        top_bm25_paras.update(ranked_para_ids)
    return top_bm25_paras


if __name__ == "__main__":
    # CLAIMS_PATH - /home/rogers/thesis/AVeriTeC/data_store/averitec/test_2025_sample.json
    # KNOWLEDGE_STORE_PATH - /home/rogers/thesis/AVeriTeC/knowledge_store/test_2025
    # python bm25_retrieval.py --CLAIMS_PATH ./data_store/averitec/dev.json --KNOWLEDGE_STORE_PATH ./knowledge_store/dev
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--CLAIMS_PATH', default='./data_store/averitec/dev.json')
    parser.add_argument('-k', '--KNOWLEDGE_STORE_PATH', default='./knowledge_store/dev/')
    args = parser.parse_args()

    download_nltk_data('punkt')
    download_nltk_data('punkt_tab')
    
    with open(args.CLAIMS_PATH, "r", encoding="utf-8") as fp:
        claims_dataset = json.load(fp)
        claims_count = len(claims_dataset)
        
    root_dir = "output"

    # Load HyDE questions
    with open(f'{root_dir}/qg_data.json', 'r', encoding='utf-8') as fp:
        qg_data = json.load(fp)
    
    # Define save directory path
    output_dir = f'{root_dir}/bm25_results'
    initialize_dir(output_dir)

    bm25_metadata = {}
    with Pool(TOKENIZATION_WORKERS) as pool:
        for index_id in tqdm(range(0, claims_count)):
            claim_object = claims_dataset[index_id]
            
            start = time.time()
            try:
                if not qg_data[str(index_id)]:
                    print(f"Skipping Index {index_id}, No HyDE Queries/Docs found")
                    continue
                    
                qg_obj = qg_data[str(index_id)]
                queries = [q + ' ' + ' '.join(a_list) for q, a_list in qg_obj.items()]        

                claim_evidence_file = os.path.join(args.KNOWLEDGE_STORE_PATH, f"{claim_object.get('claim_id', index_id)}.json")
                evidences_in_para_init, para_urls_init = chunk_into_paragraphs_parallel(claim_evidence_file, 
                                                                                        pool,
                                                                                        target_sentences_per_para=MAX_SENTENCES)
                evidences_in_para, para_urls = remove_duplicates(evidences_in_para_init, para_urls_init)
            
                bm25_index = build_bm25_index_parallel(evidences_in_para, pool)                                    
                top_bm25_paras = bm25_retrieval(queries, bm25_index, pool, top_n_per_query=TOP_N_PER_QUERY)
                
                # Aggregate paragraphs by URL
                agg_docs_ids = defaultdict(set)
                for para_id in top_bm25_paras:
                    agg_docs_ids[para_urls[para_id]].add(para_id)
                
                # Combine all paras to form one big document per URL
                url_document_map = {}
                total_paras = 0
                for url, selected_para_ids in agg_docs_ids.items():
                    unique_para_ids = sorted(selected_para_ids)
                    content = [evidences_in_para[para_id] for para_id in unique_para_ids]
                    url_document_map[url] = {'document': ' '.join(content), 'num_paras': len(unique_para_ids)}
                    total_paras += len(unique_para_ids)
    
                with open(os.path.join(output_dir, f'{index_id}.json'), 'w', encoding="utf-8") as fp:
                    json.dump(url_document_map, fp, indent=4)
                    
                bm25_metadata[index_id] = {
                    'total_time': time.time() - start,
                    'total_urls': len(url_document_map),
                    'bm25_total_paras': len(top_bm25_paras),
                    'total_paras': total_paras
                }
            except Exception as e:
                print(f"Error processing Index {index_id}: {e}")
                   
    with open(f'{root_dir}/bm25_metadata.json', 'w', encoding="utf-8") as fp:
            json.dump(bm25_metadata, fp, indent=4)