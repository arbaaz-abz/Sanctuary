#!/bin/bash

# System configuration
SYSTEM_NAME="yellow_flash"
SPLIT="dev"  # Change this to "dev", or "test"
BASE_DIR="." # "."  # Current directory

DATA_STORE="${BASE_DIR}/data_store"
KNOWLEDGE_STORE="${BASE_DIR}/knowledge_store"
export HF_HOME="${BASE_DIR}/huggingface_cache"

# Create necessary directories
mkdir -p "${DATA_STORE}/averitec"
mkdir -p "${DATA_STORE}/${SYSTEM_NAME}"
mkdir -p "${KNOWLEDGE_STORE}/${SPLIT}"
mkdir -p "${HF_HOME}"

# Execute each script from src directory
python question_doc_generator.py --CLAIMS_PATH "${DATA_STORE}/averitec/${SPLIT}.json" || exit 1

python bm25_retrieval.py --CLAIMS_PATH "${DATA_STORE}/averitec/${SPLIT}.json" --KNOWLEDGE_STORE_PATH "${KNOWLEDGE_STORE}/${SPLIT}" || exit 1

python semantic_filtering.py --CLAIMS_PATH "${DATA_STORE}/averitec/${SPLIT}.json" || exit 1

python veracity_prediction.py --CLAIMS_PATH "${DATA_STORE}/averitec/${SPLIT}.json" --OUTPUT_FILE "output/${SPLIT}_veracity_prediction.json" || exit 1

python prepare_leaderboard_submission.py --filename "output/${SPLIT}_veracity_prediction.json" || exit 1