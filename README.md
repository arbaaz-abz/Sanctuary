<h1 align="center">SANCTUARY - Automated Fact-Checker</h1>

<p align="center">
  <em>An End-to-End, reproducible pipeline for fact-checking of real-world claims.</em>
</p>

---

## ✨ Pipeline Stages
| Module | What it does |
|--------|--------------|
| `question_doc_generator.py` | Few-shot LLM prompting to initially assess a claim and expand it using HyDE-based multi-hop **question + answer** pairs (Q-A) generation. |
| `bm25_retrieval.py`         | Coarse and efficient BM25 wrapper to pull top candidate paragraphs from evidence articles relevant to answering each question. |
| `semantic_filtering.py`     | A Semantic Chucking and Semantic Retrieval based method to extract critical snippets to address each claim query. |
| `veracity_prediction.py`    | Few-shot LLM classifier that maps bundled Q-A pairs to **Supported / Refuted / Not Enough Evidence, Conflicting Evidence/Cherrypicking** labels. |

---

## 🚀 How to run

```bash
download_data.sh
installation.sh
conda activate [YOUR ENVIRONMENT]
run_system.sh
```
The script run_system.sh calls system_inference.sh while measuring the total time the system takes.

---

## 📂 Dataset Resources

### 📘 Official AVeriTeC 2024 Datasets  
Includes **Train**, **Dev**, and **Test** claims along with the complete knowledge store.  
📥 Hugging Face Repository:  
[https://huggingface.co/chenxwh/AVeriTeC/tree/main](https://huggingface.co/chenxwh/AVeriTeC/tree/main)

---

### 🧪 AVeriTeC 2025 Test Set  
Contains newly released **Test claims** and corresponding **knowledge store** for the 2025 edition.  
📁 Google Drive Folder:  
[https://drive.google.com/drive/folders/1DzcJogH3592Ibv19uFWUI84FpbL7NCcC](https://drive.google.com/drive/folders/1DzcJogH3592Ibv19uFWUI84FpbL7NCcC)

