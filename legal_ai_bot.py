from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import json
from sentence_transformers import SentenceTransformer, util
import torch

# Download tokenizer
nltk.download("punkt")

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to "http://localhost:5500"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

# -------------------- Utils --------------------

def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def flatten_sections(data, source):
    docs = []
    for item in data:
        docs.append({
            "source": source,
            "chapter": item.get("chapter"),
            "section": item.get("section") or item.get("Section"),
            "title": item.get("section_title"),
            "text": item.get("section_desc")
        })
    return docs

# -------------------- BM25 --------------------

class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.corpus = [word_tokenize(doc["text"].lower()) for doc in documents]
        self.bm25 = BM25Okapi(self.corpus)

    def get_top_k(self, query: str, k: int = 5):
        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.documents[i] for i in top_indices]

# -------------------- BERT --------------------

class BERTSemanticRetriever:
    def __init__(self, documents, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.texts = [doc["text"] for doc in documents]
        self.embeddings = self.model.encode(self.texts, convert_to_tensor=True)
        self.documents = documents

    def get_top_k(self, query: str, k: int = 5):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, self.embeddings)[0]
        top_indices = torch.topk(scores, k).indices
        return [self.documents[i] for i in top_indices]

# -------------------- Hybrid --------------------

def merge_results(bm25_docs, bert_docs):
    doc_scores = {}
    for i, doc in enumerate(bm25_docs):
        doc_scores[doc['text']] = doc_scores.get(doc['text'], 0) + (5 - i)

    for i, doc in enumerate(bert_docs):
        doc_scores[doc['text']] = doc_scores.get(doc['text'], 0) + (5 - i)

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [next(doc for doc in bm25_docs + bert_docs if doc["text"] == text) for text, _ in sorted_docs[:5]]

# -------------------- Load Data --------------------

ipc_data = flatten_sections(load_json("data/ipc.json"), "IPC")
crpc_data = flatten_sections(load_json("data/crpc.json"), "CrPC")
all_documents = ipc_data + crpc_data

bm25 = BM25Retriever(all_documents)
bert = BERTSemanticRetriever(all_documents)

# -------------------- FastAPI Endpoint --------------------

@app.get("/ask")
def ask(query: str):
    bm25_results = bm25.get_top_k(query)
    bert_results = bert.get_top_k(query)
    merged = merge_results(bm25_results, bert_results)
    return {"answers": merged}
