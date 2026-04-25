# Retrieval-Augmented Generation (RAG) - Complete Guide

## ⚡ Interview Quick Summary

> **Core insight**: RAG = separate retrieval from generation. The key challenge is making retrieval accurate enough that the generator can trust it, while keeping latency acceptable.

### RAG Architecture Decision Tree

```
Do you need up-to-date information beyond training cutoff?
  YES → RAG is essential

Is the knowledge base large (can't fit in context)?
  YES → RAG (chunking + vector search)
  NO  → Long context model might work (Gemini 1.5 Pro, Claude)

Do you need citations/sources?
  YES → RAG with metadata tracking

Is query-to-document matching simple keyword-based?
  YES → BM25 alone may suffice
  NO  → Dense retrieval (embeddings) or hybrid

Do you have high accuracy requirements?
  YES → Add re-ranking (cross-encoder) after retrieval
```

### Common Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| Retrieval misses | Answer says "I don't know" for findable facts | Better chunking, hybrid search |
| Retrieved but ignored | Context has answer, LLM ignores it | Reorder context, stronger prompt |
| Hallucination despite context | LLM adds facts not in context | Explicit "only use context" instruction, faithfulness check |
| Lost in middle | Middle chunks ignored | Put key docs at start/end |
| Chunk boundary cuts | Important text split across chunks | Use overlap (20%), semantic chunking |

### 🚨 Top Interview Pitfalls
- Forgetting that **embedding model choice** matters — use domain-appropriate embeddings
- Not addressing **chunking strategy** — chunk size and overlap directly affect retrieval quality
- Saying "just use cosine similarity" without mentioning re-ranking for production accuracy
- Missing **data freshness**: who updates the index, and how often?
- No mention of **evaluation**: how do you know if RAG is working?

### RAG Evaluation Framework (RAGAS)

```python
# Key metrics for evaluating RAG systems
RAG_METRICS = {
    # Retrieval quality
    "context_precision":  "Fraction of retrieved chunks that are relevant",
    "context_recall":     "Fraction of relevant chunks that were retrieved",
    
    # Generation quality
    "faithfulness":       "Answer is fully grounded in context (no hallucination)",
    "answer_relevancy":   "Answer actually addresses the question",
    
    # End-to-end
    "answer_correctness": "Answer matches ground truth (requires reference)",
}

# Minimum viable RAG evaluation
def evaluate_rag_response(question, context_docs, answer, llm_judge):
    # 1. Faithfulness: is every claim in the answer supported by context?
    faithfulness_prompt = f"""
    Context: {context_docs}
    Answer: {answer}
    
    Is every statement in the Answer fully supported by the Context? 
    Return: faithful/unfaithful and list any unsupported claims.
    """
    
    # 2. Relevancy: does the answer address the question?
    relevancy_prompt = f"""
    Question: {question}
    Answer: {answer}
    Does the Answer address the Question? Score 1-5.
    """
    
    faithfulness = llm_judge.evaluate(faithfulness_prompt)
    relevancy    = llm_judge.evaluate(relevancy_prompt)
    return {"faithfulness": faithfulness, "relevancy": relevancy}
```

---

## Table of Contents
1. [RAG Fundamentals](#rag-fundamentals)
2. [Document Processing](#document-processing)
3. [Embedding Models](#embedding-models)
4. [Vector Databases](#vector-databases)
5. [Advanced RAG Techniques](#advanced-rag-techniques)
6. [Evaluation and Optimization](#evaluation-and-optimization)
7. [Interview Questions](#interview-questions)

---

## 1. RAG Fundamentals

### RAG Architecture Overview

```python
"""
RAG: Retrieval-Augmented Generation

Why RAG?
1. Reduces hallucinations (grounded in documents)
2. Provides up-to-date information
3. Enables domain-specific knowledge
4. More cost-effective than fine-tuning
5. Transparent sources (citations)

Basic RAG Pipeline:
Query → Retriever → Context → Generator → Answer

Components:
1. Document Store: Raw documents
2. Embedding Model: Text → vectors
3. Vector Database: Store/search embeddings
4. Retriever: Find relevant documents
5. Generator (LLM): Produce answer
"""

class BasicRAG:
    """Simple RAG implementation."""
    
    def __init__(self, embedding_model, vector_store, llm):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm = llm
    
    def index_documents(self, documents):
        """Index documents into vector store."""
        for doc in documents:
            embedding = self.embedding_model.encode(doc.text)
            self.vector_store.add(
                embedding=embedding,
                metadata={"text": doc.text, "source": doc.source}
            )
    
    def retrieve(self, query, k=5):
        """Retrieve top-k relevant documents."""
        query_embedding = self.embedding_model.encode(query)
        results = self.vector_store.search(query_embedding, k=k)
        return results
    
    def generate(self, query, context_docs):
        """Generate answer using retrieved context."""
        context = "\n\n".join([doc["text"] for doc in context_docs])
        
        prompt = f"""Answer the question based on the following context.
If the answer cannot be found in the context, say "I don't know."

Context:
{context}

Question: {query}

Answer:"""
        
        return self.llm.generate(prompt)
    
    def query(self, question, k=5):
        """End-to-end RAG query."""
        # Retrieve
        docs = self.retrieve(question, k=k)
        
        # Generate
        answer = self.generate(question, docs)
        
        return {
            "answer": answer,
            "sources": docs
        }
```

### RAG vs Fine-tuning

```python
"""
When to use RAG vs Fine-tuning:

RAG Advantages:
- No training required
- Easy to update knowledge
- Transparent sources
- Works with any LLM
- Lower cost for knowledge updates

Fine-tuning Advantages:
- Better task-specific behavior
- Faster inference (no retrieval)
- Better for style/format changes
- Works offline

Use RAG when:
- Knowledge changes frequently
- Need citations/sources
- Domain-specific factual QA
- Limited training data

Use Fine-tuning when:
- Changing model behavior/style
- Specific output format required
- Task-specific optimization
- Latency critical

Often best: RAG + Fine-tuned model
"""
```

---

## 2. Document Processing

### Text Chunking Strategies

```python
from typing import List
import re

class TextChunker:
    """Various chunking strategies for documents."""
    
    @staticmethod
    def fixed_size_chunks(text: str, chunk_size: int = 1000, 
                         overlap: int = 200) -> List[str]:
        """
        Fixed-size chunking with overlap.
        Simple but may cut mid-sentence.
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    @staticmethod
    def sentence_chunks(text: str, max_chunk_size: int = 1000) -> List[str]:
        """
        Sentence-based chunking.
        Respects sentence boundaries.
        """
        import nltk
        sentences = nltk.sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            if current_size + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sentence)
            current_size += len(sentence)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    @staticmethod
    def semantic_chunks(text: str, embedding_model, 
                       similarity_threshold: float = 0.5) -> List[str]:
        """
        Semantic chunking based on topic shifts.
        Groups related sentences together.
        """
        import nltk
        import numpy as np
        
        sentences = nltk.sent_tokenize(text)
        embeddings = embedding_model.encode(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # Calculate similarity with previous sentence
            similarity = np.dot(embeddings[i], embeddings[i-1])
            
            if similarity < similarity_threshold:
                # Topic shift detected
                chunks.append(" ".join(current_chunk))
                current_chunk = []
            
            current_chunk.append(sentences[i])
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    @staticmethod
    def recursive_chunks(text: str, chunk_size: int = 1000,
                        separators: List[str] = None) -> List[str]:
        """
        Recursive character text splitter.
        Tries to split on semantic boundaries.
        """
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        
        def split_text(text, separators):
            if len(text) <= chunk_size:
                return [text]
            
            for sep in separators:
                if sep in text:
                    parts = text.split(sep)
                    chunks = []
                    current = ""
                    
                    for part in parts:
                        if len(current) + len(part) + len(sep) <= chunk_size:
                            current += part + sep
                        else:
                            if current:
                                chunks.append(current.strip())
                            current = part + sep
                    
                    if current:
                        chunks.append(current.strip())
                    
                    # Recursively split any chunks that are too large
                    result = []
                    for chunk in chunks:
                        if len(chunk) > chunk_size:
                            result.extend(split_text(chunk, separators[1:]))
                        else:
                            result.append(chunk)
                    
                    return result
            
            # No separator found, force split
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        return split_text(text, separators)


# LangChain text splitters
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_text(document_text)
```

### Document Loaders

```python
from typing import List, Dict
import os

class DocumentLoader:
    """Load various document formats."""
    
    @staticmethod
    def load_pdf(path: str) -> str:
        """Load PDF document."""
        import fitz  # PyMuPDF
        
        text = ""
        doc = fitz.open(path)
        for page in doc:
            text += page.get_text()
        return text
    
    @staticmethod
    def load_docx(path: str) -> str:
        """Load Word document."""
        from docx import Document
        
        doc = Document(path)
        return "\n".join([para.text for para in doc.paragraphs])
    
    @staticmethod
    def load_html(path: str) -> str:
        """Load HTML and extract text."""
        from bs4 import BeautifulSoup
        
        with open(path, 'r') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        
        # Remove scripts and styles
        for tag in soup(['script', 'style']):
            tag.decompose()
        
        return soup.get_text(separator='\n')
    
    @staticmethod
    def load_markdown(path: str) -> str:
        """Load Markdown file."""
        with open(path, 'r') as f:
            return f.read()
    
    @staticmethod
    def load_directory(dir_path: str, extensions: List[str] = None) -> List[Dict]:
        """Load all documents from directory."""
        if extensions is None:
            extensions = ['.pdf', '.docx', '.txt', '.md', '.html']
        
        documents = []
        
        for root, _, files in os.walk(dir_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in extensions:
                    path = os.path.join(root, file)
                    
                    if ext == '.pdf':
                        text = DocumentLoader.load_pdf(path)
                    elif ext == '.docx':
                        text = DocumentLoader.load_docx(path)
                    elif ext == '.html':
                        text = DocumentLoader.load_html(path)
                    else:
                        with open(path, 'r') as f:
                            text = f.read()
                    
                    documents.append({
                        "text": text,
                        "source": path,
                        "filename": file
                    })
        
        return documents
```

---

## 3. Embedding Models

### Embedding Model Comparison

```python
"""
Popular Embedding Models:

Open Source:
- all-MiniLM-L6-v2: Fast, good quality (384 dim)
- all-mpnet-base-v2: Higher quality (768 dim)
- bge-large-en: State-of-art open source (1024 dim)
- e5-large-v2: Microsoft, excellent quality (1024 dim)
- GTE-large: Alibaba, competitive (1024 dim)

Commercial:
- OpenAI text-embedding-ada-002: 1536 dim, $0.0001/1K tokens
- OpenAI text-embedding-3-small: 1536 dim, cheaper
- OpenAI text-embedding-3-large: 3072 dim, best quality
- Cohere embed-v3: Good multilingual support
- Voyage AI: High quality, various sizes

Selection Criteria:
1. Quality (MTEB benchmark scores)
2. Dimension (storage/compute trade-off)
3. Speed (latency requirements)
4. Cost (for commercial APIs)
5. Max sequence length
"""

from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    """Wrapper for embedding models."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts, batch_size: int = 32, 
               normalize: bool = True) -> np.ndarray:
        """Encode texts to embeddings."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.encode([text1])[0]
        emb2 = self.encode([text2])[0]
        return np.dot(emb1, emb2)


# OpenAI embeddings
import openai

def openai_embed(texts: List[str], model: str = "text-embedding-3-small"):
    """Get embeddings from OpenAI API."""
    response = openai.embeddings.create(
        model=model,
        input=texts
    )
    return [item.embedding for item in response.data]


# Instruction-tuned embeddings (better for retrieval)
class InstructEmbedding:
    """
    Instruction-following embeddings (e.g., E5, BGE).
    Add task-specific instructions to improve retrieval.
    """
    def __init__(self, model_name: str = "intfloat/e5-large-v2"):
        self.model = SentenceTransformer(model_name)
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode query with instruction."""
        instructed = f"query: {query}"
        return self.model.encode(instructed, normalize_embeddings=True)
    
    def encode_document(self, document: str) -> np.ndarray:
        """Encode document with instruction."""
        instructed = f"passage: {document}"
        return self.model.encode(instructed, normalize_embeddings=True)
```

---

## 4. Vector Databases

### Vector Database Implementations

```python
import numpy as np
from typing import List, Dict, Any

# FAISS (Facebook AI Similarity Search)
import faiss

class FAISSVectorStore:
    """FAISS-based vector store."""
    
    def __init__(self, dimension: int, index_type: str = "flat"):
        self.dimension = dimension
        self.documents = []
        
        if index_type == "flat":
            # Exact search (brute force)
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "ivf":
            # Approximate search with inverted file
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        elif index_type == "hnsw":
            # Hierarchical Navigable Small World
            self.index = faiss.IndexHNSWFlat(dimension, 32)
    
    def add(self, embeddings: np.ndarray, documents: List[Dict]):
        """Add documents with their embeddings."""
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.documents.extend(documents)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar documents."""
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append({
                    **self.documents[idx],
                    "score": float(score)
                })
        
        return results


# Chroma (simple, popular for prototyping)
import chromadb

class ChromaVectorStore:
    """ChromaDB vector store."""
    
    def __init__(self, collection_name: str = "documents"):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add(self, texts: List[str], embeddings: List[List[float]], 
            metadatas: List[Dict] = None, ids: List[str] = None):
        """Add documents."""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """Search by embedding."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        return [
            {
                "text": doc,
                "metadata": meta,
                "score": 1 - dist  # Convert distance to similarity
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]


# Pinecone (managed, production-ready)
"""
import pinecone

pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")

# Create index
pinecone.create_index(
    name="my-index",
    dimension=1536,
    metric="cosine"
)

index = pinecone.Index("my-index")

# Upsert vectors
index.upsert(
    vectors=[
        {"id": "doc1", "values": embedding1, "metadata": {"text": "..."}},
        {"id": "doc2", "values": embedding2, "metadata": {"text": "..."}},
    ]
)

# Query
results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)
"""
```

---

## 5. Advanced RAG Techniques

### Hybrid Search

```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    """
    Combine dense (semantic) and sparse (keyword) retrieval.
    
    Dense: Good for semantic similarity
    Sparse: Good for exact keyword matches
    Hybrid: Best of both worlds
    """
    
    def __init__(self, embedding_model, documents: List[str]):
        self.embedding_model = embedding_model
        self.documents = documents
        
        # Dense index
        self.embeddings = embedding_model.encode(documents)
        
        # Sparse index (BM25)
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
    
    def search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Dict]:
        """
        Hybrid search with configurable weighting.
        alpha: Weight for dense scores (1-alpha for sparse)
        """
        # Dense scores
        query_emb = self.embedding_model.encode([query])[0]
        dense_scores = np.dot(self.embeddings, query_emb)
        
        # Normalize dense scores
        dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-6)
        
        # Sparse scores
        sparse_scores = self.bm25.get_scores(query.lower().split())
        sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min() + 1e-6)
        
        # Combine scores
        combined_scores = alpha * dense_scores + (1 - alpha) * sparse_scores
        
        # Get top-k
        top_indices = np.argsort(combined_scores)[-k:][::-1]
        
        return [
            {
                "text": self.documents[i],
                "score": combined_scores[i],
                "dense_score": dense_scores[i],
                "sparse_score": sparse_scores[i]
            }
            for i in top_indices
        ]
```

### Query Transformation

```python
class QueryTransformer:
    """Transform queries for better retrieval."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def expand_query(self, query: str) -> List[str]:
        """Generate multiple query variations."""
        prompt = f"""Generate 3 alternative versions of this search query 
that might help find relevant information:

Original: {query}

Alternatives:
1."""
        
        response = self.llm.generate(prompt)
        queries = [query] + self._parse_queries(response)
        return queries
    
    def decompose_query(self, query: str) -> List[str]:
        """Break complex query into sub-queries."""
        prompt = f"""Break down this complex question into simpler sub-questions:

Question: {query}

Sub-questions:
1."""
        
        response = self.llm.generate(prompt)
        return self._parse_queries(response)
    
    def hypothetical_document(self, query: str) -> str:
        """
        HyDE: Generate hypothetical document that would answer the query.
        Use this for retrieval instead of query.
        """
        prompt = f"""Write a short paragraph that would be a perfect answer 
to this question (even if you're not sure of the facts):

Question: {query}

Answer:"""
        
        return self.llm.generate(prompt, max_tokens=200)
    
    def step_back(self, query: str) -> str:
        """Generate a more general/abstract query."""
        prompt = f"""Given this specific question, generate a more general 
question that would help provide context:

Specific: {query}

General:"""
        
        return self.llm.generate(prompt, max_tokens=100)
```

### Reranking

```python
from sentence_transformers import CrossEncoder

class Reranker:
    """
    Rerank retrieved documents for better precision.
    
    Two-stage retrieval:
    1. Fast retrieval with bi-encoder (embedding similarity)
    2. Accurate reranking with cross-encoder (joint encoding)
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict]:
        """Rerank documents by relevance to query."""
        # Create query-document pairs
        pairs = [(query, doc) for doc in documents]
        
        # Score with cross-encoder
        scores = self.model.predict(pairs)
        
        # Sort by score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [
            {"text": doc, "score": float(score)}
            for doc, score in scored_docs[:top_k]
        ]


class LLMReranker:
    """Use LLM to rerank documents."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict]:
        """Rerank using LLM scoring."""
        prompt = f"""Rate the relevance of each document to the query on a scale of 1-10.

Query: {query}

"""
        for i, doc in enumerate(documents):
            prompt += f"Document {i+1}: {doc[:500]}...\n\n"
        
        prompt += "Relevance scores (format: Doc1: X, Doc2: Y, ...):"
        
        response = self.llm.generate(prompt)
        scores = self._parse_scores(response, len(documents))
        
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [
            {"text": doc, "score": score}
            for doc, score in scored_docs[:top_k]
        ]
```

### Advanced RAG Patterns

```python
class AdvancedRAG:
    """Advanced RAG with multiple techniques."""
    
    def __init__(self, embedding_model, vector_store, llm, reranker=None):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm = llm
        self.reranker = reranker
        self.query_transformer = QueryTransformer(llm)
    
    def query_with_hyde(self, question: str, k: int = 5) -> Dict:
        """HyDE: Hypothetical Document Embeddings."""
        # Generate hypothetical answer
        hypothetical = self.query_transformer.hypothetical_document(question)
        
        # Retrieve using hypothetical document
        docs = self.retrieve(hypothetical, k=k)
        
        return self.generate_answer(question, docs)
    
    def query_with_expansion(self, question: str, k: int = 5) -> Dict:
        """Multi-query retrieval with query expansion."""
        # Generate query variations
        queries = self.query_transformer.expand_query(question)
        
        # Retrieve for each query
        all_docs = []
        seen_texts = set()
        
        for q in queries:
            docs = self.retrieve(q, k=k)
            for doc in docs:
                if doc["text"] not in seen_texts:
                    all_docs.append(doc)
                    seen_texts.add(doc["text"])
        
        # Rerank combined results
        if self.reranker:
            all_docs = self.reranker.rerank(question, [d["text"] for d in all_docs], k)
        
        return self.generate_answer(question, all_docs[:k])
    
    def query_with_decomposition(self, question: str, k: int = 3) -> Dict:
        """Answer complex questions by decomposition."""
        # Decompose into sub-questions
        sub_questions = self.query_transformer.decompose_query(question)
        
        # Answer each sub-question
        sub_answers = []
        for sub_q in sub_questions:
            docs = self.retrieve(sub_q, k=k)
            answer = self.generate_answer(sub_q, docs)
            sub_answers.append(f"Q: {sub_q}\nA: {answer['answer']}")
        
        # Synthesize final answer
        synthesis_prompt = f"""Based on these sub-answers, provide a comprehensive answer 
to the original question.

Original question: {question}

Sub-answers:
{chr(10).join(sub_answers)}

Final answer:"""
        
        final_answer = self.llm.generate(synthesis_prompt)
        
        return {
            "answer": final_answer,
            "sub_questions": sub_questions,
            "sub_answers": sub_answers
        }
```

---

## 6. Evaluation and Optimization

### RAG Evaluation Metrics

```python
"""
RAG Evaluation Dimensions:

1. Retrieval Quality
   - Recall@K: % of relevant docs in top K
   - Precision@K: % of top K that are relevant
   - MRR: Mean Reciprocal Rank
   - NDCG: Normalized Discounted Cumulative Gain

2. Generation Quality
   - Faithfulness: Is answer supported by context?
   - Answer Relevance: Does answer address question?
   - Correctness: Is answer factually correct?

3. End-to-End
   - Answer accuracy
   - Hallucination rate
   - Source attribution accuracy
"""

class RAGEvaluator:
    """Evaluate RAG system performance."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def evaluate_retrieval(self, query: str, retrieved_docs: List[str], 
                          relevant_docs: List[str]) -> Dict:
        """Evaluate retrieval quality."""
        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)
        
        # Precision and Recall
        true_positives = len(retrieved_set & relevant_set)
        precision = true_positives / len(retrieved_docs) if retrieved_docs else 0
        recall = true_positives / len(relevant_docs) if relevant_docs else 0
        
        # MRR
        mrr = 0
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_set:
                mrr = 1 / (i + 1)
                break
        
        return {
            "precision": precision,
            "recall": recall,
            "mrr": mrr,
            "f1": 2 * precision * recall / (precision + recall + 1e-6)
        }
    
    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """Check if answer is grounded in context."""
        prompt = f"""Rate how well the answer is supported by the context.
Score from 0 (not supported) to 1 (fully supported).

Context: {context}

Answer: {answer}

Score (0-1):"""
        
        response = self.llm.generate(prompt)
        try:
            return float(re.search(r"(\d\.?\d*)", response).group(1))
        except:
            return 0.5
    
    def evaluate_relevance(self, question: str, answer: str) -> float:
        """Check if answer addresses the question."""
        prompt = f"""Rate how well the answer addresses the question.
Score from 0 (irrelevant) to 1 (perfectly relevant).

Question: {question}

Answer: {answer}

Score (0-1):"""
        
        response = self.llm.generate(prompt)
        try:
            return float(re.search(r"(\d\.?\d*)", response).group(1))
        except:
            return 0.5
    
    def evaluate_correctness(self, question: str, answer: str, 
                            ground_truth: str) -> float:
        """Check answer correctness against ground truth."""
        prompt = f"""Compare the answer to the ground truth and rate correctness.
Score from 0 (completely wrong) to 1 (completely correct).

Question: {question}

Ground Truth: {ground_truth}

Answer: {answer}

Score (0-1):"""
        
        response = self.llm.generate(prompt)
        try:
            return float(re.search(r"(\d\.?\d*)", response).group(1))
        except:
            return 0.5


# RAGAS evaluation
"""
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)
"""
```

---

## 7. Interview Questions

```python
# Q1: What are the key components of a RAG system?
"""
1. Document Store: Raw documents/knowledge base
2. Chunker: Split documents into retrievable units
3. Embedding Model: Convert text to vectors
4. Vector Database: Store and search embeddings
5. Retriever: Find relevant chunks for query
6. Reranker (optional): Improve retrieval precision
7. Generator (LLM): Produce final answer
8. Prompt Template: Format context for LLM

Pipeline: Query → Embed → Retrieve → (Rerank) → Generate
"""


# Q2: How do you choose chunk size for RAG?
"""
Considerations:

Too small:
- Loses context
- More chunks to retrieve/process
- May split important info

Too large:
- Includes irrelevant info
- Reduces retrieval precision
- May exceed context window

Guidelines:
- Typical: 500-1500 characters
- Overlap: 10-20% of chunk size
- Consider document structure
- Test on your specific use case

Best practice:
- Start with ~1000 chars, 200 overlap
- Use semantic boundaries (paragraphs, sections)
- Evaluate retrieval quality
"""


# Q3: When would you use hybrid search over dense retrieval?
"""
Dense retrieval (embeddings only):
- Semantic similarity focus
- Good for paraphrased queries
- May miss exact matches

Hybrid (dense + sparse):
- Combines semantic + keyword matching
- Better for technical terms, names, codes
- More robust overall

Use hybrid when:
- Domain has specific terminology
- Exact matches are important
- Users use varied query styles

Typical weighting: 0.5-0.7 dense, 0.3-0.5 sparse
"""


# Q4: Explain the HyDE technique
"""
HyDE: Hypothetical Document Embeddings

Process:
1. Generate hypothetical answer to query
2. Embed the hypothetical answer
3. Retrieve using hypothetical embedding

Why it works:
- Hypothetical answer is in "document space"
- Better semantic match with actual documents
- Query often short, documents are longer

When to use:
- Complex questions
- Abstract queries
- When direct query retrieval is poor

Limitations:
- Extra LLM call (latency/cost)
- Hypothetical may be wrong direction
"""


# Q5: How do you evaluate RAG system quality?
"""
Retrieval metrics:
- Recall@K: Find all relevant docs?
- Precision@K: Are retrieved docs relevant?
- MRR: Where does first relevant doc appear?

Generation metrics:
- Faithfulness: Grounded in retrieved context?
- Answer relevance: Addresses the question?
- Correctness: Factually accurate?

End-to-end:
- Answer accuracy (vs ground truth)
- Hallucination rate
- User satisfaction

Tools: RAGAS, LangSmith, custom evaluation
"""


# Q6: What causes hallucinations in RAG and how to reduce them?
"""
Causes:
1. Poor retrieval (wrong context)
2. LLM ignores context
3. Conflicting information in context
4. Question unanswerable from context

Mitigation:
1. Improve retrieval quality
   - Better embeddings, reranking
   
2. Better prompting
   - "Only answer from context"
   - "Say 'I don't know' if unsure"
   
3. Output validation
   - Check answer against sources
   - Require citations
   
4. Context selection
   - Filter low-relevance chunks
   - Check for contradictions
"""


# Q7: Compare different embedding models for RAG
"""
Key factors:
1. Quality (MTEB benchmark)
2. Dimension (storage/speed)
3. Max sequence length
4. Cost (for API models)

Open source (quality order):
- BGE-large-en-v1.5 (1024 dim)
- E5-large-v2 (1024 dim)
- all-mpnet-base-v2 (768 dim)
- all-MiniLM-L6-v2 (384 dim, fastest)

Commercial:
- OpenAI text-embedding-3-large (best quality)
- Cohere embed-v3 (good multilingual)

Recommendation:
- Prototype: all-MiniLM-L6-v2 (fast, free)
- Production: BGE or E5 (open) or OpenAI (commercial)
"""


# Q8: What is the role of reranking in RAG?
"""
Two-stage retrieval:

Stage 1 - Retriever (bi-encoder):
- Fast: Independent encoding of query/doc
- Retrieves top ~100 candidates
- May miss nuanced relevance

Stage 2 - Reranker (cross-encoder):
- Accurate: Joint encoding of query+doc
- Slower: O(n) for n candidates
- Reorders top candidates

Benefits:
- Better precision in final results
- Catches semantic nuances
- Fixes retriever mistakes

Common rerankers:
- cross-encoder/ms-marco-MiniLM
- Cohere Rerank
- LLM-based reranking
"""


# Q9: How do you handle multi-hop questions in RAG?
"""
Multi-hop: Questions requiring multiple retrieval steps

Example: "What is the population of the city where Apple was founded?"
- Step 1: Where was Apple founded? → Cupertino
- Step 2: Population of Cupertino? → Answer

Approaches:

1. Query decomposition
   - Break into sub-questions
   - Answer sequentially
   
2. Iterative retrieval
   - Retrieve → Generate partial answer
   - Use partial answer for next retrieval
   
3. Graph-based RAG
   - Build knowledge graph from documents
   - Traverse graph for multi-hop reasoning

4. Chain-of-thought RAG
   - Let LLM reason through steps
   - Retrieve at each reasoning step
"""


# Q10: How do you scale RAG for production?
"""
Scaling considerations:

1. Indexing
   - Batch document processing
   - Incremental updates
   - Distributed embedding generation

2. Storage
   - Approximate nearest neighbor (HNSW, IVF)
   - Managed vector DB (Pinecone, Weaviate)
   - Sharding for large datasets

3. Retrieval
   - Caching frequent queries
   - Async retrieval
   - Load balancing

4. Generation
   - Model serving (vLLM, TGI)
   - Streaming responses
   - Request batching

5. Monitoring
   - Latency tracking
   - Quality metrics
   - Cost monitoring
"""
```
