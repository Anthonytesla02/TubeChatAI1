import numpy as np
import pickle
import logging
import os
from typing import List, Dict, Tuple, Optional
from app import db
from models import Transcript, TranscriptChunk

# Try to import optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available, using TF-IDF fallback")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("faiss not available, using sklearn fallback")

# Global model instance (lazy loaded)
_embedding_model = None
_faiss_indices = {}  # Cache for FAISS indices by transcript_id

def get_embedding_model():
    """Get or create the sentence transformer model"""
    global _embedding_model
    if _embedding_model is None:
        try:
            # Use a lightweight but effective model
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logging.info("Loaded sentence transformer model: all-MiniLM-L6-v2")
        except Exception as e:
            logging.error(f"Failed to load embedding model: {e}")
            raise
    return _embedding_model

def generate_embedding(text: str) -> np.ndarray:
    """Generate embedding for a single text"""
    try:
        model = get_embedding_model()
        # Generate embedding and normalize for cosine similarity
        embedding = model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        # Normalize for cosine similarity using inner product
        faiss.normalize_L2(embedding)
        return embedding[0]  # Return single vector
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        return np.zeros(384)  # Return zero vector as fallback (all-MiniLM-L6-v2 has 384 dimensions)

def generate_embeddings_batch(texts: List[str]) -> np.ndarray:
    """Generate embeddings for multiple texts efficiently"""
    try:
        model = get_embedding_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        return embeddings
    except Exception as e:
        logging.error(f"Error generating batch embeddings: {e}")
        return np.zeros((len(texts), 384))  # Return zero vectors as fallback

def store_embedding(chunk: TranscriptChunk, embedding: np.ndarray):
    """Store embedding in database as binary data"""
    try:
        # Convert to float32 for efficiency and store as bytes
        embedding_bytes = embedding.astype(np.float32).tobytes()
        chunk.embedding_vector = embedding_bytes
        logging.debug(f"Stored embedding for chunk {chunk.id}, size: {len(embedding_bytes)} bytes")
    except Exception as e:
        logging.error(f"Error storing embedding: {e}")
        chunk.embedding_vector = None

def load_embedding(chunk: TranscriptChunk) -> Optional[np.ndarray]:
    """Load embedding from database binary data"""
    try:
        if chunk.embedding_vector is None:
            return None
        # Convert from bytes back to numpy array
        embedding = np.frombuffer(chunk.embedding_vector, dtype=np.float32)
        return embedding
    except Exception as e:
        logging.error(f"Error loading embedding for chunk {chunk.id}: {e}")
        return None

def build_faiss_index(transcript_id: int) -> Optional[faiss.Index]:
    """Build or retrieve FAISS index for a transcript"""
    try:
        # Check cache first
        if transcript_id in _faiss_indices:
            return _faiss_indices[transcript_id]
        
        # Get all chunks with embeddings for this transcript
        chunks = TranscriptChunk.query.filter_by(transcript_id=transcript_id).filter(
            TranscriptChunk.embedding_vector.isnot(None)
        ).order_by(TranscriptChunk.chunk_index).all()
        
        if not chunks:
            logging.warning(f"No chunks with embeddings found for transcript {transcript_id}")
            return None
        
        # Load embeddings
        embeddings = []
        for chunk in chunks:
            embedding = load_embedding(chunk)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                logging.warning(f"Could not load embedding for chunk {chunk.id}")
        
        if not embeddings:
            logging.warning(f"No valid embeddings found for transcript {transcript_id}")
            return None
        
        # Create FAISS index
        embeddings_array = np.vstack(embeddings)
        dimension = embeddings_array.shape[1]
        
        # Use IndexFlatIP for inner product (good for normalized vectors)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_array)
        
        # Cache the index
        _faiss_indices[transcript_id] = index
        
        logging.info(f"Built FAISS index for transcript {transcript_id} with {len(embeddings)} chunks")
        return index
        
    except Exception as e:
        logging.error(f"Error building FAISS index for transcript {transcript_id}: {e}")
        return None

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    try:
        if np.all(a == 0) or np.all(b == 0):
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    except Exception as e:
        logging.error(f"Error calculating cosine similarity: {e}")
        return 0.0

def search_similar_chunks_vector(transcript_id: int, query: str, top_k: int = 5) -> List[Dict]:
    """Search for similar chunks using vector similarity with FAISS"""
    try:
        # Generate query embedding
        query_embedding = generate_embedding(query)
        
        # Get FAISS index
        index = build_faiss_index(transcript_id)
        if index is None:
            logging.warning(f"No FAISS index available for transcript {transcript_id}")
            return []
        
        # Get chunks for this transcript (in order)
        chunks = TranscriptChunk.query.filter_by(transcript_id=transcript_id).filter(
            TranscriptChunk.embedding_vector.isnot(None)
        ).order_by(TranscriptChunk.chunk_index).all()
        
        if not chunks:
            return []
        
        # Search using FAISS
        query_vector = query_embedding.reshape(1, -1)
        similarities, indices = index.search(query_vector, min(top_k, len(chunks)))
        
        # Build results
        results = []
        for i, (similarity, chunk_idx) in enumerate(zip(similarities[0], indices[0])):
            if chunk_idx < len(chunks):
                chunk = chunks[chunk_idx]
                results.append({
                    'chunk_text': chunk.chunk_text,
                    'chunk_index': chunk.chunk_index,
                    'similarity': float(similarity),
                    'start_time': chunk.start_time,
                    'end_time': chunk.end_time,
                    'chunk_id': chunk.id
                })
        
        # Filter out chunks with very low similarity (< 0.3 for embeddings)
        relevant_chunks = [chunk for chunk in results if chunk['similarity'] > 0.3]
        
        logging.debug(f"Vector search returned {len(relevant_chunks)} relevant chunks for query: {query[:50]}...")
        return relevant_chunks
        
    except Exception as e:
        logging.error(f"Error in vector similarity search: {e}")
        return []

def clear_faiss_cache(transcript_id: Optional[int] = None):
    """Clear FAISS index cache"""
    global _faiss_indices
    if transcript_id is None:
        _faiss_indices.clear()
        logging.info("Cleared all FAISS indices from cache")
    elif transcript_id in _faiss_indices:
        del _faiss_indices[transcript_id]
        logging.info(f"Cleared FAISS index for transcript {transcript_id}")

def get_embedding_stats(transcript_id: int) -> Dict:
    """Get statistics about embeddings for a transcript"""
    try:
        chunks = TranscriptChunk.query.filter_by(transcript_id=transcript_id).all()
        chunks_with_embeddings = TranscriptChunk.query.filter_by(transcript_id=transcript_id).filter(
            TranscriptChunk.embedding_vector.isnot(None)
        ).all()
        
        return {
            'total_chunks': len(chunks),
            'chunks_with_embeddings': len(chunks_with_embeddings),
            'embedding_coverage': len(chunks_with_embeddings) / len(chunks) if chunks else 0,
            'faiss_index_cached': transcript_id in _faiss_indices
        }
    except Exception as e:
        logging.error(f"Error getting embedding stats: {e}")
        return {'error': str(e)}