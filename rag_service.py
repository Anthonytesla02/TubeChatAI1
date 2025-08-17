import numpy as np
import pickle
import logging
from app import db
from models import Transcript, TranscriptChunk
import re
from typing import List, Dict, Tuple
from collections import Counter
import math

# Import the improved RAG service
try:
    from improved_rag_service import improved_rag
    IMPROVED_RAG_AVAILABLE = True
    logging.info("Improved RAG service loaded successfully")
except ImportError as e:
    logging.warning(f"Improved RAG service not available: {e}")
    IMPROVED_RAG_AVAILABLE = False

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for better context preservation
    """
    # Clean the text
    text = re.sub(r'\s+', ' ', text.strip())
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If this is not the last chunk, try to end at a sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            last_part = text[max(start, end-100):end]
            sentence_end = max(
                last_part.rfind('.'),
                last_part.rfind('!'),
                last_part.rfind('?')
            )
            
            if sentence_end != -1:
                # Adjust end to include the sentence ending
                end = max(start, end-100) + sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        
        # Prevent infinite loop
        if start >= len(text):
            break
    
    return chunks

def process_transcript_for_rag(transcript_id: int):
    """
    Process a transcript for RAG: use improved RAG service if available, fallback to basic
    """
    try:
        if IMPROVED_RAG_AVAILABLE:
            # Use the improved TF-IDF based service
            improved_rag.process_transcript_for_rag(transcript_id)
            return
        
        # Fallback to basic processing
        transcript = Transcript.query.get(transcript_id)
        if not transcript:
            raise ValueError(f"Transcript {transcript_id} not found")
        
        # Delete existing chunks if any
        TranscriptChunk.query.filter_by(transcript_id=transcript_id).delete()
        
        # Chunk the transcript
        chunks = chunk_text(transcript.transcript_text)
        logging.info(f"Created {len(chunks)} text chunks for transcript {transcript_id}")
        
        # Create chunk records
        chunk_objects = []
        for i, chunk_content in enumerate(chunks):
            try:
                chunk = TranscriptChunk()
                chunk.transcript_id = transcript_id
                chunk.chunk_text = chunk_content
                chunk.chunk_index = i
                chunk.embedding_vector = None
                
                db.session.add(chunk)
                chunk_objects.append(chunk)
                
                # Commit every 10 chunks to avoid memory issues
                if i % 10 == 0:
                    db.session.commit()
                    
            except Exception as e:
                logging.error(f"Error processing chunk {i}: {e}")
                continue
        
        db.session.commit()
        
        # Update transcript status
        transcript.embeddings_processed = True
        db.session.commit()
        
        logging.info(f"Successfully processed transcript {transcript_id} for RAG")
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error processing transcript {transcript_id} for RAG: {e}")
        raise

def calculate_text_similarity(query: str, text: str) -> float:
    """
    Calculate simple text similarity using TF-IDF-like approach
    """
    def get_words(text):
        return re.findall(r'\b\w+\b', text.lower())
    
    query_words = set(get_words(query))
    text_words = set(get_words(text))
    
    if not query_words or not text_words:
        return 0.0
    
    # Simple Jaccard similarity
    intersection = len(query_words.intersection(text_words))
    union = len(query_words.union(text_words))
    
    return intersection / union if union > 0 else 0.0

def search_similar_chunks(transcript_id: int, query: str, top_k: int = 5) -> List[Dict]:
    """
    Search for similar chunks using improved TF-IDF similarity (with basic text fallback)
    Returns list of dictionaries with chunk text and similarity scores
    """
    try:
        # Try improved TF-IDF search first if available
        if IMPROVED_RAG_AVAILABLE:
            try:
                results = improved_rag.search_similar_chunks_tfidf(transcript_id, query, top_k)
                if results:
                    logging.debug(f"TF-IDF search returned {len(results)} chunks with similarities: {[r['similarity'] for r in results[:3]]}")
                    return results
                else:
                    logging.warning(f"TF-IDF search returned no results for transcript {transcript_id}, falling back to basic text similarity")
            except Exception as e:
                logging.error(f"TF-IDF search failed: {e}, falling back to basic text similarity")
        
        # Fallback to basic text-based similarity
        logging.info(f"Using basic text similarity search for transcript {transcript_id}")
        chunks = TranscriptChunk.query.filter_by(transcript_id=transcript_id).all()
        
        if not chunks:
            logging.warning(f"No chunks found for transcript {transcript_id}")
            return []
        
        # Calculate similarities
        similarities = []
        
        for chunk in chunks:
            try:
                # Calculate text similarity
                similarity = calculate_text_similarity(query, chunk.chunk_text)
                
                similarities.append({
                    'chunk_text': chunk.chunk_text,
                    'chunk_index': chunk.chunk_index,
                    'similarity': float(similarity),
                    'start_time': chunk.start_time,
                    'end_time': chunk.end_time,
                    'chunk_id': chunk.id
                })
                
            except Exception as e:
                logging.error(f"Error processing chunk {chunk.id}: {e}")
                continue
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Filter out chunks with very low similarity
        relevant_chunks = [chunk for chunk in similarities if chunk['similarity'] > 0.1]
        
        logging.debug(f"Basic text search returned {len(relevant_chunks)} chunks with similarities: {[r['similarity'] for r in relevant_chunks[:3]]}")
        return relevant_chunks[:top_k]
        
    except Exception as e:
        logging.error(f"Error searching similar chunks: {e}")
        return []

def get_context_for_query(transcript_id: int, query: str, max_context_length: int = 2000) -> str:
    """
    Get relevant context for a query, combining multiple chunks if needed
    """
    try:
        # Use improved RAG service if available for better context formatting
        if IMPROVED_RAG_AVAILABLE:
            return improved_rag.get_context_for_query(transcript_id, query, max_context_length)
        
        # Fallback to basic context building
        similar_chunks = search_similar_chunks(transcript_id, query, top_k=5)
        
        if not similar_chunks:
            return ""
        
        # Combine chunks into context, respecting max length
        context_parts = []
        current_length = 0
        
        for chunk in similar_chunks:
            chunk_text = chunk['chunk_text']
            
            # Add separator if not first chunk
            if context_parts:
                separator = "\n\n"
                if current_length + len(separator) + len(chunk_text) > max_context_length:
                    break
                context_parts.append(separator)
                current_length += len(separator)
            
            # Check if adding this chunk would exceed limit
            if current_length + len(chunk_text) > max_context_length:
                # Truncate the chunk to fit
                remaining_space = max_context_length - current_length
                if remaining_space > 100:  # Only add if we have meaningful space
                    context_parts.append(chunk_text[:remaining_space-3] + "...")
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "".join(context_parts)
        
    except Exception as e:
        logging.error(f"Error getting context for query: {e}")
        return ""
