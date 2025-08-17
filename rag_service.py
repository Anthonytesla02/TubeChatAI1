import numpy as np
import pickle
import logging
from app import db
from models import Transcript, TranscriptChunk
import re
from typing import List, Dict, Tuple
from collections import Counter
import math

# Simple text-based similarity for now (will add embeddings later)
# This provides basic functionality until sentence-transformers is properly configured

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
    Process a transcript for RAG: create chunks (simplified text-based version)
    """
    try:
        transcript = Transcript.query.get(transcript_id)
        if not transcript:
            raise ValueError(f"Transcript {transcript_id} not found")
        
        # Delete existing chunks if any
        TranscriptChunk.query.filter_by(transcript_id=transcript_id).delete()
        
        # Chunk the transcript
        chunks = chunk_text(transcript.transcript_text)
        
        # Process each chunk
        for i, chunk_content in enumerate(chunks):
            try:
                # Create chunk record (without embeddings for now)
                chunk = TranscriptChunk()
                chunk.transcript_id = transcript_id
                chunk.chunk_text = chunk_content
                chunk.chunk_index = i
                chunk.embedding_vector = None  # Will be added when we configure embeddings
                
                db.session.add(chunk)
                
                # Commit every 10 chunks to avoid memory issues
                if i % 10 == 0:
                    db.session.commit()
                    
            except Exception as e:
                logging.error(f"Error processing chunk {i}: {e}")
                continue
        
        db.session.commit()
        logging.info(f"Successfully processed {len(chunks)} chunks for transcript {transcript_id}")
        
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
    Search for similar chunks using simple text similarity
    Returns list of dictionaries with chunk text and similarity scores
    """
    try:
        # Get all chunks for this transcript
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
                    'end_time': chunk.end_time
                })
                
            except Exception as e:
                logging.error(f"Error processing chunk {chunk.id}: {e}")
                continue
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Filter out chunks with very low similarity (< 0.1)
        relevant_chunks = [chunk for chunk in similarities if chunk['similarity'] > 0.1]
        
        return relevant_chunks[:top_k]
        
    except Exception as e:
        logging.error(f"Error searching similar chunks: {e}")
        return []

def get_context_for_query(transcript_id: int, query: str, max_context_length: int = 2000) -> str:
    """
    Get relevant context for a query, combining multiple chunks if needed
    """
    try:
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
