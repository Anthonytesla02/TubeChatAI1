import numpy as np
import logging
from app import db
from models import Transcript, TranscriptChunk
import re
from typing import List, Dict, Tuple
from collections import Counter
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class ImprovedRAGService:
    """
    Improved RAG service with better text similarity using TF-IDF and cosine similarity
    This provides much better results than simple Jaccard similarity
    """
    
    def __init__(self):
        self.vectorizers = {}  # Cache vectorizers per transcript
        self.tfidf_matrices = {}  # Cache TF-IDF matrices per transcript
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
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
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better TF-IDF results
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:]', ' ', text)
        
        return text.strip()
    
    def build_tfidf_index(self, transcript_id: int):
        """
        Build TF-IDF index for a transcript
        """
        try:
            # Get all chunks for this transcript
            chunks = TranscriptChunk.query.filter_by(transcript_id=transcript_id).order_by(
                TranscriptChunk.chunk_index
            ).all()
            
            if not chunks:
                logging.warning(f"No chunks found for transcript {transcript_id}")
                return False
            
            # Preprocess chunk texts
            chunk_texts = [self.preprocess_text(chunk.chunk_text) for chunk in chunks]
            
            # Create TF-IDF vectorizer with optimized parameters
            vectorizer = TfidfVectorizer(
                max_features=10000,  # Limit vocabulary size
                ngram_range=(1, 2),  # Use unigrams and bigrams
                stop_words='english',  # Remove common English stop words
                min_df=1,  # Minimum document frequency
                max_df=0.8,  # Maximum document frequency
                sublinear_tf=True,  # Use sublinear TF scaling
                norm='l2'  # L2 normalization
            )
            
            # Fit and transform the texts
            tfidf_matrix = vectorizer.fit_transform(chunk_texts)
            
            # Cache the vectorizer and matrix
            self.vectorizers[transcript_id] = vectorizer
            self.tfidf_matrices[transcript_id] = tfidf_matrix
            
            logging.info(f"Built TF-IDF index for transcript {transcript_id} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logging.error(f"Error building TF-IDF index for transcript {transcript_id}: {e}")
            return False
    
    def search_similar_chunks_tfidf(self, transcript_id: int, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar chunks using TF-IDF and cosine similarity
        """
        try:
            # Build index if not cached
            if transcript_id not in self.vectorizers:
                if not self.build_tfidf_index(transcript_id):
                    return []
            
            vectorizer = self.vectorizers[transcript_id]
            tfidf_matrix = self.tfidf_matrices[transcript_id]
            
            # Get chunks for this transcript
            chunks = TranscriptChunk.query.filter_by(transcript_id=transcript_id).order_by(
                TranscriptChunk.chunk_index
            ).all()
            
            if not chunks:
                return []
            
            # Preprocess and vectorize the query
            query_processed = self.preprocess_text(query)
            query_vector = vectorizer.transform([query_processed])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if idx < len(chunks) and similarities[idx] > 0.05:  # Filter very low similarities
                    chunk = chunks[idx]
                    results.append({
                        'chunk_text': chunk.chunk_text,
                        'chunk_index': chunk.chunk_index,
                        'similarity': float(similarities[idx]),
                        'start_time': chunk.start_time,
                        'end_time': chunk.end_time,
                        'chunk_id': chunk.id
                    })
            
            logging.debug(f"TF-IDF search returned {len(results)} chunks for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logging.error(f"Error in TF-IDF similarity search: {e}")
            return []
    
    def process_transcript_for_rag(self, transcript_id: int):
        """
        Process a transcript for RAG: create chunks and build TF-IDF index
        """
        try:
            transcript = Transcript.query.get(transcript_id)
            if not transcript:
                raise ValueError(f"Transcript {transcript_id} not found")
            
            # Delete existing chunks if any
            TranscriptChunk.query.filter_by(transcript_id=transcript_id).delete()
            
            # Clear cached indices
            if transcript_id in self.vectorizers:
                del self.vectorizers[transcript_id]
            if transcript_id in self.tfidf_matrices:
                del self.tfidf_matrices[transcript_id]
            
            # Chunk the transcript
            chunks = self.chunk_text(transcript.transcript_text)
            logging.info(f"Created {len(chunks)} text chunks for transcript {transcript_id}")
            
            # Create chunk records
            chunk_objects = []
            for i, chunk_content in enumerate(chunks):
                try:
                    chunk = TranscriptChunk()
                    chunk.transcript_id = transcript_id
                    chunk.chunk_text = chunk_content
                    chunk.chunk_index = i
                    chunk.embedding_vector = None  # Not using embeddings in this version
                    
                    db.session.add(chunk)
                    chunk_objects.append(chunk)
                    
                    # Commit every 10 chunks to avoid memory issues
                    if i % 10 == 0:
                        db.session.commit()
                        
                except Exception as e:
                    logging.error(f"Error processing chunk {i}: {e}")
                    continue
            
            db.session.commit()
            logging.info(f"Successfully created {len(chunk_objects)} chunk records")
            
            # Build TF-IDF index
            if chunk_objects:
                self.build_tfidf_index(transcript_id)
            
            # Update transcript status
            transcript.embeddings_processed = True
            db.session.commit()
            
            logging.info(f"Successfully processed transcript {transcript_id} for RAG")
            
        except Exception as e:
            db.session.rollback()
            logging.error(f"Error processing transcript {transcript_id} for RAG: {e}")
            raise
    
    def get_context_for_query(self, transcript_id: int, query: str, max_context_length: int = 2000) -> str:
        """
        Get relevant context for a query, combining multiple chunks if needed
        """
        try:
            similar_chunks = self.search_similar_chunks_tfidf(transcript_id, query, top_k=5)
            
            if not similar_chunks:
                return ""
            
            # Combine chunks into context, respecting max length
            context_parts = []
            current_length = 0
            
            for i, chunk in enumerate(similar_chunks):
                chunk_text = chunk['chunk_text']
                
                # Add citation and separator
                citation = f"\n\n[Chunk {chunk['chunk_index']}] "
                if chunk.get('start_time') and chunk.get('end_time'):
                    citation += f"({chunk['start_time']:.1f}s-{chunk['end_time']:.1f}s) "
                citation += f"(Relevance: {chunk['similarity']:.2f})\n"
                
                # Add separator if not first chunk
                if context_parts:
                    if current_length + len(citation) + len(chunk_text) > max_context_length:
                        break
                    context_parts.append(citation)
                    current_length += len(citation)
                else:
                    context_parts.append(citation)
                    current_length += len(citation)
                
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

# Global instance
improved_rag = ImprovedRAGService()