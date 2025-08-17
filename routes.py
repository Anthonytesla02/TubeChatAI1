from flask import render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import login_required, current_user
from app import app, db
from models import User, Transcript, ChatSession, ChatMessage
from auth import auth_bp
from youtube_service import extract_youtube_transcript
from rag_service import process_transcript_for_rag, search_similar_chunks
from mistral_service import generate_chat_response
import json
import logging

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/auth')

@app.route('/')
def index():
    """Landing page - shows dashboard if logged in, otherwise landing page"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard showing transcripts and chat sessions"""
    transcripts = Transcript.query.filter_by(user_id=current_user.id).order_by(Transcript.created_at.desc()).all()
    recent_sessions = ChatSession.query.filter_by(user_id=current_user.id).order_by(ChatSession.created_at.desc()).limit(5).all()
    
    return render_template('dashboard.html', 
                         transcripts=transcripts, 
                         recent_sessions=recent_sessions)

@app.route('/add_transcript', methods=['POST'])
@login_required
def add_transcript():
    """Add a new YouTube transcript"""
    youtube_url = request.form.get('youtube_url', '').strip()
    
    if not youtube_url:
        flash('Please provide a YouTube URL', 'error')
        return redirect(url_for('dashboard'))
    
    try:
        # Extract transcript
        transcript_data = extract_youtube_transcript(youtube_url)
        
        # Check if transcript already exists
        existing = Transcript.query.filter_by(
            user_id=current_user.id,
            video_id=transcript_data['video_id']
        ).first()
        
        if existing:
            flash('This video transcript already exists in your collection', 'warning')
            return redirect(url_for('dashboard'))
        
        # Create new transcript
        transcript = Transcript(
            user_id=current_user.id,
            youtube_url=youtube_url,
            video_title=transcript_data['title'],
            video_id=transcript_data['video_id'],
            transcript_text=transcript_data['transcript']
        )
        
        db.session.add(transcript)
        db.session.commit()
        
        # Process for RAG in background (in a real app, use Celery)
        try:
            process_transcript_for_rag(transcript.id)
            transcript.embeddings_processed = True
            db.session.commit()
            flash(f'Successfully added transcript for "{transcript_data["title"]}"', 'success')
        except Exception as e:
            logging.error(f"Error processing embeddings: {e}")
            flash(f'Transcript added but embeddings processing failed: {str(e)}', 'warning')
        
    except Exception as e:
        logging.error(f"Error adding transcript: {e}")
        flash(f'Error processing YouTube video: {str(e)}', 'error')
    
    return redirect(url_for('dashboard'))

@app.route('/chat/<int:transcript_id>')
@login_required
def chat(transcript_id):
    """Chat interface for a specific transcript"""
    transcript = Transcript.query.filter_by(id=transcript_id, user_id=current_user.id).first_or_404()
    
    if not transcript.embeddings_processed:
        flash('Transcript is still being processed. Please try again in a moment.', 'info')
        return redirect(url_for('dashboard'))
    
    # Get or create chat session
    session_id = request.args.get('session_id')
    if session_id:
        chat_session = ChatSession.query.filter_by(
            id=session_id, 
            user_id=current_user.id,
            transcript_id=transcript_id
        ).first()
    else:
        chat_session = None
    
    if not chat_session:
        # Create new session
        chat_session = ChatSession(
            user_id=current_user.id,
            transcript_id=transcript_id,
            session_name=f"Chat about {transcript.video_title[:50]}..."
        )
        db.session.add(chat_session)
        db.session.commit()
    
    # Get chat messages
    messages = ChatMessage.query.filter_by(session_id=chat_session.id).order_by(ChatMessage.timestamp).all()
    
    return render_template('chat.html', 
                         transcript=transcript, 
                         chat_session=chat_session,
                         messages=messages)

@app.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    """API endpoint for chat messages"""
    data = request.get_json()
    
    if not data or 'message' not in data or 'session_id' not in data:
        return jsonify({'error': 'Invalid request data'}), 400
    
    session_id = data['session_id']
    user_message = data['message'].strip()
    
    if not user_message:
        return jsonify({'error': 'Message cannot be empty'}), 400
    
    # Verify session belongs to user
    chat_session = ChatSession.query.filter_by(
        id=session_id,
        user_id=current_user.id
    ).first()
    
    if not chat_session:
        return jsonify({'error': 'Chat session not found'}), 404
    
    try:
        # Save user message
        user_msg = ChatMessage(
            session_id=session_id,
            message_type='user',
            content=user_message
        )
        db.session.add(user_msg)
        db.session.commit()
        
        # Search for relevant chunks
        relevant_chunks = search_similar_chunks(chat_session.transcript_id, user_message)
        
        # Generate response using Mistral AI
        response_data = generate_chat_response(user_message, relevant_chunks)
        
        # Save assistant message
        assistant_msg = ChatMessage(
            session_id=session_id,
            message_type='assistant',
            content=response_data['response'],
            context_chunks=json.dumps(response_data['context_chunks']),
            confidence_score=response_data.get('confidence_score', 0.0)
        )
        db.session.add(assistant_msg)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'response': response_data['response'],
            'context_chunks': response_data['context_chunks'],
            'confidence_score': response_data.get('confidence_score', 0.0)
        })
        
    except Exception as e:
        logging.error(f"Error in chat API: {e}")
        return jsonify({'error': f'Failed to process message: {str(e)}'}), 500

@app.route('/delete_transcript/<int:transcript_id>', methods=['POST'])
@login_required
def delete_transcript(transcript_id):
    """Delete a transcript and all associated data"""
    transcript = Transcript.query.filter_by(id=transcript_id, user_id=current_user.id).first_or_404()
    
    try:
        db.session.delete(transcript)
        db.session.commit()
        flash('Transcript deleted successfully', 'success')
    except Exception as e:
        logging.error(f"Error deleting transcript: {e}")
        flash('Error deleting transcript', 'error')
    
    return redirect(url_for('dashboard'))

@app.route('/delete_session/<int:session_id>', methods=['POST'])
@login_required
def delete_session(session_id):
    """Delete a chat session"""
    chat_session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first_or_404()
    
    try:
        db.session.delete(chat_session)
        db.session.commit()
        flash('Chat session deleted successfully', 'success')
    except Exception as e:
        logging.error(f"Error deleting session: {e}")
        flash('Error deleting chat session', 'error')
    
    return redirect(url_for('dashboard'))

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500
