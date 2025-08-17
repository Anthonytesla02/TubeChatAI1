# YouTube RAG Chatbot

## Overview

This is a Flask-based web application that allows users to upload YouTube videos, extract their transcripts, and have AI-powered conversations about the video content using Retrieval Augmented Generation (RAG). Users can ask questions about specific YouTube videos and receive contextual answers based on the actual transcript content.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Web Framework
- **Flask**: Core web framework handling HTTP requests, routing, and templating
- **Flask-Login**: User session management and authentication
- **Flask-SQLAlchemy**: Database ORM for data persistence
- **Werkzeug ProxyFix**: Production deployment support for reverse proxy setups

### Authentication System
- User registration and login with username/email support
- Password hashing using Werkzeug's security utilities
- Session-based authentication with remember-me functionality
- Email and password validation with security requirements

### Database Architecture
- **SQLAlchemy ORM** with declarative base model
- **User Model**: Stores user credentials and metadata
- **Transcript Model**: Stores YouTube video information and extracted transcripts
- **TranscriptChunk Model**: Stores chunked transcript segments with embeddings for RAG
- **ChatSession/ChatMessage Models**: Manages conversation history and context

### RAG (Retrieval Augmented Generation) System
- **Sentence Transformers**: Uses 'all-MiniLM-L6-v2' model for text embeddings
- **Text Chunking**: Splits transcripts into overlapping segments for better context preservation
- **Vector Similarity Search**: Finds relevant transcript chunks based on user queries
- **Binary Embedding Storage**: Embeddings stored as binary data in database for efficient retrieval

### YouTube Integration
- **yt-dlp**: Extracts video metadata and subtitles/transcripts from YouTube URLs
- **Multi-language Support**: Attempts to extract English subtitles in various formats
- **Video ID Extraction**: Supports multiple YouTube URL formats
- **Automatic Subtitle Processing**: Handles both manual and auto-generated captions

### AI Integration
- **Mistral AI API**: Powers the conversational AI responses
- **Context-aware Responses**: Incorporates relevant transcript chunks into prompts
- **RAG Pipeline**: Combines similarity search results with LLM generation
- **Conversation Memory**: Maintains chat session history for context

### Frontend Architecture
- **Bootstrap 5**: Responsive UI framework with modern design
- **Custom CSS**: Enhanced styling with CSS variables and animations
- **JavaScript**: Interactive chat interface with real-time messaging
- **Template Inheritance**: Jinja2 templating with base template system

### Security Features
- Environment-based configuration for sensitive data
- Password strength validation (minimum 8 characters, letters, numbers)
- Email format validation using regex patterns
- User input sanitization and validation
- Session security with configurable session secrets

## External Dependencies

### Core Dependencies
- **Flask Ecosystem**: Flask, Flask-SQLAlchemy, Flask-Login for web framework
- **yt-dlp**: YouTube video and subtitle extraction
- **sentence-transformers**: Text embedding generation for RAG
- **numpy**: Numerical operations for embedding processing

### AI Services
- **Mistral AI API**: Language model for generating contextual responses
- **Sentence Transformers Model**: all-MiniLM-L6-v2 for embedding generation

### Database
- **SQLAlchemy**: Database abstraction layer
- **Database URL**: Configurable via environment variable (supports PostgreSQL, SQLite, etc.)
- **Connection Pooling**: Configured with pool recycling and pre-ping for reliability

### Frontend Libraries
- **Bootstrap 5.3.0**: CSS framework from CDN
- **Font Awesome 6.4.0**: Icon library
- **Google Fonts**: Inter font family for typography

### Development Tools
- **Werkzeug**: WSGI utilities and development server
- **Logging**: Python logging for debugging and monitoring

### Environment Configuration
- **SESSION_SECRET**: Flask session encryption key
- **DATABASE_URL**: Database connection string
- **MISTRAL_API_KEY**: API key for Mistral AI service