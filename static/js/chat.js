/**
 * Chat interface JavaScript for YouTube RAG Chatbot
 * Handles real-time chat functionality
 */

// Chat state
let chatState = {
    isLoading: false,
    sessionId: null,
    messageHistory: [],
    contextVisible: false
};

// Initialize chat when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeChat();
});

/**
 * Initialize chat interface
 */
function initializeChat() {
    // Get session ID from global variable
    chatState.sessionId = window.SESSION_ID;
    
    if (!chatState.sessionId) {
        console.error('No session ID found');
        showError('Session not found. Please refresh the page.');
        return;
    }
    
    // Initialize chat components
    initializeChatForm();
    initializeChatMessages();
    initializeKeyboardShortcuts();
    
    // Focus on input
    const messageInput = document.getElementById('messageInput');
    if (messageInput) {
        messageInput.focus();
    }
    
    console.log('Chat initialized for session:', chatState.sessionId);
}

/**
 * Initialize chat form handling
 */
function initializeChatForm() {
    const chatForm = document.getElementById('chatForm');
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    
    if (!chatForm || !messageInput || !sendButton) {
        console.error('Chat form elements not found');
        return;
    }
    
    // Handle form submission
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        handleSendMessage();
    });
    
    // Handle input changes
    messageInput.addEventListener('input', function() {
        updateSendButton();
    });
    
    // Handle Enter key (without Shift)
    messageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!chatState.isLoading && this.value.trim()) {
                handleSendMessage();
            }
        }
    });
    
    // Update send button state initially
    updateSendButton();
}

/**
 * Initialize chat messages area
 */
function initializeChatMessages() {
    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages) {
        // Scroll to bottom
        scrollToBottom();
        
        // Set up auto-scroll on new messages
        const observer = new MutationObserver(function() {
            scrollToBottom();
        });
        
        observer.observe(chatMessages, { 
            childList: true, 
            subtree: true 
        });
    }
}

/**
 * Initialize keyboard shortcuts
 */
function initializeKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Focus input on any typing (when not in an input)
        if (!e.ctrlKey && !e.altKey && !e.metaKey && 
            e.target.tagName !== 'INPUT' && 
            e.target.tagName !== 'TEXTAREA' &&
            e.key.length === 1) {
            
            const messageInput = document.getElementById('messageInput');
            if (messageInput) {
                messageInput.focus();
            }
        }
        
        // Escape to clear input
        if (e.key === 'Escape') {
            const messageInput = document.getElementById('messageInput');
            if (messageInput && document.activeElement === messageInput) {
                messageInput.blur();
            }
        }
    });
}

/**
 * Handle sending a message
 */
async function handleSendMessage() {
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();
    
    if (!message || chatState.isLoading) {
        return;
    }
    
    // Clear input immediately
    messageInput.value = '';
    updateSendButton();
    
    // Add user message to chat
    addMessageToChat('user', message);
    
    // Set loading state
    setLoadingState(true);
    
    try {
        // Send message to API
        const response = await sendMessageToAPI(message);
        
        if (response.success) {
            // Add AI response to chat
            addMessageToChat('assistant', response.response, {
                contextChunks: response.context_chunks,
                confidence: response.confidence_score
            });
            
            // Update context display
            updateContextDisplay(response.context_chunks);
        } else {
            throw new Error(response.error || 'Failed to get response');
        }
        
    } catch (error) {
        console.error('Error sending message:', error);
        addErrorMessage(error.message || 'Failed to send message. Please try again.');
    } finally {
        setLoadingState(false);
        messageInput.focus();
    }
}

/**
 * Send message to API
 */
async function sendMessageToAPI(message) {
    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest'
        },
        body: JSON.stringify({
            message: message,
            session_id: chatState.sessionId
        })
    });
    
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
    }
    
    return await response.json();
}

/**
 * Add message to chat interface
 */
function addMessageToChat(type, content, metadata = {}) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    
    // Remove welcome message if it exists
    const welcomeMessage = chatMessages.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }
    
    // Create message element
    const messageElement = document.createElement('div');
    messageElement.className = `message ${type} animate-fade-in`;
    
    const currentTime = new Date().toLocaleTimeString([], { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
    
    let confidenceBadge = '';
    if (type === 'assistant' && metadata.confidence) {
        confidenceBadge = `<span class="confidence-badge ms-2">${metadata.confidence}% confidence</span>`;
    }
    
    messageElement.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-${type === 'user' ? 'user' : 'robot'}"></i>
        </div>
        <div class="message-content">
            <div class="message-header">
                <strong>${type === 'user' ? 'You' : 'AI Assistant'}</strong>
                <small class="text-muted ms-2">${currentTime}</small>
                ${confidenceBadge}
            </div>
            <div class="message-text">${formatMessageContent(content)}</div>
        </div>
    `;
    
    chatMessages.appendChild(messageElement);
    
    // Store in message history
    chatState.messageHistory.push({
        type: type,
        content: content,
        timestamp: new Date(),
        metadata: metadata
    });
    
    // Animate in
    setTimeout(() => {
        messageElement.style.opacity = '1';
        messageElement.style.transform = 'translateY(0)';
    }, 10);
    
    scrollToBottom();
}

/**
 * Add error message to chat
 */
function addErrorMessage(errorText) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    
    const errorElement = document.createElement('div');
    errorElement.className = 'message error animate-fade-in';
    errorElement.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-exclamation-triangle text-danger"></i>
        </div>
        <div class="message-content">
            <div class="message-header">
                <strong class="text-danger">Error</strong>
                <small class="text-muted ms-2">${new Date().toLocaleTimeString([], { 
                    hour: '2-digit', 
                    minute: '2-digit' 
                })}</small>
            </div>
            <div class="message-text text-danger">${errorText}</div>
        </div>
    `;
    
    chatMessages.appendChild(errorElement);
    scrollToBottom();
}

/**
 * Format message content (handle line breaks, etc.)
 */
function formatMessageContent(content) {
    return content
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>');
}

/**
 * Update context display
 */
function updateContextDisplay(contextChunks) {
    const contextDisplay = document.getElementById('contextDisplay');
    const contextContent = document.getElementById('contextContent');
    
    if (!contextDisplay || !contextContent) return;
    
    if (contextChunks && contextChunks.length > 0) {
        contextContent.innerHTML = '';
        
        contextChunks.forEach((chunk, index) => {
            const chunkElement = document.createElement('div');
            chunkElement.className = 'context-chunk mb-2 p-2 bg-white rounded';
            chunkElement.innerHTML = `
                <div class="context-chunk-header mb-1">
                    <small class="text-muted fw-medium">
                        Context ${index + 1} (${chunk.similarity}% match)
                    </small>
                </div>
                <div class="context-chunk-text">
                    <small>${chunk.text}</small>
                </div>
            `;
            contextContent.appendChild(chunkElement);
        });
        
        contextDisplay.style.display = 'block';
        chatState.contextVisible = true;
    } else {
        contextDisplay.style.display = 'none';
        chatState.contextVisible = false;
    }
}

/**
 * Set loading state
 */
function setLoadingState(isLoading) {
    chatState.isLoading = isLoading;
    
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    
    if (messageInput) {
        messageInput.disabled = isLoading;
        messageInput.placeholder = isLoading ? 'AI is thinking...' : 'Ask a question about the video...';
    }
    
    if (sendButton) {
        sendButton.disabled = isLoading;
        if (isLoading) {
            sendButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        } else {
            sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
        }
    }
    
    // Show/hide typing indicator
    toggleTypingIndicator(isLoading);
}

/**
 * Toggle typing indicator
 */
function toggleTypingIndicator(show) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    
    const existingIndicator = chatMessages.querySelector('.typing-indicator');
    
    if (show && !existingIndicator) {
        const indicator = document.createElement('div');
        indicator.className = 'message assistant typing-indicator';
        indicator.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="message-text">
                    <div class="typing-animation">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            </div>
        `;
        chatMessages.appendChild(indicator);
        scrollToBottom();
    } else if (!show && existingIndicator) {
        existingIndicator.remove();
    }
}

/**
 * Update send button state
 */
function updateSendButton() {
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    
    if (messageInput && sendButton) {
        const hasText = messageInput.value.trim().length > 0;
        sendButton.disabled = !hasText || chatState.isLoading;
        
        if (!chatState.isLoading) {
            sendButton.className = hasText ? 'btn btn-primary' : 'btn btn-secondary';
        }
    }
}

/**
 * Scroll chat to bottom
 */
function scrollToBottom() {
    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages) {
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 100);
    }
}

/**
 * Show error notification
 */
function showError(message) {
    if (window.YouTubeRAG && window.YouTubeRAG.showNotification) {
        window.YouTubeRAG.showNotification(message, 'error');
    } else {
        alert('Error: ' + message);
    }
}

/**
 * Export chat history
 */
function exportChatHistory() {
    if (chatState.messageHistory.length === 0) {
        showError('No messages to export');
        return;
    }
    
    const transcript = chatState.messageHistory.map(msg => {
        const timestamp = msg.timestamp.toLocaleString();
        const sender = msg.type === 'user' ? 'You' : 'AI Assistant';
        return `[${timestamp}] ${sender}: ${msg.content}`;
    }).join('\n\n');
    
    const blob = new Blob([transcript], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-history-${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    URL.revokeObjectURL(url);
    
    if (window.YouTubeRAG && window.YouTubeRAG.showNotification) {
        window.YouTubeRAG.showNotification('Chat history exported successfully', 'success');
    }
}

/**
 * Clear current chat session (client-side only)
 */
function clearCurrentChat() {
    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages) {
        chatMessages.innerHTML = `
            <div class="welcome-message text-center">
                <div class="welcome-icon mb-3">
                    <i class="fas fa-comments"></i>
                </div>
                <h5>Chat cleared!</h5>
                <p class="text-muted">
                    Start a new conversation about the video.
                </p>
            </div>
        `;
    }
    
    chatState.messageHistory = [];
    updateContextDisplay([]);
    
    const messageInput = document.getElementById('messageInput');
    if (messageInput) {
        messageInput.focus();
    }
}

/**
 * Handle connection errors
 */
function handleConnectionError() {
    addErrorMessage('Connection error. Please check your internet connection and try again.');
}

/**
 * Retry failed message
 */
function retryLastMessage() {
    const lastMessage = chatState.messageHistory.slice(-1)[0];
    if (lastMessage && lastMessage.type === 'user') {
        handleSendMessage();
    }
}

// Global chat functions
window.ChatInterface = {
    sendMessage: handleSendMessage,
    exportHistory: exportChatHistory,
    clearChat: clearCurrentChat,
    retryMessage: retryLastMessage,
    scrollToBottom: scrollToBottom
};

// Handle page visibility changes
document.addEventListener('visibilitychange', function() {
    if (!document.hidden) {
        // Page became visible again, focus input
        const messageInput = document.getElementById('messageInput');
        if (messageInput && !chatState.isLoading) {
            setTimeout(() => messageInput.focus(), 100);
        }
    }
});

// Handle online/offline status
window.addEventListener('online', function() {
    if (window.YouTubeRAG && window.YouTubeRAG.showNotification) {
        window.YouTubeRAG.showNotification('Connection restored', 'success', 2000);
    }
});

window.addEventListener('offline', function() {
    if (window.YouTubeRAG && window.YouTubeRAG.showNotification) {
        window.YouTubeRAG.showNotification('Connection lost. Please check your internet.', 'warning');
    }
});
