#!/usr/bin/env python3
"""
MemoryManager Demonstration Script

This script demonstrates the key features of the MemoryManager class 
for the Sovereign AI Long-Term Memory (RAG) system.

Key Features Demonstrated:
- User and conversation management
- Message operations and context windows
- Document management and chunking
- Embedding operations and storage
- Search capabilities (keyword and semantic)
- System health monitoring
- Performance statistics
- Privacy controls and access logging
"""

import json
import time
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime

# Add the src directory to the path to import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sovereign.config import Config
from sovereign.memory_manager import (
    MemoryManager, MessageSender, ConversationStatus, SearchMode
)

def create_demo_config():
    """Create a demo configuration with a temporary database"""
    config = Config()
    
    # Create temporary database for demo
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    
    config.database.db_path = temp_db.name
    config.database.enable_rag = True
    
    return config

def demo_user_management(manager):
    """Demonstrate user management capabilities"""
    print("\\n" + "="*50)
    print("🧑 USER MANAGEMENT DEMO")
    print("="*50)
    
    # Create users
    admin_id = manager.create_user(
        username="admin",
        email="admin@example.com",
        privacy_level=1,
        settings={"role": "administrator"}
    )
    
    user_id = manager.create_user(
        username="demo_user",
        email="demo@example.com",
        privacy_level=2,
        settings={"role": "user"}
    )
    
    print(f"✅ Created admin user (ID: {admin_id})")
    print(f"✅ Created demo user (ID: {user_id})")
    
    # Set current user
    manager.set_current_user(user_id)
    current_user = manager.get_current_user()
    print(f"✅ Set current user to: {current_user}")
    
    return user_id, admin_id

def demo_conversation_management(manager):
    """Demonstrate conversation management capabilities"""
    print("\\n" + "="*50)
    print("💬 CONVERSATION MANAGEMENT DEMO")
    print("="*50)
    
    # Create conversations
    conv1_id = manager.start_conversation(
        title="Technical Discussion",
        privacy_level=1,
        context_window_size=5
    )
    
    conv2_id = manager.start_conversation(
        title="Project Planning",
        privacy_level=2,
        context_window_size=8
    )
    
    conv3_id = manager.start_conversation(
        title="Research Notes",
        privacy_level=1,
        context_window_size=10
    )
    
    print(f"✅ Created 3 conversations:")
    print(f"   - Technical Discussion (ID: {conv1_id})")
    print(f"   - Project Planning (ID: {conv2_id})")
    print(f"   - Research Notes (ID: {conv3_id})")
    
    # Set current conversation
    manager.set_current_conversation(conv1_id)
    
    # Close one conversation
    manager.close_conversation(conv3_id)
    
    # List conversations
    all_conversations = manager.list_conversations()
    active_conversations = manager.list_conversations(status=ConversationStatus.ACTIVE)
    
    print(f"✅ Total conversations: {len(all_conversations)}")
    print(f"✅ Active conversations: {len(active_conversations)}")
    
    return conv1_id, conv2_id, conv3_id

def demo_message_operations(manager, conversation_id):
    """Demonstrate message operations and context windows"""
    print("\\n" + "="*50)
    print("📝 MESSAGE OPERATIONS DEMO")
    print("="*50)
    
    # Add messages to conversation
    messages = [
        ("Hello, I have a question about machine learning algorithms.", MessageSender.USER),
        ("I'd be happy to help! What would you like to know about ML algorithms?", MessageSender.ASSISTANT),
        ("Can you explain the difference between supervised and unsupervised learning?", MessageSender.USER),
        ("Certainly! Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data.", MessageSender.ASSISTANT),
        ("What are some common supervised learning algorithms?", MessageSender.USER),
        ("Common supervised learning algorithms include linear regression, decision trees, random forests, and neural networks.", MessageSender.ASSISTANT),
        ("And what about unsupervised learning algorithms?", MessageSender.USER),
        ("Popular unsupervised algorithms include k-means clustering, hierarchical clustering, and principal component analysis (PCA).", MessageSender.ASSISTANT),
    ]
    
    message_ids = []
    for message_text, sender in messages:
        msg_id = manager.add_message(
            message=message_text,
            sender=sender,
            conversation_id=conversation_id,
            token_count=len(message_text.split()),
            response_time=0.5 if sender == MessageSender.ASSISTANT else None
        )
        message_ids.append(msg_id)
        time.sleep(0.1)  # Small delay to ensure different timestamps
    
    print(f"✅ Added {len(messages)} messages to conversation")
    
    # Get context window
    context = manager.get_context_window(conversation_id)
    print(f"✅ Context window size: {len(context)} messages")
    print(f"   Most recent message: '{context[0].message[:50]}...'")
    
    # Get all messages
    all_messages = manager.get_conversation_messages(conversation_id)
    print(f"✅ Total messages in conversation: {len(all_messages)}")
    
    return message_ids

def demo_document_management(manager):
    """Demonstrate document management and chunking"""
    print("\\n" + "="*50)
    print("📄 DOCUMENT MANAGEMENT DEMO")
    print("="*50)
    
    # Add documents
    documents = [
        {
            "title": "Machine Learning Overview",
            "content": """Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. 
            
            Supervised learning uses labeled training data to learn a mapping from inputs to outputs. Common supervised learning tasks include classification (predicting categories) and regression (predicting continuous values).
            
            Unsupervised learning finds patterns in data without labeled examples. Common unsupervised learning tasks include clustering (grouping similar data points) and dimensionality reduction (simplifying data while preserving important information).
            
            Reinforcement learning involves training agents to make decisions by rewarding good actions and penalizing bad ones. This is commonly used in game playing, robotics, and autonomous systems.""",
            "source": "AI Textbook",
            "document_type": "educational"
        },
        {
            "title": "Data Science Best Practices",
            "content": """Data science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data. 
            
            The data science process typically involves several key steps: data collection, data cleaning and preprocessing, exploratory data analysis, feature engineering, model selection and training, model evaluation, and deployment.
            
            Data quality is crucial for successful data science projects. Common data quality issues include missing values, outliers, inconsistent formats, and duplicate records. Proper data preprocessing techniques such as imputation, normalization, and feature scaling can help address these issues.
            
            Model validation is essential to ensure that machine learning models generalize well to new, unseen data. Common validation techniques include cross-validation, holdout validation, and bootstrap sampling.""",
            "source": "Data Science Handbook",
            "document_type": "reference"
        },
        {
            "title": "Python Programming Guide",
            "content": """Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in data science, machine learning, web development, and automation.
            
            Key Python features include dynamic typing, automatic memory management, extensive standard library, and a large ecosystem of third-party packages. Popular data science libraries include NumPy for numerical computing, Pandas for data manipulation, Matplotlib for visualization, and Scikit-learn for machine learning.
            
            Python's syntax is designed to be intuitive and easy to read. It uses indentation to define code blocks, which promotes clean and consistent code structure. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.""",
            "source": "Programming Tutorial",
            "document_type": "tutorial"
        }
    ]
    
    document_ids = []
    for doc in documents:
        doc_id = manager.add_document(
            title=doc["title"],
            content=doc["content"],
            source=doc["source"],
            document_type=doc["document_type"],
            privacy_level=1
        )
        document_ids.append(doc_id)
    
    print(f"✅ Added {len(documents)} documents to knowledge base")
    
    # Create chunks for all documents
    total_chunks = 0
    for doc_id in document_ids:
        chunk_ids = manager.create_chunks(doc_id, chunk_size=300, overlap_size=50)
        total_chunks += len(chunk_ids)
        print(f"✅ Created {len(chunk_ids)} chunks for document {doc_id}")
    
    print(f"✅ Total chunks created: {total_chunks}")
    
    # List documents
    all_documents = manager.list_documents()
    print(f"✅ Total documents in knowledge base: {len(all_documents)}")
    
    return document_ids

def demo_embedding_operations(manager, document_ids):
    """Demonstrate embedding operations and storage"""
    print("\\n" + "="*50)
    print("🔮 EMBEDDING OPERATIONS DEMO")
    print("="*50)
    
    # Generate embeddings for some sample texts
    sample_texts = [
        "What is machine learning?",
        "How does supervised learning work?",
        "Explain data preprocessing techniques",
        "Python programming best practices"
    ]
    
    embeddings_generated = 0
    for text in sample_texts:
        embedding = manager.generate_embedding(text)
        if embedding is not None:
            embeddings_generated += 1
    
    print(f"✅ Generated {embeddings_generated} embeddings for sample texts")
    
    # Get and store embeddings for document chunks
    embeddings_stored = 0
    for doc_id in document_ids:
        chunks = manager.get_document_chunks(doc_id)
        for chunk in chunks[:3]:  # Store embeddings for first 3 chunks per document
            embedding = manager.generate_embedding(chunk.text)
            if embedding is not None:
                success = manager.store_embedding(chunk.id, embedding)
                if success:
                    embeddings_stored += 1
    
    print(f"✅ Stored {embeddings_stored} embeddings for document chunks")
    
    # Test embedding retrieval
    retrieved_embeddings = 0
    for doc_id in document_ids:
        chunks = manager.get_document_chunks(doc_id)
        for chunk in chunks[:2]:  # Test retrieval for first 2 chunks per document
            embedding = manager.get_embedding(chunk.id)
            if embedding is not None:
                retrieved_embeddings += 1
    
    print(f"✅ Retrieved {retrieved_embeddings} embeddings from storage")
    
    return embeddings_stored

def demo_search_capabilities(manager):
    """Demonstrate search capabilities"""
    print("\\n" + "="*50)
    print("🔍 SEARCH CAPABILITIES DEMO")
    print("="*50)
    
    # Test keyword search
    search_queries = [
        "machine learning algorithms",
        "data preprocessing",
        "Python programming",
        "supervised learning"
    ]
    
    total_results = 0
    for query in search_queries:
        results = manager.search_keyword(query, k=3)
        total_results += len(results)
        if results:
            print(f"✅ Keyword search '{query}': {len(results)} results")
            print(f"   Top result: '{results[0].chunk_text[:60]}...'")
    
    print(f"✅ Total keyword search results: {total_results}")
    
    # Test semantic search (placeholder)
    semantic_results = manager.search_semantic("What is artificial intelligence?", k=5)
    print(f"✅ Semantic search: {len(semantic_results)} results (placeholder implementation)")
    
    return total_results

def demo_system_monitoring(manager):
    """Demonstrate system health monitoring and statistics"""
    print("\\n" + "="*50)
    print("📊 SYSTEM MONITORING DEMO")
    print("="*50)
    
    # Get memory statistics
    stats = manager.get_memory_stats()
    print(f"✅ Memory Statistics:")
    print(f"   - Total conversations: {stats.total_conversations}")
    print(f"   - Total messages: {stats.total_messages}")
    print(f"   - Total documents: {stats.total_documents}")
    print(f"   - Total chunks: {stats.total_chunks}")
    print(f"   - Total embeddings: {stats.total_embeddings}")
    print(f"   - Database size: {stats.db_size_mb:.2f} MB")
    
    # Health check
    health = manager.health_check()
    print(f"✅ Health Check:")
    print(f"   - Status: {health['status']}")
    print(f"   - Database connected: {health['database_connected']}")
    print(f"   - Schema valid: {health['schema_valid']}")
    print(f"   - Issues: {len(health['issues'])}")
    
    return stats, health

def demo_privacy_and_access_logging(manager):
    """Demonstrate privacy controls and access logging"""
    print("\\n" + "="*50)
    print("🔐 PRIVACY & ACCESS LOGGING DEMO")
    print("="*50)
    
    # Access logging is automatically done by the system
    # Let's check the access logs
    conn = manager.schema.get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM access_logs")
    total_logs = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT action, COUNT(*) as count
        FROM access_logs
        GROUP BY action
        ORDER BY count DESC
    """)
    
    action_counts = cursor.fetchall()
    
    print(f"✅ Access Logging:")
    print(f"   - Total access logs: {total_logs}")
    print(f"   - Actions breakdown:")
    for action, count in action_counts:
        print(f"     * {action}: {count}")
    
    return total_logs

def main():
    """Main demonstration function"""
    print("🚀 SOVEREIGN AI MEMORY MANAGER DEMONSTRATION")
    print("="*70)
    print("This demo showcases the complete Long-Term Memory (RAG) system")
    print("="*70)
    
    start_time = time.time()
    
    # Create configuration and memory manager
    config = create_demo_config()
    
    try:
        with MemoryManager(config) as manager:
            print(f"✅ MemoryManager initialized with database: {config.database.db_path}")
            
            # Run demonstrations
            user_id, admin_id = demo_user_management(manager)
            conv1_id, conv2_id, conv3_id = demo_conversation_management(manager)
            message_ids = demo_message_operations(manager, conv1_id)
            document_ids = demo_document_management(manager)
            embeddings_stored = demo_embedding_operations(manager, document_ids)
            search_results = demo_search_capabilities(manager)
            stats, health = demo_system_monitoring(manager)
            access_logs = demo_privacy_and_access_logging(manager)
            
            # Summary
            print("\\n" + "="*50)
            print("📋 DEMONSTRATION SUMMARY")
            print("="*50)
            print(f"✅ Users created: 2")
            print(f"✅ Conversations created: 3")
            print(f"✅ Messages added: {len(message_ids)}")
            print(f"✅ Documents added: {len(document_ids)}")
            print(f"✅ Chunks created: {stats.total_chunks}")
            print(f"✅ Embeddings stored: {embeddings_stored}")
            print(f"✅ Search results: {search_results}")
            print(f"✅ Access logs: {access_logs}")
            print(f"✅ System health: {health['status']}")
            
            execution_time = time.time() - start_time
            print(f"✅ Total execution time: {execution_time:.2f} seconds")
            
            print("\\n" + "="*50)
            print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
            print("="*50)
            print("The MemoryManager provides a comprehensive interface for:")
            print("- Long-term conversation storage and retrieval")
            print("- Document management and chunking")
            print("- Embedding operations and semantic search")
            print("- Privacy controls and access logging")
            print("- System health monitoring and statistics")
            print("- Production-ready performance and reliability")
            
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up temporary database
        try:
            Path(config.database.db_path).unlink()
            print(f"✅ Cleaned up temporary database: {config.database.db_path}")
        except:
            pass

if __name__ == "__main__":
    main() 