#!/usr/bin/env python3
"""
Context Window Manager Demo - Showcase Intelligent Context Selection

This demo showcases the Context Window Manager's capabilities including:
- Multiple selection strategies
- Token-aware context management
- User preference customization
- Model-specific optimizations
- Performance monitoring
"""

import asyncio
import logging
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from src.sovereign.config import Config
from src.sovereign.memory_manager import MemoryManager, MessageSender
from src.sovereign.embedding_service import EmbeddingService, EmbeddingModelType
from src.sovereign.vector_search_engine import VectorSearchEngine
from src.sovereign.context_window_manager import (
    ContextWindowManager, SelectionStrategy, ModelContextLimits, UserPreferences
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def create_sample_conversation(memory_manager: MemoryManager) -> int:
    """Create a sample conversation with realistic messages"""
    logger.info("Creating sample conversation...")
    
    # Create user and start conversation
    user_id = memory_manager.create_user("demo_user", "demo@example.com")
    memory_manager.set_current_user(user_id)
    
    conv_id = memory_manager.start_conversation(
        title="AI and Machine Learning Discussion",
        context_window_size=10
    )
    memory_manager.set_current_conversation(conv_id)
    
    # Create realistic conversation flow
    messages = [
        ("user", "Hi! I'm interested in learning about artificial intelligence. Can you help?"),
        ("assistant", "Of course! I'd be happy to help you learn about AI. What specific aspects are you most curious about?"),
        ("user", "What's the difference between machine learning and deep learning?"),
        ("assistant", "Great question! Machine learning is a subset of AI that enables computers to learn from data without being explicitly programmed. Deep learning is a subset of machine learning that uses neural networks with multiple layers (hence 'deep') to model complex patterns in data."),
        ("user", "That's interesting. How do neural networks actually work?"),
        ("assistant", "Neural networks are inspired by how the brain works. They consist of interconnected nodes (neurons) organized in layers. Each connection has a weight, and the network learns by adjusting these weights based on training data. Information flows from input layer through hidden layers to output layer."),
        ("user", "What are some practical applications of deep learning?"),
        ("assistant", "Deep learning has many applications: image recognition (like photo tagging), natural language processing (chatbots, translation), speech recognition, autonomous vehicles, medical diagnosis, recommendation systems, and even creative tasks like generating art or music."),
        ("user", "How much data do you typically need to train a deep learning model?"),
        ("assistant", "The amount of data needed varies greatly depending on the problem complexity and model architecture. Simple tasks might need thousands of examples, while complex tasks like language models can require millions or billions of examples. Transfer learning can help reduce data requirements by starting with pre-trained models."),
        ("user", "What about overfitting? I've heard that's a problem in machine learning."),
        ("assistant", "Overfitting occurs when a model learns the training data too well, including noise and irrelevant patterns, making it perform poorly on new data. It's like memorizing answers instead of understanding concepts. Techniques to prevent overfitting include regularization, dropout, cross-validation, and using more training data."),
        ("user", "Can you explain what GPT models are and how they work?"),
        ("assistant", "GPT (Generative Pre-trained Transformer) models are large language models based on the transformer architecture. They're trained on vast amounts of text to predict the next word in a sequence. This simple task enables them to learn grammar, facts, reasoning, and even some world knowledge. They use attention mechanisms to focus on relevant parts of the input when generating responses."),
        ("user", "What's the difference between supervised and unsupervised learning?"),
        ("assistant", "In supervised learning, you train models on labeled data - input-output pairs where you know the correct answer. Examples include spam detection (email → spam/not spam) or image classification. Unsupervised learning finds patterns in data without labels, like clustering customers by behavior or reducing data dimensions."),
        ("user", "How do I get started with machine learning as a beginner?"),
        ("assistant", "Here's a beginner roadmap: 1) Learn Python programming basics, 2) Understand statistics and linear algebra fundamentals, 3) Study core ML concepts (supervised/unsupervised learning, evaluation metrics), 4) Practice with libraries like scikit-learn, 5) Work on projects with real datasets, 6) Take online courses (Coursera, edX), 7) Join ML communities and competitions like Kaggle."),
    ]
    
    # Add messages with realistic timing
    base_time = datetime.now() - timedelta(hours=2)
    
    for i, (sender, content) in enumerate(messages):
        message_time = base_time + timedelta(minutes=i * 3)  # 3 minutes between messages
        
        # Manually set the created_at time by modifying the message after creation
        msg_id = memory_manager.add_message(
            message=content,
            sender=MessageSender.USER if sender == "user" else MessageSender.ASSISTANT,
            conversation_id=conv_id
        )
        
        # Update timestamp in database directly for demo purposes
        conn = memory_manager.schema.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE messages SET created_at = ? WHERE id = ?",
            (message_time.isoformat(), msg_id)
        )
        conn.commit()
    
    logger.info(f"Created conversation {conv_id} with {len(messages)} messages")
    return conv_id


async def demo_selection_strategies(context_manager: ContextWindowManager, conv_id: int):
    """Demonstrate different context selection strategies"""
    logger.info("\n" + "="*60)
    logger.info("DEMO: Context Selection Strategies")
    logger.info("="*60)
    
    query = "How do neural networks learn and what prevents overfitting?"
    
    strategies = [
        SelectionStrategy.RECENCY_ONLY,
        SelectionStrategy.RELEVANCE_ONLY,
        SelectionStrategy.HYBRID,
        SelectionStrategy.ADAPTIVE
    ]
    
    for strategy in strategies:
        logger.info(f"\n--- {strategy.value.upper()} Strategy ---")
        
        context_window = await context_manager.build_context_window(
            query=query,
            conversation_id=conv_id,
            model_name="gpt-4",
            strategy=strategy,
            max_messages=30
        )
        
        selected_items = context_window.get_selected_items()
        
        logger.info(f"Selected {len(selected_items)} messages ({context_window.total_tokens} tokens)")
        logger.info(f"Available tokens remaining: {context_window.available_tokens}")
        
        # Show top 3 selected messages with scores
        for i, item in enumerate(selected_items[:3]):
            timestamp = item.message.created_at.strftime("%H:%M")
            sender = item.message.sender.value.upper()
            content = item.message.message[:100] + "..." if len(item.message.message) > 100 else item.message.message
            
            logger.info(f"  [{i+1}] [{timestamp}] {sender}: {content}")
            logger.info(f"      Relevance: {item.relevance_score:.3f}, Recency: {item.recency_score:.3f}, "
                       f"Combined: {item.combined_score:.3f}")


async def demo_user_preferences(context_manager: ContextWindowManager, conv_id: int):
    """Demonstrate user preferences customization"""
    logger.info("\n" + "="*60)
    logger.info("DEMO: User Preferences Impact")
    logger.info("="*60)
    
    query = "Explain the difference between supervised and unsupervised learning"
    
    # Test different preference configurations
    preference_configs = [
        ("Relevance-Focused", UserPreferences(relevance_weight=0.9, recency_weight=0.1, min_relevance_score=0.2)),
        ("Recency-Focused", UserPreferences(relevance_weight=0.1, recency_weight=0.9, min_relevance_score=0.1)),
        ("Balanced", UserPreferences(relevance_weight=0.5, recency_weight=0.5, min_relevance_score=0.3)),
        ("Quality-Focused", UserPreferences(relevance_weight=0.7, recency_weight=0.3, min_relevance_score=0.5, max_age_hours=48))
    ]
    
    for name, preferences in preference_configs:
        logger.info(f"\n--- {name} Preferences ---")
        logger.info(f"Relevance weight: {preferences.relevance_weight}, Recency weight: {preferences.recency_weight}")
        logger.info(f"Min relevance score: {preferences.min_relevance_score}, Max age: {preferences.max_age_hours}h")
        
        # Set user preferences
        context_manager.set_user_preferences(1, preferences)
        
        context_window = await context_manager.build_context_window(
            query=query,
            conversation_id=conv_id,
            model_name="gpt-4",
            strategy=SelectionStrategy.HYBRID,
            user_id=1
        )
        
        selected_items = context_window.get_selected_items()
        
        logger.info(f"Selected {len(selected_items)} messages ({context_window.total_tokens} tokens)")
        
        # Show selection distribution
        if selected_items:
            avg_relevance = sum(item.relevance_score for item in selected_items) / len(selected_items)
            avg_recency = sum(item.recency_score for item in selected_items) / len(selected_items)
            logger.info(f"Average relevance: {avg_relevance:.3f}, Average recency: {avg_recency:.3f}")


async def demo_model_limits(context_manager: ContextWindowManager, conv_id: int):
    """Demonstrate model-specific context limits"""
    logger.info("\n" + "="*60)
    logger.info("DEMO: Model-Specific Context Limits")
    logger.info("="*60)
    
    query = "Give me a comprehensive overview of machine learning and AI"
    
    # Test different model configurations
    models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "claude-3-sonnet"]
    
    for model_name in models:
        logger.info(f"\n--- {model_name} Model ---")
        
        limits = context_manager.model_limits.get(model_name)
        if limits:
            logger.info(f"Model limits: {limits.max_input_tokens} input tokens, "
                       f"{limits.max_output_tokens} output tokens")
        
        context_window = await context_manager.build_context_window(
            query=query,
            conversation_id=conv_id,
            model_name=model_name,
            strategy=SelectionStrategy.HYBRID
        )
        
        selected_items = context_window.get_selected_items()
        
        logger.info(f"Selected {len(selected_items)} messages using {context_window.total_tokens} tokens")
        logger.info(f"Remaining capacity: {context_window.available_tokens} tokens")
        
        # Show efficiency
        if limits:
            efficiency = (context_window.total_tokens / limits.max_input_tokens) * 100
            logger.info(f"Context utilization: {efficiency:.1f}%")


async def demo_context_formatting(context_manager: ContextWindowManager, conv_id: int):
    """Demonstrate context text formatting"""
    logger.info("\n" + "="*60)
    logger.info("DEMO: Context Text Formatting")
    logger.info("="*60)
    
    query = "What are some practical applications of machine learning?"
    
    context_window = await context_manager.build_context_window(
        query=query,
        conversation_id=conv_id,
        model_name="gpt-4",
        strategy=SelectionStrategy.HYBRID,
        max_messages=10
    )
    
    # Get formatted context text
    context_text = context_window.get_context_text()
    
    logger.info("Formatted context for AI model:")
    logger.info("-" * 40)
    logger.info(context_text)
    logger.info("-" * 40)
    
    # Show context window metadata
    metadata = context_window.metadata
    logger.info(f"\nContext Metadata:")
    logger.info(f"  Processing time: {metadata.get('processing_time', 0):.3f}s")
    logger.info(f"  Total candidates: {metadata.get('total_candidates', 0)}")
    logger.info(f"  Selected count: {metadata.get('selected_count', 0)}")
    logger.info(f"  Model used: {metadata.get('model_name', 'unknown')}")


async def demo_performance_monitoring(context_manager: ContextWindowManager, conv_id: int):
    """Demonstrate performance monitoring and statistics"""
    logger.info("\n" + "="*60)
    logger.info("DEMO: Performance Monitoring")
    logger.info("="*60)
    
    # Perform multiple context window operations
    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "What prevents overfitting?",
        "Applications of deep learning?",
        "Supervised vs unsupervised learning?"
    ]
    
    logger.info("Performing multiple context window builds...")
    
    for i, query in enumerate(queries):
        await context_manager.build_context_window(
            query=query,
            conversation_id=conv_id,
            model_name="gpt-4",
            strategy=SelectionStrategy.HYBRID
        )
        logger.info(f"Completed query {i+1}/{len(queries)}")
    
    # Show performance statistics
    stats = context_manager.get_statistics()
    
    logger.info("\nPerformance Statistics:")
    logger.info(f"  Total requests: {stats['total_requests']}")
    logger.info(f"  Total processing time: {stats['total_processing_time']:.3f}s")
    logger.info(f"  Average processing time: {stats['total_processing_time']/max(1, stats['total_requests']):.3f}s")
    logger.info(f"  Average context size: {stats['average_context_size']:.1f} items")
    logger.info(f"  Cache hits: {stats['cache_hits']}")
    logger.info(f"  Cache size: {stats['cache_size']}")
    
    logger.info("\nStrategy usage counts:")
    for strategy, count in stats['selection_strategy_counts'].items():
        logger.info(f"  {strategy}: {count}")


async def main():
    """Main demo function"""
    logger.info("Starting Context Window Manager Demo")
    logger.info("="*60)
    
    # Create temporary directory for demo
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "demo_context.db"
    
    try:
        # Create configuration
        config = Config()
        config.database.db_path = str(db_path)
        
        # Initialize components
        logger.info("Initializing components...")
        
        # Create memory manager
        memory_manager = MemoryManager(config)
        
        # Create embedding service (with mock for demo)
        embedding_service = EmbeddingService(
            default_model=EmbeddingModelType.BGE_BASE_EN_V15,
            enable_gpu=False
        )
        
        # Create vector search engine
        vector_search_engine = VectorSearchEngine(
            db_path=str(db_path),
            embedding_service=embedding_service
        )
        await vector_search_engine.initialize()
        
        # Create context window manager
        context_manager = ContextWindowManager(
            memory_manager=memory_manager,
            vector_search_engine=vector_search_engine,
            config=config
        )
        
        # Add custom model limits for demo
        custom_model = ModelContextLimits(
            model_name="demo-model",
            total_tokens=2048,
            max_input_tokens=1500,
            max_output_tokens=500,
            encoding_name="cl100k_base"
        )
        context_manager.add_model_limits("demo-model", custom_model)
        
        # Create sample conversation
        conv_id = await create_sample_conversation(memory_manager)
        
        # Run demos
        await demo_selection_strategies(context_manager, conv_id)
        await demo_user_preferences(context_manager, conv_id)
        await demo_model_limits(context_manager, conv_id)
        await demo_context_formatting(context_manager, conv_id)
        await demo_performance_monitoring(context_manager, conv_id)
        
        logger.info("\n" + "="*60)
        logger.info("DEMO COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        # Show final statistics
        final_stats = context_manager.get_statistics()
        logger.info(f"Final statistics: {final_stats['total_requests']} requests processed "
                   f"in {final_stats['total_processing_time']:.3f}s")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        raise
    
    finally:
        # Cleanup
        logger.info("Cleaning up resources...")
        try:
            memory_manager.close()
            await vector_search_engine.cleanup()
            context_manager.close()
        except:
            pass
        
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        
        logger.info("Demo cleanup completed")


if __name__ == "__main__":
    asyncio.run(main()) 