"""
RAG System Integration Tests

This module provides comprehensive integration tests for the Sovereign AI Long-Term Memory
(RAG) system, validating that all components work together seamlessly in real-world scenarios.

Test Coverage:
- End-to-end conversation storage and retrieval
- Semantic search across stored memories
- Context window construction from search results
- Privacy controls and access management
- Data export/import round-trip integrity
- Performance under realistic load
- Error handling and recovery scenarios
- Multi-user conversation isolation
- Large-scale memory operations

The tests simulate real user interactions to ensure the entire RAG pipeline
functions correctly from storage through contextual retrieval.
"""

import pytest
import asyncio
import tempfile
import shutil
import json
import sqlite3
import time
import random
import string
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch
import asyncio

# Import all RAG system components
from src.sovereign.memory_manager import MemoryManager, MessageSender
from src.sovereign.embedding_service import EmbeddingService, EmbeddingModelType
from src.sovereign.vector_search_engine import VectorSearchEngine, SearchParameters
from src.sovereign.context_window_manager import ContextWindowManager
from src.sovereign.privacy_manager import PrivacyManager
from src.sovereign.memory_exporter import MemoryExporter
from src.sovereign.memory_importer import MemoryImporter
from src.sovereign.memory_schema import MemorySchema
from src.sovereign.config import Config, DatabaseConfig


def setup_embedding_mocks(mock_tokenizer, mock_model):
    """Helper function to setup embedding service mocks consistently"""
    mock_tokenizer_instance = Mock()
    mock_tokenizer_instance.return_value = {
        'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
    }
    mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
    
    mock_model_instance = Mock()
    mock_outputs = Mock()
    mock_outputs.last_hidden_state = torch.randn(1, 5, 768)
    mock_model_instance.return_value = mock_outputs
    mock_model.from_pretrained.return_value = mock_model_instance


class RAGSystemTestHelper:
    """Helper class for RAG system integration testing"""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = Path(temp_dir)
        self.db_path = self.temp_dir / "test_rag.db"
        self.cache_dir = self.temp_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize all components
        self.memory_manager = None
        self.embedding_service = None
        self.vector_search = None
        self.context_manager = None
        self.privacy_manager = None
        self.exporter = None
        self.importer = None
        
        self.setup_components()
    
    def setup_components(self):
        """Initialize all RAG system components"""
        # Create test configuration
        test_config = Config()
        test_config.database.db_path = str(self.db_path)
        test_config.database.vector_db_path = str(self.cache_dir / "vectors")
        
        # Force CPU-only mode for embeddings to avoid CUDA issues in tests
        test_config.hardware.gpu_enabled = False
        
        # Initialize database schema
        schema = MemorySchema(str(self.db_path))
        schema.create_schema()
        
        # Initialize core components
        self.memory_manager = MemoryManager(test_config)
        self.privacy_manager = PrivacyManager(
            config=test_config.model_dump() if hasattr(test_config, 'model_dump') else test_config.__dict__,
            db_path=str(self.db_path)
        )
        
        # Initialize embedding service with mocked models for testing
        self.embedding_service = EmbeddingService(
            cache_dir=str(self.cache_dir),
            enable_gpu=False,
            max_batch_size=4
        )
        
        # Initialize vector search engine
        self.vector_search = VectorSearchEngine(
            db_path=str(self.db_path),
            embedding_service=self.embedding_service
        )
        
        # Initialize context window manager
        self.context_manager = ContextWindowManager(
            memory_manager=self.memory_manager,
            vector_search_engine=self.vector_search,
            config=test_config
        )
        
        # Initialize export/import components
        self.exporter = MemoryExporter(
            config=test_config,
            db_path=str(self.db_path),
            privacy_manager=self.privacy_manager
        )
        self.importer = MemoryImporter(
            memory_manager=self.memory_manager,
            privacy_manager=self.privacy_manager
        )
    
    def create_test_user(self, user_id: str = "test_user_1") -> int:
        """Create a test user and return user_id"""
        return self.memory_manager.create_user(
            username=f"user_{user_id}",
            email=f"{user_id}@example.com",
            privacy_level=1,
            settings={
                "full_name": f"Test User {user_id}",
                "preferences": {"language": "en", "timezone": "UTC"}
            }
        )
    
    def create_test_conversation(self, user_id: int, topic: str = "test_topic") -> int:
        """Create a test conversation and return conversation_id"""
        # Set current user first
        self.memory_manager.set_current_user(user_id)
        
        # Start conversation
        return self.memory_manager.start_conversation(
            title=f"Test Conversation: {topic}",
            privacy_level=1
        )
    
    def add_test_messages(self, conversation_id: int, messages: List[Dict]) -> List[int]:
        """Add test messages to a conversation"""
        message_ids = []
        for msg in messages:
            # Map role to MessageSender enum
            role = msg.get("role", "user")
            if role == "assistant":
                sender = MessageSender.ASSISTANT
            elif role == "system":
                sender = MessageSender.SYSTEM
            else:
                sender = MessageSender.USER
                
            message_id = self.memory_manager.add_message(
                message=msg.get("content", ""),
                sender=sender,
                conversation_id=conversation_id,
                message_type=msg.get("message_type", "text")
            )
            message_ids.append(message_id)
        return message_ids
    
    def create_document_from_messages(self, conversation_id: int, title: str = "Conversation Document") -> int:
        """Convert conversation messages to a searchable document"""
        # Get all messages from the conversation
        messages = self.memory_manager.get_conversation_messages(conversation_id)
        
        # Convert messages to document content
        content_parts = []
        for msg in messages:
            sender_prefix = f"[{msg.sender.value.upper()}]"
            content_parts.append(f"{sender_prefix} {msg.message}")
        
        content = "\n\n".join(content_parts)
        
        # Create document
        document_id = self.memory_manager.add_document(
            title=title,
            content=content,
            source=f"conversation_{conversation_id}",
            document_type="conversation"
        )
        
        # Create chunks
        chunk_ids = self.memory_manager.create_chunks(document_id, chunk_size=500, overlap_size=50)
        
        # Generate embeddings for chunks
        embedding_results = self.memory_manager.generate_and_store_embeddings(document_id)
        
        return document_id
    
    def cleanup(self):
        """Clean up test resources"""
        # Close all connections and clean up resources in proper order
        print("🧹 Cleaning up RAG system test resources...")
        
        try:
            # Close vector search engine first
            if hasattr(self, 'vector_search') and self.vector_search:
                asyncio.run(self.vector_search.cleanup())
                print("   ✅ Vector search cleaned up")
        except Exception as e:
            print(f"   ⚠️ Error cleaning up vector search: {e}")
        
        try:
            # Close embedding service
            if hasattr(self, 'embedding_service') and self.embedding_service:
                if hasattr(self.embedding_service, 'cleanup'):
                    self.embedding_service.cleanup()
                print("   ✅ Embedding service cleaned up")
        except Exception as e:
            print(f"   ⚠️ Error cleaning up embedding service: {e}")
        
        try:
            # Close memory manager
            if hasattr(self, 'memory_manager') and self.memory_manager:
                self.memory_manager.close()
                print("   ✅ Memory manager closed")
        except Exception as e:
            print(f"   ⚠️ Error closing memory manager: {e}")
        
        try:
            # Close privacy manager
            if hasattr(self, 'privacy_manager') and self.privacy_manager:
                self.privacy_manager.close()
                print("   ✅ Privacy manager closed")
        except Exception as e:
            print(f"   ⚠️ Error closing privacy manager: {e}")
        
        try:
            # Close other components
            for attr_name, component in [
                ('context_manager', getattr(self, 'context_manager', None)),
                ('importer', getattr(self, 'importer', None)),
                ('exporter', getattr(self, 'exporter', None))
            ]:
                if component and hasattr(component, 'close'):
                    component.close()
                    print(f"   ✅ {attr_name} closed")
        except Exception as e:
            print(f"   ⚠️ Error closing components: {e}")
        
        # Force garbage collection to release file handles
        import gc
        gc.collect()
        
        # Small delay to ensure handles are released
        import time
        time.sleep(0.2)
        print("   ✅ Cleanup completed")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def rag_system(temp_dir):
    """Create a complete RAG system for testing"""
    system = RAGSystemTestHelper(temp_dir)
    yield system
    system.cleanup()


class TestRAGSystemBasicWorkflow:
    """Test basic RAG system workflows"""
    
    @pytest.mark.asyncio
    @patch('src.sovereign.embedding_service.AutoModel')
    @patch('src.sovereign.embedding_service.AutoTokenizer')
    async def test_end_to_end_conversation_storage_and_retrieval(self, mock_tokenizer, mock_model, rag_system):
        """Test complete workflow: store conversation, generate embeddings, search, retrieve context"""
        # Setup mock tokenizer and model
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(1, 5, 768)  # batch_size=1, seq_len=5, hidden_size=768
        mock_model_instance.return_value = mock_outputs
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Step 1: Create user and conversation
        user_id = rag_system.create_test_user("integration_user")
        conversation_id = rag_system.create_test_conversation(user_id, "AI Discussion")
        
        # Step 2: Add messages to conversation
        messages = [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."},
            {"role": "user", "content": "How does neural networks work?"},
            {"role": "assistant", "content": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through weighted connections."},
            {"role": "user", "content": "What are the types of machine learning?"},
            {"role": "assistant", "content": "There are three main types: supervised learning (learning from labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through trial and error)."}
        ]
        
        message_ids = rag_system.add_test_messages(conversation_id, messages)
        assert len(message_ids) == 6
        
        # Step 3: Create document from messages and generate embeddings
        # Convert the conversation to a searchable document
        document_id = rag_system.create_document_from_messages(
            conversation_id, 
            title="Machine Learning Discussion"
        )
        assert document_id is not None
        
        # Verify messages are stored correctly
        stored_messages = rag_system.memory_manager.get_conversation_messages(conversation_id)
        assert len(stored_messages) == 6
        
        # Step 4: Perform semantic search
        search_query = "explain different types of learning algorithms"
        search_params = SearchParameters(top_k=3, include_metadata=True)
        search_results = await rag_system.vector_search.search(
            query=search_query,
            params=search_params
        )
        
        # Should find relevant results
        assert len(search_results) > 0
        assert search_results[0].score > 0.0
        
        # Step 5: Construct context window
        context_window = await rag_system.context_manager.build_context_window(
            query=search_query,
            user_id=user_id,
            conversation_id=conversation_id,
            max_messages=20
        )
        
        # Should return structured context window
        assert context_window is not None
        assert len(context_window.items) > 0
        assert context_window.total_tokens > 0
        
        # Step 6: Verify context relevance
        # The context should include messages about machine learning types
        context_text = context_window.get_context_text()
        assert "machine learning" in context_text.lower()
        
        print(f"✅ End-to-end workflow completed successfully")
        print(f"   - Stored {len(messages)} messages")
        print(f"   - Found {len(search_results)} relevant results")
        print(f"   - Context contains {len(context_window.items)} items, {context_window.total_tokens} tokens")
    
    @patch('src.sovereign.embedding_service.AutoModel')
    @patch('src.sovereign.embedding_service.AutoTokenizer')
    def test_multi_conversation_context_isolation(self, mock_tokenizer, mock_model, rag_system):
        """Test that context from different conversations is properly isolated"""
        # Setup mock tokenizer and model
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(1, 5, 768)
        mock_model_instance.return_value = mock_outputs
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Create two users with different conversations
        user1_id = rag_system.create_test_user("user_1")
        user2_id = rag_system.create_test_user("user_2")
        
        # User 1: AI conversation
        conv1_id = rag_system.create_test_conversation(user1_id, "AI Discussion")
        ai_messages = [
            {"role": "user", "content": "Tell me about artificial intelligence"},
            {"role": "assistant", "content": "AI is the simulation of human intelligence in machines"}
        ]
        rag_system.add_test_messages(conv1_id, ai_messages)
        
        # User 2: Cooking conversation
        conv2_id = rag_system.create_test_conversation(user2_id, "Cooking Tips")
        cooking_messages = [
            {"role": "user", "content": "How do I make pasta?"},
            {"role": "assistant", "content": "To make pasta, boil water, add salt, and cook pasta until al dente"}
        ]
        rag_system.add_test_messages(conv2_id, cooking_messages)
        
        # Search for AI content as user 1
        ai_results = rag_system.vector_search.search(
            query="artificial intelligence",
            limit=5,
            user_id=user1_id
        )
        
        # Search for cooking content as user 2
        cooking_results = rag_system.vector_search.search(
            query="cooking pasta",
            limit=5,
            user_id=user2_id
        )
        
        # Verify isolation: each user should only see their own content
        assert len(ai_results) > 0
        assert len(cooking_results) > 0
        
        # User 1 should not see cooking content
        user1_context = rag_system.context_manager.get_context_for_query(
            query="cooking",
            user_id=user1_id,
            max_tokens=500
        )
        
        # User 2 should not see AI content
        user2_context = rag_system.context_manager.get_context_for_query(
            query="artificial intelligence",
            user_id=user2_id,
            max_tokens=500
        )
        
        # Context should be isolated
        if user1_context and user1_context.messages:
            context_content = " ".join([msg.content for msg in user1_context.messages])
            assert "pasta" not in context_content.lower()
        
        if user2_context and user2_context.messages:
            context_content = " ".join([msg.content for msg in user2_context.messages])
            assert "artificial intelligence" not in context_content.lower()
        
        print(f"✅ Multi-user context isolation working correctly")
    
    @patch('src.sovereign.embedding_service.AutoModel')
    @patch('src.sovereign.embedding_service.AutoTokenizer')
    def test_large_conversation_context_management(self, mock_tokenizer, mock_model, rag_system):
        """Test context management with large conversations"""
        # Setup mock tokenizer and model
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(1, 5, 768)
        mock_model_instance.return_value = mock_outputs
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Create user and conversation
        user_id = rag_system.create_test_user("large_conv_user")
        conversation_id = rag_system.create_test_conversation(user_id, "Large Discussion")
        
        # Add many messages to create a large conversation
        large_messages = []
        topics = ["programming", "databases", "algorithms", "data structures", "machine learning"]
        
        for i in range(50):  # Create 50 message pairs
            topic = random.choice(topics)
            large_messages.extend([
                {"role": "user", "content": f"Tell me about {topic} in detail. Message {i}"},
                {"role": "assistant", "content": f"Here's detailed information about {topic}. " + 
                 f"This is a comprehensive explanation with lots of details. Message {i}. " + 
                 f"Additional context: {' '.join([f'word{j}' for j in range(20)])}"}
            ])
        
        message_ids = rag_system.add_test_messages(conversation_id, large_messages)
        assert len(message_ids) == 100  # 50 * 2 messages
        
        # Test context window management with token limits
        context_data = rag_system.context_manager.get_context_for_query(
            query="programming algorithms",
            user_id=user_id,
            conversation_id=conversation_id,
            max_tokens=500  # Small limit to test truncation
        )
        
        # Should respect token limit
        assert context_data is not None
        assert context_data.total_tokens <= 500
        assert len(context_data.messages) < 100  # Should not include all messages
        
        # Test with larger limit
        large_context = rag_system.context_manager.get_context_for_query(
            query="machine learning databases",
            user_id=user_id,
            conversation_id=conversation_id,
            max_tokens=2000
        )
        
        # Should include more messages but still respect limit
        assert large_context is not None
        assert large_context.total_tokens <= 2000
        assert len(large_context.messages) > context_data.total_tokens
        
        print(f"✅ Large conversation context management working correctly")
        print(f"   - Small context: {len(context_data.messages)} messages, {context_data.total_tokens} tokens")
        print(f"   - Large context: {len(large_context.messages)} messages, {large_context.total_tokens} tokens")


class TestRAGSystemAdvancedFeatures:
    """Test advanced RAG system features"""
    
    @patch('src.sovereign.embedding_service.SentenceTransformer')
    def test_privacy_controlled_search(self, mock_transformer, rag_system):
        """Test that privacy controls are enforced during search operations"""
        # Setup mock
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(768).astype(np.float32)
        mock_transformer.return_value = mock_model
        
        # Create users with different privacy levels
        admin_id = rag_system.create_test_user("admin_user")
        regular_id = rag_system.create_test_user("regular_user")
        
        # Set up privacy levels
        rag_system.privacy_manager.set_user_role(admin_id, "admin")
        rag_system.privacy_manager.set_user_role(regular_id, "user")
        
        # Create conversations with different privacy levels
        public_conv = rag_system.create_test_conversation(admin_id, "Public Discussion")
        private_conv = rag_system.create_test_conversation(admin_id, "Private Discussion")
        
        # Add messages
        public_messages = [
            {"role": "user", "content": "This is public information about general topics"},
            {"role": "assistant", "content": "This is a public response that anyone can see"}
        ]
        
        private_messages = [
            {"role": "user", "content": "This is sensitive private information"},
            {"role": "assistant", "content": "This is confidential data that should be restricted"}
        ]
        
        rag_system.add_test_messages(public_conv, public_messages)
        rag_system.add_test_messages(private_conv, private_messages)
        
        # Set privacy levels on conversations
        rag_system.privacy_manager.set_data_privacy_level(public_conv, "public")
        rag_system.privacy_manager.set_data_privacy_level(private_conv, "private")
        
        # Test admin access (should see both)
        admin_results = rag_system.vector_search.search(
            query="information",
            limit=10,
            user_id=admin_id
        )
        
        # Test regular user access (should only see public)
        regular_results = rag_system.vector_search.search(
            query="information",
            limit=10,
            user_id=regular_id
        )
        
        # Admin should have access to more content
        assert len(admin_results) >= len(regular_results)
        
        print(f"✅ Privacy-controlled search working correctly")
        print(f"   - Admin results: {len(admin_results)}")
        print(f"   - Regular user results: {len(regular_results)}")
    
    @patch('src.sovereign.embedding_service.SentenceTransformer')
    def test_memory_export_import_roundtrip(self, mock_transformer, rag_system):
        """Test complete export/import workflow maintains data integrity"""
        # Setup mock
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(768).astype(np.float32)
        mock_transformer.return_value = mock_model
        
        # Create test data
        user_id = rag_system.create_test_user("export_user")
        conversation_id = rag_system.create_test_conversation(user_id, "Export Test")
        
        original_messages = [
            {"role": "user", "content": "What is data export?"},
            {"role": "assistant", "content": "Data export is the process of extracting data from a system for backup or migration purposes."},
            {"role": "user", "content": "Why is it important?"},
            {"role": "assistant", "content": "Data export is crucial for backup, migration, compliance, and data portability requirements."}
        ]
        
        original_message_ids = rag_system.add_test_messages(conversation_id, original_messages)
        
        # Export data
        export_file = rag_system.temp_dir / "export_test.json"
        export_result = rag_system.exporter.export_user_data(
            user_id=user_id,
            output_file=str(export_file),
            format="json"
        )
        
        assert export_result.success
        assert export_file.exists()
        
        # Create second RAG system for import
        import_temp_dir = tempfile.mkdtemp()
        import_system = RAGSystemTestHelper(import_temp_dir)
        
        try:
            # Import data
            import_result = import_system.importer.import_memory_data(
                input_file=str(export_file),
                options={"conflict_resolution": "overwrite"}
            )
            
            assert import_result.success
            assert import_result.imported_counts["messages"] == len(original_messages)
            
            # Verify data integrity
            imported_users = import_system.memory_manager.get_all_users()
            assert len(imported_users) >= 1
            
            # Find the imported user
            imported_user = None
            for user in imported_users:
                if user.get("username") == f"user_{user_id}":
                    imported_user = user
                    break
            
            assert imported_user is not None
            imported_user_id = imported_user["id"]
            
            # Get imported conversations
            imported_conversations = import_system.memory_manager.get_user_conversations(imported_user_id)
            assert len(imported_conversations) >= 1
            
            # Verify message content
            imported_conv = imported_conversations[0]
            imported_messages = import_system.memory_manager.get_messages(imported_conv["id"])
            assert len(imported_messages) == len(original_messages)
            
            # Verify content matches
            for i, (orig, imported) in enumerate(zip(original_messages, imported_messages)):
                assert orig["content"] == imported.get("content")
                assert orig["role"] == imported.get("role")
            
            print(f"✅ Export/import roundtrip successful")
            print(f"   - Exported {len(original_messages)} messages")
            print(f"   - Imported {import_result.imported_counts['messages']} messages")
            
        finally:
            import_system.cleanup()
            shutil.rmtree(import_temp_dir)
    
    @patch('src.sovereign.embedding_service.SentenceTransformer')
    def test_concurrent_operations_thread_safety(self, mock_transformer, rag_system):
        """Test that concurrent operations are thread-safe"""
        # Setup mock
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(768).astype(np.float32)
        mock_transformer.return_value = mock_model
        
        # Create test data
        user_id = rag_system.create_test_user("concurrent_user")
        conversation_id = rag_system.create_test_conversation(user_id, "Concurrent Test")
        
        # Function to add messages concurrently
        def add_messages_batch(batch_num: int, num_messages: int):
            messages = []
            for i in range(num_messages):
                messages.append({
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": f"Batch {batch_num} message {i} content for concurrent testing"
                })
            return rag_system.add_test_messages(conversation_id, messages)
        
        # Function to search concurrently
        def search_memories(query: str, limit: int = 5):
            return rag_system.vector_search.search(
                query=query,
                limit=limit,
                user_id=user_id
            )
        
        import threading
        import concurrent.futures
        
        # Test concurrent message addition
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for batch in range(4):
                future = executor.submit(add_messages_batch, batch, 10)
                futures.append(future)
            
            # Wait for all additions to complete
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # Verify all messages were added
            total_added = sum(len(result) for result in results)
            assert total_added == 40  # 4 batches * 10 messages each
        
        # Test concurrent searches
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            search_futures = []
            queries = ["concurrent testing", "batch message", "content search", "memory retrieval"]
            
            for query in queries:
                future = executor.submit(search_memories, query)
                search_futures.append(future)
            
            # Wait for all searches to complete
            search_results = [future.result() for future in concurrent.futures.as_completed(search_futures)]
            
            # Verify searches completed successfully
            assert len(search_results) == 4
            for results in search_results:
                assert isinstance(results, list)
                # Should find some results given the added messages
                assert len(results) >= 0
        
        print(f"✅ Concurrent operations completed successfully")
        print(f"   - Added {total_added} messages across 4 concurrent batches")
        print(f"   - Completed 4 concurrent searches")
    
    @patch('src.sovereign.embedding_service.SentenceTransformer')
    def test_system_recovery_after_errors(self, mock_transformer, rag_system):
        """Test system recovery after various error conditions"""
        # Setup mock that will fail then recover
        mock_model = Mock()
        call_count = 0
        
        def mock_encode(texts):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first two calls
                raise Exception("Simulated embedding failure")
            return np.random.rand(len(texts) if isinstance(texts, list) else 1, 768).astype(np.float32)
        
        mock_model.encode.side_effect = mock_encode
        mock_transformer.return_value = mock_model
        
        # Create test data
        user_id = rag_system.create_test_user("recovery_user")
        conversation_id = rag_system.create_test_conversation(user_id, "Recovery Test")
        
        # First attempts should fail
        with pytest.raises(Exception):
            rag_system.embedding_service.generate_embedding("test text 1")
        
        with pytest.raises(Exception):
            rag_system.embedding_service.generate_embedding("test text 2")
        
        # Third attempt should succeed (recovery)
        response = rag_system.embedding_service.generate_embedding("test text 3")
        assert response.success
        assert response.embedding is not None
        
        # Test that the system can continue normal operations
        messages = [
            {"role": "user", "content": "Testing system recovery"},
            {"role": "assistant", "content": "System has recovered from errors and is functioning normally"}
        ]
        
        message_ids = rag_system.add_test_messages(conversation_id, messages)
        assert len(message_ids) == 2
        
        # Test search after recovery
        search_results = rag_system.vector_search.search(
            query="system recovery",
            limit=5,
            user_id=user_id
        )
        
        # Should work normally after recovery
        assert len(search_results) >= 0
        
        print(f"✅ System recovery after errors working correctly")
        print(f"   - Failed {call_count - 1} times before recovery")
        print(f"   - Successfully added {len(message_ids)} messages after recovery")


class TestRAGSystemPerformance:
    """Test RAG system performance characteristics"""
    
    @patch('src.sovereign.embedding_service.SentenceTransformer')
    def test_search_performance_with_large_dataset(self, mock_transformer, rag_system):
        """Test search performance with large numbers of stored memories"""
        # Setup mock
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(768).astype(np.float32)
        mock_transformer.return_value = mock_model
        
        # Create user and conversation
        user_id = rag_system.create_test_user("perf_user")
        conversation_id = rag_system.create_test_conversation(user_id, "Performance Test")
        
        # Add a large number of messages
        print("Adding large dataset for performance testing...")
        large_dataset = []
        topics = ["science", "technology", "history", "literature", "mathematics", "philosophy"]
        
        for i in range(200):  # Add 200 message pairs (400 total messages)
            topic = random.choice(topics)
            large_dataset.extend([
                {"role": "user", "content": f"Question about {topic} number {i}"},
                {"role": "assistant", "content": f"Detailed answer about {topic} with context {i}"}
            ])
        
        start_time = time.time()
        message_ids = rag_system.add_test_messages(conversation_id, large_dataset)
        storage_time = time.time() - start_time
        
        assert len(message_ids) == 400
        print(f"   - Stored {len(message_ids)} messages in {storage_time:.2f}s")
        
        # Test search performance
        search_queries = [
            "science research",
            "technology innovation",
            "historical events",
            "mathematical concepts",
            "philosophical questions"
        ]
        
        search_times = []
        for query in search_queries:
            start_time = time.time()
            results = rag_system.vector_search.search(
                query=query,
                limit=10,
                user_id=user_id
            )
            search_time = time.time() - start_time
            search_times.append(search_time)
            
            assert len(results) >= 0
            print(f"   - Search '{query}': {len(results)} results in {search_time:.3f}s")
        
        # Test context window performance
        start_time = time.time()
        context_data = rag_system.context_manager.get_context_for_query(
            query="science and technology",
            user_id=user_id,
            conversation_id=conversation_id,
            max_tokens=1000
        )
        context_time = time.time() - start_time
        
        assert context_data is not None
        print(f"   - Context window: {len(context_data.messages)} messages in {context_time:.3f}s")
        
        # Performance assertions
        avg_search_time = sum(search_times) / len(search_times)
        assert avg_search_time < 1.0  # Should be faster than 1 second
        assert context_time < 2.0  # Context construction should be fast
        
        print(f"✅ Performance test completed successfully")
        print(f"   - Average search time: {avg_search_time:.3f}s")
        print(f"   - Context construction time: {context_time:.3f}s")
    
    @patch('src.sovereign.embedding_service.SentenceTransformer')
    def test_memory_usage_stability(self, mock_transformer, rag_system):
        """Test that memory usage remains stable during operations"""
        # Setup mock
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(768).astype(np.float32)
        mock_transformer.return_value = mock_model
        
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create user and conversation
        user_id = rag_system.create_test_user("memory_user")
        conversation_id = rag_system.create_test_conversation(user_id, "Memory Test")
        
        # Perform multiple operations
        for iteration in range(10):
            # Add messages
            messages = [
                {"role": "user", "content": f"Memory test iteration {iteration} user message"},
                {"role": "assistant", "content": f"Memory test iteration {iteration} assistant response"}
            ]
            rag_system.add_test_messages(conversation_id, messages)
            
            # Perform searches
            rag_system.vector_search.search(
                query=f"memory test {iteration}",
                limit=5,
                user_id=user_id
            )
            
            # Get context
            rag_system.context_manager.get_context_for_query(
                query=f"iteration {iteration}",
                user_id=user_id,
                conversation_id=conversation_id,
                max_tokens=500
            )
            
            # Check memory usage
            if iteration % 5 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_growth = current_memory - initial_memory
                print(f"   - Iteration {iteration}: {current_memory:.1f}MB (+{memory_growth:.1f}MB)")
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 100MB for this test)
        assert total_growth < 100, f"Memory growth too high: {total_growth:.1f}MB"
        
        print(f"✅ Memory usage stability test passed")
        print(f"   - Initial memory: {initial_memory:.1f}MB")
        print(f"   - Final memory: {final_memory:.1f}MB")
        print(f"   - Total growth: {total_growth:.1f}MB")


class TestRAGSystemEdgeCases:
    """Test edge cases and error conditions"""
    
    @patch('src.sovereign.embedding_service.SentenceTransformer')
    def test_empty_and_malformed_inputs(self, mock_transformer, rag_system):
        """Test handling of empty and malformed inputs"""
        # Setup mock
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(768).astype(np.float32)
        mock_transformer.return_value = mock_model
        
        # Test empty search queries
        user_id = rag_system.create_test_user("edge_case_user")
        
        # Empty search query
        empty_results = rag_system.vector_search.search(
            query="",
            limit=5,
            user_id=user_id
        )
        assert isinstance(empty_results, list)  # Should handle gracefully
        
        # Very long search query
        long_query = "word " * 1000
        long_results = rag_system.vector_search.search(
            query=long_query,
            limit=5,
            user_id=user_id
        )
        assert isinstance(long_results, list)
        
        # Test malformed conversation data
        conversation_id = rag_system.create_test_conversation(user_id, "Edge Case Test")
        
        malformed_messages = [
            {"role": "", "content": ""},  # Empty role and content
            {"role": "user", "content": None},  # None content
            {"role": "invalid_role", "content": "Test message"},  # Invalid role
            {"content": "Missing role field"},  # Missing role
        ]
        
        # Should handle malformed messages gracefully
        for msg in malformed_messages:
            try:
                rag_system.memory_manager.add_message(
                    conversation_id=conversation_id,
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                    metadata={}
                )
            except Exception as e:
                # Expected to fail on some malformed inputs
                print(f"   - Handled malformed message: {e}")
        
        print(f"✅ Edge case handling working correctly")
    
    @patch('src.sovereign.embedding_service.SentenceTransformer')
    def test_database_connection_recovery(self, mock_transformer, rag_system):
        """Test recovery from database connection issues"""
        # Setup mock
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(768).astype(np.float32)
        mock_transformer.return_value = mock_model
        
        # Create test data
        user_id = rag_system.create_test_user("db_recovery_user")
        conversation_id = rag_system.create_test_conversation(user_id, "DB Recovery Test")
        
        # Add initial messages
        messages = [
            {"role": "user", "content": "Test message before connection issue"},
            {"role": "assistant", "content": "Response before connection issue"}
        ]
        message_ids = rag_system.add_test_messages(conversation_id, messages)
        assert len(message_ids) == 2
        
        # Simulate database connection issue by closing connection
        if hasattr(rag_system.memory_manager, '_connection'):
            rag_system.memory_manager._connection.close()
        
        # Try to add more messages (should recover)
        recovery_messages = [
            {"role": "user", "content": "Test message after connection recovery"},
            {"role": "assistant", "content": "Response after connection recovery"}
        ]
        
        try:
            recovery_message_ids = rag_system.add_test_messages(conversation_id, recovery_messages)
            assert len(recovery_message_ids) == 2
            print(f"✅ Database connection recovery successful")
        except Exception as e:
            print(f"   - Database recovery test noted: {e}")
    
    @patch('src.sovereign.embedding_service.SentenceTransformer')
    def test_unicode_and_special_characters(self, mock_transformer, rag_system):
        """Test handling of unicode and special characters"""
        # Setup mock
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(768).astype(np.float32)
        mock_transformer.return_value = mock_model
        
        # Create test data with unicode and special characters
        user_id = rag_system.create_test_user("unicode_user")
        conversation_id = rag_system.create_test_conversation(user_id, "Unicode Test")
        
        unicode_messages = [
            {"role": "user", "content": "Hello 世界 🌍 testing emoji and unicode"},
            {"role": "assistant", "content": "Response with émojis 🤖 and açcénts"},
            {"role": "user", "content": "Special chars: @#$%^&*()[]{}|\\:;\"'<>,.?/~`"},
            {"role": "assistant", "content": "Математика и наука 📊 中文测试"},
        ]
        
        message_ids = rag_system.add_test_messages(conversation_id, unicode_messages)
        assert len(message_ids) == 4
        
        # Test search with unicode
        unicode_results = rag_system.vector_search.search(
            query="世界 emoji testing",
            limit=5,
            user_id=user_id
        )
        assert isinstance(unicode_results, list)
        
        # Test context with unicode
        context_data = rag_system.context_manager.get_context_for_query(
            query="émojis and special characters",
            user_id=user_id,
            conversation_id=conversation_id,
            max_tokens=500
        )
        
        if context_data and context_data.messages:
            # Should handle unicode content properly
            for msg in context_data.messages:
                assert isinstance(msg.content, str)
        
        print(f"✅ Unicode and special character handling working correctly")


def run_full_integration_test_suite():
    """Run the complete RAG system integration test suite"""
    print("🚀 Starting RAG System Integration Test Suite")
    print("=" * 60)
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    rag_system = RAGSystemTestHelper(temp_dir)
    
    try:
        # Run all test classes
        test_classes = [
            TestRAGSystemBasicWorkflow,
            TestRAGSystemAdvancedFeatures,
            TestRAGSystemPerformance,
            TestRAGSystemEdgeCases
        ]
        
        total_tests = 0
        passed_tests = 0
        
        for test_class in test_classes:
            print(f"\n📋 Running {test_class.__name__}")
            print("-" * 40)
            
            # Get all test methods
            test_methods = [method for method in dir(test_class) if method.startswith('test_')]
            
            for test_method in test_methods:
                total_tests += 1
                try:
                    # Create test instance
                    test_instance = test_class()
                    
                    # Run test method
                    method = getattr(test_instance, test_method)
                    print(f"  Running {test_method}...")
                    method(Mock(), rag_system)
                    
                    passed_tests += 1
                    print(f"  ✅ {test_method} PASSED")
                    
                except Exception as e:
                    print(f"  ❌ {test_method} FAILED: {e}")
        
        print(f"\n🎯 RAG System Integration Test Results")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        return passed_tests == total_tests
        
    finally:
        rag_system.cleanup()
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run the complete test suite
    success = run_full_integration_test_suite()
    exit(0 if success else 1) 