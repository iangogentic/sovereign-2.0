"""
Vector Search Engine Demonstration

This script demonstrates the high-performance vector search capabilities using FAISS
for semantic similarity search over conversation history and memory chunks.
"""

import asyncio
import tempfile
import sqlite3
import time
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from src.sovereign.vector_search_engine import (
    VectorSearchEngine, SearchParameters, create_vector_search_engine, search_memories
)
from src.sovereign.embedding_service import EmbeddingService
from src.sovereign.memory_schema import MemorySchema
from src.sovereign.config import Config


class MockEmbeddingService:
    """Mock embedding service for demonstration"""
    
    def __init__(self):
        self.current_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_dim = 384
        
        # Create consistent embeddings for different content types
        self.content_embeddings = {
            # AI/ML related content
            "ai": np.random.RandomState(42).normal(0.5, 0.1, 384),
            "machine_learning": np.random.RandomState(43).normal(0.5, 0.1, 384),
            "neural_networks": np.random.RandomState(44).normal(0.5, 0.1, 384),
            "deep_learning": np.random.RandomState(45).normal(0.5, 0.1, 384),
            
            # Programming related content
            "python": np.random.RandomState(100).normal(0.2, 0.1, 384),
            "programming": np.random.RandomState(101).normal(0.2, 0.1, 384),
            "software": np.random.RandomState(102).normal(0.2, 0.1, 384),
            "code": np.random.RandomState(103).normal(0.2, 0.1, 384),
            
            # Web development content
            "web": np.random.RandomState(200).normal(-0.2, 0.1, 384),
            "javascript": np.random.RandomState(201).normal(-0.2, 0.1, 384),
            "react": np.random.RandomState(202).normal(-0.2, 0.1, 384),
            "frontend": np.random.RandomState(203).normal(-0.2, 0.1, 384),
            
            # Database content
            "database": np.random.RandomState(300).normal(-0.5, 0.1, 384),
            "sql": np.random.RandomState(301).normal(-0.5, 0.1, 384),
            "data": np.random.RandomState(302).normal(-0.5, 0.1, 384),
        }
    
    async def get_embedding(self, text: str) -> List[float]:
        """Generate embedding based on content"""
        text_lower = text.lower()
        
        # Find best matching content type
        best_match = None
        for content_type in self.content_embeddings:
            if content_type in text_lower:
                best_match = content_type
                break
        
        if best_match:
            base_embedding = self.content_embeddings[best_match]
        else:
            # Default random embedding
            base_embedding = np.random.normal(0, 0.1, 384)
        
        # Add some noise for variety
        noise = np.random.normal(0, 0.05, 384)
        embedding = base_embedding + noise
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()


async def setup_demo_database(db_path: Path) -> None:
    """Set up a comprehensive demo database with realistic conversation data"""
    print("🗄️ Setting up demo database with realistic conversation data...")
    
    # Create database schema
    memory_schema = MemorySchema(str(db_path))
    memory_schema.create_schema()
    memory_schema.close()
    
    with sqlite3.connect(db_path) as conn:
        
        # Realistic conversation data
        conversations = [
            ("conv_ai_basics", "AI and Machine Learning Fundamentals", "2024-01-15T09:00:00"),
            ("conv_python_tutorial", "Python Programming Tutorial", "2024-01-15T14:00:00"),
            ("conv_web_dev", "Modern Web Development Discussion", "2024-01-16T10:00:00"),
            ("conv_databases", "Database Design and Optimization", "2024-01-16T15:00:00"),
            ("conv_neural_nets", "Deep Learning and Neural Networks", "2024-01-17T11:00:00"),
        ]
        
        for conv_id, title, created_at in conversations:
            conn.execute("""
                INSERT INTO conversations (conversation_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            """, (conv_id, title, created_at, created_at))
        
        # Realistic memory chunks with varied content
        chunks = [
            # AI/ML conversation chunks
            ("chunk_001", "What is artificial intelligence? AI refers to the simulation of human intelligence in machines that are programmed to think and learn like humans.", "2024-01-15T09:15:00", "conv_ai_basics", "text"),
            ("chunk_002", "Machine learning is a subset of AI that focuses on algorithms that can learn and improve from experience without being explicitly programmed.", "2024-01-15T09:30:00", "conv_ai_basics", "text"),
            ("chunk_003", "Supervised learning uses labeled training data to learn a mapping from input to output. Examples include classification and regression tasks.", "2024-01-15T09:45:00", "conv_ai_basics", "text"),
            ("chunk_004", "Unsupervised learning finds hidden patterns in data without labeled examples. Clustering and dimensionality reduction are common techniques.", "2024-01-15T10:00:00", "conv_ai_basics", "text"),
            
            # Python programming chunks
            ("chunk_005", "Python is a high-level programming language known for its simplicity and readability. It's excellent for beginners and powerful for experts.", "2024-01-15T14:15:00", "conv_python_tutorial", "text"),
            ("chunk_006", "Python functions are defined using the 'def' keyword. They help organize code into reusable blocks and improve maintainability.", "2024-01-15T14:30:00", "conv_python_tutorial", "text"),
            ("chunk_007", "List comprehensions in Python provide a concise way to create lists. For example: [x**2 for x in range(10)] creates a list of squares.", "2024-01-15T14:45:00", "conv_python_tutorial", "text"),
            ("chunk_008", "Python dictionaries are key-value pairs that allow fast lookups. They're similar to hash maps in other programming languages.", "2024-01-15T15:00:00", "conv_python_tutorial", "text"),
            
            # Web development chunks
            ("chunk_009", "React is a JavaScript library for building user interfaces. It uses a component-based architecture and virtual DOM for efficient updates.", "2024-01-16T10:15:00", "conv_web_dev", "text"),
            ("chunk_010", "Modern web development involves frontend frameworks like React, Vue, or Angular, paired with backend APIs built with Node.js, Python, or other languages.", "2024-01-16T10:30:00", "conv_web_dev", "text"),
            ("chunk_011", "JavaScript ES6+ features like arrow functions, destructuring, and async/await have greatly improved the developer experience.", "2024-01-16T10:45:00", "conv_web_dev", "text"),
            ("chunk_012", "CSS Grid and Flexbox are powerful layout systems that make it easier to create responsive web designs without complex float-based layouts.", "2024-01-16T11:00:00", "conv_web_dev", "text"),
            
            # Database chunks
            ("chunk_013", "Database normalization is the process of organizing data to reduce redundancy and improve data integrity. It involves dividing tables and establishing relationships.", "2024-01-16T15:15:00", "conv_databases", "text"),
            ("chunk_014", "SQL joins allow you to combine data from multiple tables. INNER JOIN returns only matching records, while LEFT JOIN includes all records from the left table.", "2024-01-16T15:30:00", "conv_databases", "text"),
            ("chunk_015", "Database indexing improves query performance by creating data structures that allow faster lookups. However, indexes consume storage and slow down writes.", "2024-01-16T15:45:00", "conv_databases", "text"),
            ("chunk_016", "ACID properties (Atomicity, Consistency, Isolation, Durability) ensure database transactions are processed reliably and maintain data integrity.", "2024-01-16T16:00:00", "conv_databases", "text"),
            
            # Deep learning chunks
            ("chunk_017", "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers.", "2024-01-17T11:15:00", "conv_neural_nets", "text"),
            ("chunk_018", "Backpropagation is the algorithm used to train neural networks. It calculates gradients by propagating errors backward through the network.", "2024-01-17T11:30:00", "conv_neural_nets", "text"),
            ("chunk_019", "Convolutional Neural Networks (CNNs) are particularly effective for image recognition tasks. They use convolution operations to detect local features.", "2024-01-17T11:45:00", "conv_neural_nets", "text"),
            ("chunk_020", "Transformer architectures have revolutionized natural language processing. They use attention mechanisms to process sequences in parallel.", "2024-01-17T12:00:00", "conv_neural_nets", "text"),
        ]
        
        for chunk_id, content, timestamp, conv_id, chunk_type in chunks:
            conn.execute("""
                INSERT INTO memory_chunks (chunk_id, content, timestamp, conversation_id, chunk_type, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (chunk_id, content, timestamp, conv_id, chunk_type, json.dumps({"length": len(content)})))
        
        conn.commit()
    
    print(f"✅ Created demo database with {len(conversations)} conversations and {len(chunks)} memory chunks")


async def generate_embeddings_for_demo(db_path: Path, embedding_service: MockEmbeddingService) -> None:
    """Generate embeddings for all chunks in the demo database"""
    print("🧠 Generating embeddings for all memory chunks...")
    
    with sqlite3.connect(db_path) as conn:
        # Get all chunks
        cursor = conn.execute("SELECT chunk_id, content FROM memory_chunks ORDER BY chunk_id")
        chunks = cursor.fetchall()
        
        # Generate embeddings
        for chunk_id, content in chunks:
            embedding = await embedding_service.get_embedding(content)
            embedding_array = np.array(embedding, dtype=np.float32)
            embedding_blob = sqlite3.Binary(embedding_array.tobytes())
            
            conn.execute("""
                INSERT INTO memory_embeddings (chunk_id, embedding_data, embedding_model, embedding_timestamp)
                VALUES (?, ?, ?, '2024-01-17T12:00:00')
            """, (chunk_id, embedding_blob, embedding_service.current_model))
        
        conn.commit()
    
    print(f"✅ Generated embeddings for {len(chunks)} chunks using {embedding_service.current_model}")


async def demonstrate_index_creation(engine: VectorSearchEngine) -> None:
    """Demonstrate index creation and optimization"""
    print("\n🏗️ Index Creation and Management Demo")
    print("=" * 50)
    
    # Get initial stats
    initial_stats = await engine.get_index_stats()
    print(f"Initial state - Index exists: {initial_stats['index_exists']}")
    
    # Build index
    print("\n📊 Building FAISS index from embeddings...")
    start_time = time.time()
    success = await engine.build_index()
    build_time = time.time() - start_time
    
    if success:
        print(f"✅ Index built successfully in {build_time:.2f}s")
        
        # Get post-build stats
        stats = await engine.get_index_stats()
        print(f"📈 Index Statistics:")
        print(f"   - Total vectors: {stats['total_vectors']}")
        print(f"   - Index type: {stats['index_type']}")
        print(f"   - Embedding dimension: {stats['embedding_dim']}")
        print(f"   - Embedding model: {stats['embedding_model']}")
        print(f"   - Total chunks in DB: {stats['total_chunks']}")
        print(f"   - Total embeddings in DB: {stats['total_embeddings']}")
        
        # Test index persistence
        print(f"\n💾 Testing index persistence...")
        save_success = await engine.save_index()
        print(f"{'✅' if save_success else '❌'} Index {'saved' if save_success else 'failed to save'}")
        
    else:
        print("❌ Failed to build index")


async def demonstrate_semantic_search(engine: VectorSearchEngine) -> None:
    """Demonstrate various semantic search capabilities"""
    print("\n🔍 Semantic Search Demonstration")
    print("=" * 50)
    
    # Test queries with different domains
    test_queries = [
        ("AI and machine learning concepts", "AI/ML Domain"),
        ("Python programming and software development", "Programming Domain"),
        ("Web development with JavaScript", "Web Development Domain"),
        ("Database design and SQL queries", "Database Domain"),
        ("Neural networks and deep learning", "Deep Learning Domain"),
    ]
    
    for query, domain in test_queries:
        print(f"\n🎯 Query: '{query}' ({domain})")
        print("-" * 40)
        
        start_time = time.time()
        results = await engine.search(query, SearchParameters(top_k=3, include_metadata=True))
        search_time = time.time() - start_time
        
        print(f"⏱️ Search completed in {search_time:.3f}s")
        print(f"📋 Found {len(results)} results:")
        
        for i, result in enumerate(results, 1):
            print(f"\n   {i}. Chunk ID: {result.chunk_id}")
            print(f"      Score: {result.score:.3f} (Distance: {result.distance:.3f})")
            print(f"      Conversation: {result.metadata.get('conversation_title', 'Unknown')}")
            print(f"      Content: {result.content[:100]}...")


async def demonstrate_advanced_filtering(engine: VectorSearchEngine) -> None:
    """Demonstrate advanced search filtering capabilities"""
    print("\n🎛️ Advanced Filtering Demonstration")
    print("=" * 50)
    
    query = "programming and software development"
    
    # Basic search
    print(f"🔍 Basic search for: '{query}'")
    basic_results = await engine.search(query, SearchParameters(top_k=5, include_metadata=True))
    print(f"   Found {len(basic_results)} results")
    
    # Filter by conversation
    print(f"\n🎯 Filtered by conversation 'conv_python_tutorial':")
    filtered_results = await engine.search(
        query, 
        SearchParameters(
            top_k=5, 
            include_metadata=True,
            filter_conversation_id="conv_python_tutorial"
        )
    )
    print(f"   Found {len(filtered_results)} results")
    for result in filtered_results:
        print(f"   - {result.chunk_id}: {result.content[:80]}...")
    
    # Filter by timestamp range
    print(f"\n📅 Filtered by timestamp range (2024-01-15):")
    time_filtered_results = await engine.search(
        query,
        SearchParameters(
            top_k=5,
            include_metadata=True,
            filter_timestamp_range=("2024-01-15T00:00:00", "2024-01-15T23:59:59")
        )
    )
    print(f"   Found {len(time_filtered_results)} results")
    for result in time_filtered_results:
        print(f"   - {result.chunk_id}: {result.timestamp}")
    
    # High confidence threshold
    print(f"\n🎯 High confidence threshold (>0.8):")
    high_conf_results = await engine.search(
        query,
        SearchParameters(
            top_k=5,
            include_metadata=True,
            min_confidence=0.8
        )
    )
    print(f"   Found {len(high_conf_results)} high-confidence results")
    for result in high_conf_results:
        print(f"   - {result.chunk_id}: Score {result.score:.3f}")


async def demonstrate_batch_search(engine: VectorSearchEngine) -> None:
    """Demonstrate batch search capabilities"""
    print("\n🔄 Batch Search Demonstration")
    print("=" * 50)
    
    batch_queries = [
        "machine learning algorithms",
        "Python programming basics",
        "React web development",
        "SQL database queries",
        "neural network training"
    ]
    
    print(f"🔍 Performing batch search with {len(batch_queries)} queries...")
    
    # Time batch search
    start_time = time.time()
    batch_results = await engine.batch_search(
        batch_queries, 
        SearchParameters(top_k=2, include_metadata=True)
    )
    batch_time = time.time() - start_time
    
    print(f"⏱️ Batch search completed in {batch_time:.3f}s")
    print(f"📊 Results summary:")
    
    for i, (query, results) in enumerate(zip(batch_queries, batch_results)):
        print(f"   Query {i+1}: '{query[:30]}...' → {len(results)} results")
        if results:
            best_result = results[0]
            print(f"      Best match: {best_result.chunk_id} (Score: {best_result.score:.3f})")
    
    # Compare with sequential search
    print(f"\n⚡ Performance comparison:")
    start_time = time.time()
    sequential_results = []
    for query in batch_queries:
        results = await engine.search(query, SearchParameters(top_k=2, include_metadata=True))
        sequential_results.append(results)
    sequential_time = time.time() - start_time
    
    speedup = sequential_time / batch_time if batch_time > 0 else 0
    print(f"   Batch search: {batch_time:.3f}s")
    print(f"   Sequential search: {sequential_time:.3f}s")
    print(f"   Speedup: {speedup:.1f}x faster")


async def demonstrate_vector_operations(engine: VectorSearchEngine, embedding_service: MockEmbeddingService) -> None:
    """Demonstrate vector addition and removal"""
    print("\n➕ Vector Operations Demonstration")
    print("=" * 50)
    
    # Get initial stats
    initial_stats = await engine.get_index_stats()
    print(f"📊 Initial index size: {initial_stats['total_vectors']} vectors")
    
    # Add new vectors
    print(f"\n➕ Adding new vectors to index...")
    new_chunks = [
        ("chunk_new_1", "Quantum computing represents the next frontier in computational power, leveraging quantum mechanics principles."),
        ("chunk_new_2", "Blockchain technology enables decentralized systems with cryptographic security and immutable ledgers."),
    ]
    
    new_chunk_ids = []
    new_embeddings = []
    
    for chunk_id, content in new_chunks:
        embedding = await embedding_service.get_embedding(content)
        new_chunk_ids.append(chunk_id)
        new_embeddings.append(embedding)
    
    new_embeddings_array = np.array(new_embeddings, dtype=np.float32)
    
    success = await engine.add_vectors(new_chunk_ids, new_embeddings_array)
    print(f"{'✅' if success else '❌'} {'Added' if success else 'Failed to add'} {len(new_chunk_ids)} new vectors")
    
    # Verify addition
    updated_stats = await engine.get_index_stats()
    print(f"📊 Updated index size: {updated_stats['total_vectors']} vectors")
    
    # Test search with new content
    print(f"\n🔍 Testing search with new content...")
    results = await engine.search("quantum computing and blockchain", SearchParameters(top_k=3))
    print(f"   Found {len(results)} results including new content")
    for result in results:
        if result.chunk_id in new_chunk_ids:
            print(f"   ✨ New chunk found: {result.chunk_id} (Score: {result.score:.3f})")


async def demonstrate_performance_monitoring(engine: VectorSearchEngine) -> None:
    """Demonstrate performance monitoring and optimization"""
    print("\n📈 Performance Monitoring Demonstration")
    print("=" * 50)
    
    # Perform multiple searches to generate stats
    print("🔍 Performing multiple searches to generate performance data...")
    test_queries = [
        "artificial intelligence",
        "machine learning",
        "python programming",
        "web development",
        "database optimization"
    ]
    
    for query in test_queries:
        await engine.search(query, SearchParameters(top_k=5))
    
    # Get comprehensive stats
    stats = await engine.get_index_stats()
    perf_stats = stats['performance_stats']
    
    print(f"\n📊 Performance Statistics:")
    print(f"   Total searches performed: {perf_stats['total_searches']}")
    print(f"   Total search time: {perf_stats['total_search_time']:.3f}s")
    print(f"   Average search time: {perf_stats['total_search_time'] / max(perf_stats['total_searches'], 1):.3f}s")
    print(f"   Cache hits: {perf_stats['cache_hits']}")
    print(f"   Cache hit rate: {perf_stats['cache_hits'] / max(perf_stats['total_searches'], 1) * 100:.1f}%")
    print(f"   Current cache size: {stats['cache_size']} entries")
    
    # Test optimization
    print(f"\n⚡ Running index optimization...")
    optimization_success = await engine.optimize_index()
    print(f"{'✅' if optimization_success else '❌'} Index optimization {'completed' if optimization_success else 'failed'}")


async def main():
    """Run the comprehensive vector search engine demonstration"""
    print("🚀 Vector Search Engine Demonstration")
    print("=====================================")
    print("This demo showcases FAISS-based semantic search capabilities")
    print("with realistic conversation data and advanced features.\n")
    
    # Setup
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "demo_memory.db"
    
    try:
        # Initialize services
        embedding_service = MockEmbeddingService()
        
        # Setup demo data
        await setup_demo_database(db_path)
        await generate_embeddings_for_demo(db_path, embedding_service)
        
        # Create vector search engine
        print("🔧 Initializing Vector Search Engine...")
        engine = await create_vector_search_engine(
            db_path=str(db_path),
            embedding_service=embedding_service,
            index_dir=str(Path(temp_dir) / "indices")
        )
        
        # Run demonstrations
        await demonstrate_index_creation(engine)
        await demonstrate_semantic_search(engine)
        await demonstrate_advanced_filtering(engine)
        await demonstrate_batch_search(engine)
        await demonstrate_vector_operations(engine, embedding_service)
        await demonstrate_performance_monitoring(engine)
        
        # Final summary
        print(f"\n🎉 Demonstration Complete!")
        print("=" * 50)
        final_stats = await engine.get_index_stats()
        print(f"Final Statistics:")
        print(f"   - Total vectors in index: {final_stats['total_vectors']}")
        print(f"   - Total searches performed: {final_stats['performance_stats']['total_searches']}")
        print(f"   - Index type: {final_stats['index_type']}")
        print(f"   - Average search performance: {final_stats['performance_stats']['total_search_time'] / max(final_stats['performance_stats']['total_searches'], 1):.3f}s per query")
        
        # Cleanup
        await engine.cleanup()
        print(f"\n✅ Demonstration completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup temp directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main()) 