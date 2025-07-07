# Sovereign AI API Reference

## Overview

This document provides comprehensive API reference for all public interfaces in the Sovereign AI system. It includes detailed method signatures, parameters, return values, and usage examples.

## Core APIs

### 1. Model Orchestrator API

The ModelOrchestrator is the primary interface for interacting with the Sovereign AI system.

#### Class: `ModelOrchestrator`

```python
from sovereign import ModelOrchestrator, QueryContext, Config

# Initialize orchestrator
config = Config()
orchestrator = ModelOrchestrator(config)
await orchestrator.initialize()
```

#### Methods

##### `process_query(user_input: str, context: Optional[QueryContext] = None) -> OrchestrationResult`

Process a user query and return an intelligent response.

**Parameters:**
- `user_input` (str): The user's query or input text
- `context` (Optional[QueryContext]): Additional context for the query

**Returns:**
- `OrchestrationResult`: Complete result with response and metadata

**Example:**
```python
from sovereign import QueryContext
from datetime import datetime

# Simple query
result = await orchestrator.process_query("What is the weather like?")
print(f"Response: {result.response}")
print(f"Model used: {result.model_used}")
print(f"Processing time: {result.processing_time}s")

# Query with context
context = QueryContext(
    user_input="What about tomorrow?",
    timestamp=datetime.now(),
    session_id="user123",
    previous_queries=["What is the weather like?"],
    conversation_history=[
        {"role": "user", "content": "What is the weather like?"},
        {"role": "assistant", "content": "Today is sunny with 75°F"}
    ]
)

result = await orchestrator.process_query("What about tomorrow?", context)
```

##### `determine_complexity(query: str, context: Optional[QueryContext] = None) -> Tuple[QueryComplexity, float]`

Analyze query complexity to determine appropriate model selection.

**Parameters:**
- `query` (str): Query text to analyze
- `context` (Optional[QueryContext]): Additional context

**Returns:**
- `Tuple[QueryComplexity, float]`: Complexity level and confidence score

**Example:**
```python
from sovereign import QueryComplexity

complexity, confidence = orchestrator.determine_complexity("Hello!")
# Returns: (QueryComplexity.SIMPLE, 0.95)

complexity, confidence = orchestrator.determine_complexity(
    "Analyze the performance implications of implementing a distributed caching layer"
)
# Returns: (QueryComplexity.COMPLEX, 0.87)
```

##### `get_status() -> Dict[str, Any]`

Get current system status and health information.

**Returns:**
- `Dict[str, Any]`: System status information

**Example:**
```python
status = await orchestrator.get_status()
print(f"System uptime: {status['uptime']}")
print(f"Memory usage: {status['memory_usage']}")
print(f"Active models: {status['active_models']}")
```

### 2. Memory Management API

#### Class: `MemoryManager`

```python
from sovereign.memory_manager import MemoryManager, Memory, MemoryType

memory_manager = MemoryManager()
await memory_manager.initialize()
```

#### Methods

##### `store_memory(memory: Memory) -> str`

Store a new memory in the system.

**Parameters:**
- `memory` (Memory): Memory object to store

**Returns:**
- `str`: Unique memory ID

**Example:**
```python
from sovereign.memory_manager import Memory, MemoryType
from datetime import datetime

memory = Memory(
    content="User prefers dark mode interface",
    memory_type=MemoryType.PREFERENCE,
    importance=0.8,
    tags=["ui", "preferences"],
    timestamp=datetime.now()
)

memory_id = await memory_manager.store_memory(memory)
print(f"Stored memory with ID: {memory_id}")
```

##### `retrieve_memories(query: str, limit: int = 10, filters: Optional[Dict] = None) -> List[Memory]`

Retrieve memories relevant to a query.

**Parameters:**
- `query` (str): Search query
- `limit` (int): Maximum number of memories to return
- `filters` (Optional[Dict]): Search filters

**Returns:**
- `List[Memory]`: List of relevant memories

**Example:**
```python
# Basic search
memories = await memory_manager.retrieve_memories("user preferences", limit=5)

# Search with filters
memories = await memory_manager.retrieve_memories(
    "interface settings",
    limit=10,
    filters={
        "memory_type": MemoryType.PREFERENCE,
        "importance_min": 0.5
    }
)

for memory in memories:
    print(f"Memory: {memory.content}")
    print(f"Relevance: {memory.relevance_score}")
```

##### `update_memory(memory_id: str, updates: Dict[str, Any]) -> bool`

Update an existing memory.

**Parameters:**
- `memory_id` (str): ID of memory to update
- `updates` (Dict[str, Any]): Fields to update

**Returns:**
- `bool`: Success status

**Example:**
```python
success = await memory_manager.update_memory(
    memory_id="mem_123",
    updates={
        "importance": 0.9,
        "tags": ["ui", "preferences", "updated"]
    }
)
```

### 3. Vector Search API

#### Class: `VectorSearchEngine`

```python
from sovereign.vector_search_engine import VectorSearchEngine, Document

search_engine = VectorSearchEngine()
await search_engine.initialize()
```

#### Methods

##### `search(query: str, k: int = 10, filters: Optional[Dict] = None) -> List[SearchResult]`

Perform semantic search on indexed documents.

**Parameters:**
- `query` (str): Search query
- `k` (int): Number of results to return
- `filters` (Optional[Dict]): Search filters

**Returns:**
- `List[SearchResult]`: Search results with scores

**Example:**
```python
# Basic search
results = await search_engine.search("machine learning algorithms", k=5)

for result in results:
    print(f"Document: {result.document.title}")
    print(f"Score: {result.score:.3f}")
    print(f"Content: {result.document.content[:200]}...")

# Search with filters
results = await search_engine.search(
    "neural networks",
    k=10,
    filters={
        "document_type": "research_paper",
        "date_after": "2023-01-01"
    }
)
```

##### `add_document(document: Document) -> str`

Add a new document to the search index.

**Parameters:**
- `document` (Document): Document to index

**Returns:**
- `str`: Document ID

**Example:**
```python
from sovereign.vector_search_engine import Document

document = Document(
    title="Introduction to Neural Networks",
    content="Neural networks are computing systems inspired by biological neural networks...",
    metadata={
        "author": "Dr. Smith",
        "date": "2023-12-01",
        "category": "tutorial"
    }
)

doc_id = await search_engine.add_document(document)
print(f"Added document with ID: {doc_id}")
```

### 4. Tool Framework API

#### Class: `ToolExecutionEngine`

```python
from sovereign.tool_execution_engine import ToolExecutionEngine, ToolCall

tool_engine = ToolExecutionEngine()
await tool_engine.initialize()
```

#### Methods

##### `execute_tool(tool_call: ToolCall) -> ToolResult`

Execute a tool with specified parameters.

**Parameters:**
- `tool_call` (ToolCall): Tool call specification

**Returns:**
- `ToolResult`: Execution result

**Example:**
```python
from sovereign.tool_execution_engine import ToolCall, SandboxLevel

# Execute web search tool
tool_call = ToolCall(
    tool_name="web_search",
    parameters={
        "query": "latest AI research papers",
        "num_results": 5
    },
    timeout=30,
    sandbox_level=SandboxLevel.NETWORK_ALLOWED
)

result = await tool_engine.execute_tool(tool_call)

if result.success:
    print(f"Search results: {result.result}")
else:
    print(f"Error: {result.error_message}")
```

##### `get_available_tools() -> List[ToolDefinition]`

Get list of available tools.

**Returns:**
- `List[ToolDefinition]`: Available tools

**Example:**
```python
tools = await tool_engine.get_available_tools()

for tool in tools:
    print(f"Tool: {tool.name}")
    print(f"Description: {tool.description}")
    print(f"Parameters: {tool.parameters}")
```

### 5. Performance Monitoring API

#### Class: `PerformanceMonitor`

```python
from sovereign.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start_monitoring()
```

#### Methods

##### `get_current_metrics() -> PerformanceMetrics`

Get current system performance metrics.

**Returns:**
- `PerformanceMetrics`: Current metrics

**Example:**
```python
metrics = monitor.get_current_metrics()

print(f"CPU Usage: {metrics.cpu_usage:.1f}%")
print(f"Memory Usage: {metrics.memory_usage:.1f}%")
print(f"GPU Usage: {metrics.gpu_usage:.1f}%")
print(f"Response Time: {metrics.avg_response_time:.3f}s")
```

##### `get_performance_report(start_time: Optional[datetime], end_time: Optional[datetime]) -> Dict[str, Any]`

Generate performance report for specified time period.

**Parameters:**
- `start_time` (Optional[datetime]): Report start time
- `end_time` (Optional[datetime]): Report end time

**Returns:**
- `Dict[str, Any]`: Performance report

**Example:**
```python
from datetime import datetime, timedelta

# Get last hour performance
end_time = datetime.now()
start_time = end_time - timedelta(hours=1)

report = monitor.get_performance_report(start_time, end_time)

print(f"Average CPU: {report['avg_cpu_usage']:.1f}%")
print(f"Peak Memory: {report['peak_memory_usage']:.1f}%")
print(f"Total Queries: {report['total_queries']}")
print(f"Error Rate: {report['error_rate']:.2f}%")
```

### 6. Embedding Service API

#### Class: `EmbeddingService`

```python
from sovereign.embedding_service import EmbeddingService, EmbeddingModelType

embedding_service = EmbeddingService(
    default_model=EmbeddingModelType.BGE_BASE_EN_V15
)
```

#### Methods

##### `generate_embedding(text: str, model_type: Optional[EmbeddingModelType] = None) -> EmbeddingResponse`

Generate embedding for text.

**Parameters:**
- `text` (str): Text to embed
- `model_type` (Optional[EmbeddingModelType]): Specific model to use

**Returns:**
- `EmbeddingResponse`: Embedding result

**Example:**
```python
# Generate embedding with default model
response = embedding_service.generate_embedding("Hello world")

if response.success:
    print(f"Embedding shape: {response.embedding.shape}")
    print(f"Model used: {response.model_used}")
    print(f"Processing time: {response.processing_time:.3f}s")
else:
    print(f"Error: {response.error_message}")

# Generate with specific model
response = embedding_service.generate_embedding(
    "Complex technical document",
    model_type=EmbeddingModelType.E5_LARGE_V2
)
```

##### `generate_embeddings(requests: List[EmbeddingRequest]) -> List[EmbeddingResponse]`

Generate embeddings for multiple texts in batch.

**Parameters:**
- `requests` (List[EmbeddingRequest]): List of embedding requests

**Returns:**
- `List[EmbeddingResponse]`: List of embedding responses

**Example:**
```python
from sovereign.embedding_service import EmbeddingRequest

requests = [
    EmbeddingRequest(text="First document"),
    EmbeddingRequest(text="Second document"),
    EmbeddingRequest(text="Third document")
]

responses = embedding_service.generate_embeddings(requests)

for i, response in enumerate(responses):
    print(f"Document {i+1}: {'Success' if response.success else 'Failed'}")
```

## Data Structures

### Core Data Types

#### `QueryContext`

```python
@dataclass
class QueryContext:
    user_input: str
    timestamp: datetime
    session_id: str
    previous_queries: List[str]
    conversation_history: List[Dict[str, Any]]
    screen_context: Optional[Dict[str, Any]] = None
    voice_context: Optional[Dict[str, Any]] = None
    user_preferences: Optional[Dict[str, Any]] = None
```

#### `OrchestrationResult`

```python
@dataclass
class OrchestrationResult:
    response: str
    model_used: ModelChoice
    complexity_level: QueryComplexity
    processing_time: float
    handoff_occurred: bool
    cache_hit: bool
    confidence_score: float
    reasoning: str
    telemetry: Dict[str, Any]
```

#### `Memory`

```python
@dataclass
class Memory:
    id: str
    content: str
    timestamp: datetime
    memory_type: MemoryType
    importance: float
    tags: List[str]
    embedding: Optional[np.ndarray]
    metadata: Dict[str, Any]
    relevance_score: Optional[float] = None
```

#### `ToolCall`

```python
@dataclass
class ToolCall:
    tool_name: str
    parameters: Dict[str, Any]
    execution_id: str
    timeout: int
    sandbox_level: SandboxLevel
```

#### `ToolResult`

```python
@dataclass
class ToolResult:
    execution_id: str
    success: bool
    result: Any
    error_message: Optional[str]
    execution_time: float
    resources_used: Dict[str, Any]
```

### Enums

#### `QueryComplexity`

```python
class QueryComplexity(Enum):
    SIMPLE = "simple"           # Basic queries, greetings
    MODERATE = "moderate"       # Questions requiring context
    COMPLEX = "complex"         # Multi-step reasoning
    VERY_COMPLEX = "very_complex"  # Advanced analysis
```

#### `ModelChoice`

```python
class ModelChoice(Enum):
    TALKER = "talker"      # Fast response model
    THINKER = "thinker"    # Complex reasoning model
    BOTH = "both"          # Collaborative processing
```

#### `MemoryType`

```python
class MemoryType(Enum):
    CONVERSATION = "conversation"    # Chat history
    PREFERENCE = "preference"        # User preferences
    KNOWLEDGE = "knowledge"          # Learned facts
    CONTEXT = "context"             # Environmental context
    TASK = "task"                   # Task-related information
```

#### `EmbeddingModelType`

```python
class EmbeddingModelType(Enum):
    BGE_BASE_EN_V15 = "BAAI/bge-base-en-v1.5"
    E5_LARGE_V2 = "intfloat/e5-large-v2"
    MINILM_L6_V2 = "sentence-transformers/all-MiniLM-L6-v2"
```

## Usage Examples

### 1. Basic Chat Application

```python
import asyncio
from sovereign import ModelOrchestrator, Config, QueryContext
from datetime import datetime

async def simple_chat():
    # Initialize system
    config = Config()
    orchestrator = ModelOrchestrator(config)
    await orchestrator.initialize()
    
    session_id = "chat_session_1"
    conversation_history = []
    
    try:
        while True:
            # Get user input
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            
            # Create context
            context = QueryContext(
                user_input=user_input,
                timestamp=datetime.now(),
                session_id=session_id,
                previous_queries=[h['content'] for h in conversation_history if h['role'] == 'user'],
                conversation_history=conversation_history
            )
            
            # Process query
            result = await orchestrator.process_query(user_input, context)
            
            # Display response
            print(f"Assistant: {result.response}")
            print(f"(Model: {result.model_used.value}, Time: {result.processing_time:.2f}s)")
            
            # Update conversation history
            conversation_history.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": result.response}
            ])
    
    finally:
        await orchestrator.close()

# Run the chat
asyncio.run(simple_chat())
```

### 2. Memory-Enhanced Application

```python
import asyncio
from sovereign import ModelOrchestrator, Config
from sovereign.memory_manager import MemoryManager, Memory, MemoryType

async def memory_enhanced_app():
    # Initialize system with memory
    config = Config()
    orchestrator = ModelOrchestrator(config)
    await orchestrator.initialize()
    
    memory_manager = orchestrator.memory_manager
    
    # Store some initial memories
    user_prefs = Memory(
        content="User prefers technical explanations with code examples",
        memory_type=MemoryType.PREFERENCE,
        importance=0.9,
        tags=["user_preference", "communication_style"]
    )
    
    await memory_manager.store_memory(user_prefs)
    
    # Use memories in conversation
    user_input = "Explain how neural networks work"
    
    # Retrieve relevant memories
    relevant_memories = await memory_manager.retrieve_memories(
        "user preferences communication", 
        limit=3
    )
    
    # Enhance context with memories
    context_enhancement = "\\n".join([
        f"Relevant memory: {mem.content}" 
        for mem in relevant_memories
    ])
    
    enhanced_input = f"{user_input}\\n\\nContext: {context_enhancement}"
    
    result = await orchestrator.process_query(enhanced_input)
    print(f"Enhanced response: {result.response}")
    
    await orchestrator.close()

asyncio.run(memory_enhanced_app())
```

### 3. Tool-Enabled Application

```python
import asyncio
from sovereign import ModelOrchestrator, Config
from sovereign.tool_execution_engine import ToolCall, SandboxLevel

async def tool_enabled_app():
    config = Config()
    orchestrator = ModelOrchestrator(config)
    await orchestrator.initialize()
    
    # Get available tools
    tools = await orchestrator.tool_execution_engine.get_available_tools()
    print("Available tools:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")
    
    # Process query that might use tools
    result = await orchestrator.process_query(
        "Search for the latest news about artificial intelligence"
    )
    
    print(f"Response: {result.response}")
    
    if result.telemetry.get('tools_used'):
        print(f"Tools used: {result.telemetry['tools_used']}")
    
    await orchestrator.close()

asyncio.run(tool_enabled_app())
```

### 4. Performance Monitoring Application

```python
import asyncio
import time
from sovereign import ModelOrchestrator, Config
from sovereign.performance_monitor import PerformanceMonitor

async def performance_monitoring_app():
    config = Config()
    orchestrator = ModelOrchestrator(config)
    await orchestrator.initialize()
    
    monitor = orchestrator.performance_monitor
    
    # Process several queries while monitoring
    queries = [
        "Hello, how are you?",
        "Explain quantum computing",
        "What's the weather like?",
        "Write a Python function to sort a list",
        "Analyze the pros and cons of renewable energy"
    ]
    
    start_time = time.time()
    
    for query in queries:
        result = await orchestrator.process_query(query)
        print(f"Query: {query[:30]}... -> {result.model_used.value}")
        
        # Get current metrics
        metrics = monitor.get_current_metrics()
        print(f"  CPU: {metrics.cpu_usage:.1f}%, Memory: {metrics.memory_usage:.1f}%")
    
    # Get performance report
    end_time = time.time()
    
    # Wait a moment for metrics to settle
    await asyncio.sleep(1)
    
    from datetime import datetime, timedelta
    report_start = datetime.now() - timedelta(seconds=end_time - start_time + 10)
    report = monitor.get_performance_report(report_start)
    
    print("\\nPerformance Report:")
    print(f"Total queries processed: {len(queries)}")
    print(f"Average response time: {report.get('avg_response_time', 0):.3f}s")
    print(f"Peak CPU usage: {report.get('peak_cpu_usage', 0):.1f}%")
    print(f"Peak memory usage: {report.get('peak_memory_usage', 0):.1f}%")
    
    await orchestrator.close()

asyncio.run(performance_monitoring_app())
```

## Error Handling

### Common Exceptions

#### `ValidationError`
Raised when input parameters are invalid.

```python
try:
    result = await orchestrator.process_query("")  # Empty query
except ValidationError as e:
    print(f"Invalid input: {e}")
```

#### `ProcessingError`
Raised when query processing fails.

```python
try:
    result = await orchestrator.process_query("Complex query")
except ProcessingError as e:
    print(f"Processing failed: {e}")
    print(f"Error details: {e.details}")
```

#### `TimeoutError`
Raised when operations exceed time limits.

```python
try:
    result = await orchestrator.process_query("Very complex analysis", timeout=5)
except TimeoutError as e:
    print(f"Operation timed out: {e}")
```

#### `ResourceError`
Raised when system resources are insufficient.

```python
try:
    result = await orchestrator.process_query("Resource intensive task")
except ResourceError as e:
    print(f"Insufficient resources: {e}")
    print(f"Required: {e.required_resources}")
    print(f"Available: {e.available_resources}")
```

### Best Practices

1. **Always use async/await** for API calls
2. **Handle exceptions gracefully** with appropriate fallbacks
3. **Initialize systems properly** before use
4. **Close resources** when done to prevent memory leaks
5. **Use context managers** when available
6. **Monitor performance** for production applications
7. **Validate inputs** before processing
8. **Log errors** for debugging and monitoring

### Example Error Handling Pattern

```python
import asyncio
import logging
from sovereign import ModelOrchestrator, Config
from sovereign.exceptions import ValidationError, ProcessingError, TimeoutError

async def robust_query_processing(query: str) -> Optional[str]:
    """Robust query processing with comprehensive error handling"""
    
    orchestrator = None
    try:
        # Initialize with timeout
        config = Config()
        orchestrator = ModelOrchestrator(config)
        
        # Set reasonable timeout
        await asyncio.wait_for(orchestrator.initialize(), timeout=30)
        
        # Validate input
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")
        
        # Process with timeout
        result = await asyncio.wait_for(
            orchestrator.process_query(query.strip()),
            timeout=60
        )
        
        return result.response
        
    except ValidationError as e:
        logging.warning(f"Input validation failed: {e}")
        return "Please provide a valid query."
        
    except TimeoutError as e:
        logging.error(f"Operation timed out: {e}")
        return "Sorry, the request is taking too long. Please try a simpler query."
        
    except ProcessingError as e:
        logging.error(f"Processing error: {e}")
        return "I encountered an error processing your request. Please try again."
        
    except Exception as e:
        logging.critical(f"Unexpected error: {e}")
        return "An unexpected error occurred. Please contact support."
        
    finally:
        # Always clean up
        if orchestrator:
            try:
                await orchestrator.close()
            except Exception as e:
                logging.error(f"Error during cleanup: {e}")

# Usage
async def main():
    response = await robust_query_processing("Hello, world!")
    print(response)

asyncio.run(main())
```

This API reference provides comprehensive documentation for integrating with and using the Sovereign AI system effectively and reliably. 