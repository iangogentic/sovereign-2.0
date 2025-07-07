# Sovereign AI System Architecture

## Overview

The Sovereign AI Agent is a sophisticated, privacy-first AI assistant designed for local execution with advanced features including dual-model architecture, memory management, performance optimization, and comprehensive tool integration.

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Sovereign AI Agent System                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                              Interface Layer                                │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────────┤
│   CLI Interface │   GUI Interface │ Voice Interface │  External API Gateway   │
│     (cli.py)    │    (gui.py)     │ (voice_interface│                         │
│                 │                 │     .py)        │                         │
└─────────────────┴─────────────────┴─────────────────┴─────────────────────────┘
                                     │
                            ┌────────┴────────┐
                            │   Core Shell    │ ← Main Entry Point & Service
                            │ & ServiceManager│   Orchestration (lazy-loaded)
                            └────────┬────────┘
                                     │
                            ┌────────┴────────┐
                            │   Orchestrator  │ ← Legacy Coordination Hub
                            │ (orchestrator.py│
                            └────────┬────────┘
                                     │
┌─────────────────────────────────────┴─────────────────────────────────────────┐
│                              Core Models Layer                               │
├─────────────────────────────────┬─────────────────────────────────────────────┤
│         Talker Model            │             Thinker Model                   │
│     (talker_model.py)          │          (thinker_model.py)                │
│   • Fast responses             │     • Complex reasoning                     │
│   • Simple queries             │     • Multi-step analysis                   │
│   • Real-time interaction      │     • Tool coordination                     │
└─────────────────────────────────┴─────────────────────────────────────────────┘
                                     │
┌─────────────────────────────────────┴─────────────────────────────────────────┐
│                          Service Management Layer                            │
│                              (Lazy-Loaded)                                   │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────────┤
│ ModelService    │ MemoryService   │ ToolService     │ ScreenContextService    │
│(model_service   │(memory_service  │(tool_service    │(screen_context_service  │
│    .py)         │    .py)         │    .py)         │      .py)               │
│• Lazy loading   │• Memory mgmt    │• Tool discovery │• Screen capture (lazy)  │
│• Model mgmt     │  coordination   │  & execution    │• OCR processing (lazy)  │
│• Response gen   │• Search wrapper │• Result proc.   │• Privacy filtering      │
├─────────────────┼─────────────────┼─────────────────┼─────────────────────────┤
│ VoiceService    │                 │                 │                         │
│(voice_service   │                 │                 │                         │
│    .py)         │                 │                 │                         │
│• Speech-to-text │                 │                 │                         │
│  (lazy)         │                 │                 │                         │
│• Text-to-speech │                 │                 │                         │
│  (lazy)         │                 │                 │                         │
│• Mic processing │                 │                 │                         │
│  (lazy)         │                 │                 │                         │
└─────────────────┴─────────────────┴─────────────────┴─────────────────────────┘
                                     │
┌─────────────────────────────────────┴─────────────────────────────────────────┐
│                            Memory Management Layer                           │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────────┤
│ Memory Manager  │ Vector Search   │ Embedding       │  Context Window         │
│(memory_manager  │    Engine       │   Service       │     Manager             │
│    .py)         │(vector_search_  │(embedding_      │(context_window_         │
│                 │  engine.py)     │ service.py)     │   manager.py)           │
├─────────────────┼─────────────────┼─────────────────┼─────────────────────────┤
│ Memory Schema   │ Memory Pruner   │ Memory Import/  │ Privacy Manager         │
│(memory_schema   │(memory_pruner   │    Export       │(privacy_manager.py)     │
│    .py)         │    .py)         │                 │                         │
└─────────────────┴─────────────────┴─────────────────┴─────────────────────────┘
                                     │
┌─────────────────────────────────────┴─────────────────────────────────────────┐
│                        Performance & Monitoring Layer                        │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────────┤
│ Performance     │ Memory Leak     │ Automated       │                         │
│   Monitor       │   Detector      │ Performance     │                         │
│(performance_    │(memory_leak_    │   Testing       │                         │
│ monitor.py)     │ detector.py)    │(automated_      │                         │
│                 │                 │performance_     │                         │
│                 │                 │testing.py)      │                         │
└─────────────────┴─────────────────┴─────────────────┴─────────────────────────┘
                                     │
┌─────────────────────────────────────┴─────────────────────────────────────────┐
│                           Tool Framework Layer                               │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────────┤
│ Tool Discovery  │ Tool Execution  │ Tool Results    │ Tool Integration        │
│    Engine       │     Engine      │   Processor     │    Framework            │
│(tool_discovery_ │(tool_execution_ │(tool_result_    │(tool_integration_       │
│  engine.py)     │  engine.py)     │ processor.py)   │ framework.py)           │
├─────────────────┼─────────────────┼─────────────────┼─────────────────────────┤
│ Tool Definition │ Extensibility   │   Core Tools    │   Example Tools         │
│    Schema       │    Manager      │  (core_tools    │  (example_tools         │
│(tool_definition_│(tool_            │     .py)        │      .py)               │
│ schema.py)      │extensibility_   │                 │                         │
│                 │manager.py)      │                 │                         │
└─────────────────┴─────────────────┴─────────────────┴─────────────────────────┘
                                     │
┌─────────────────────────────────────┴─────────────────────────────────────────┐
│                        Screen Context & Privacy Layer                        │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────────┤
│ Screen Context  │ Screen Context  │ Consent         │                         │
│    Manager      │  Integration    │  Manager        │                         │
│(screen_context_ │(screen_context_ │(consent_        │                         │
│ manager.py)     │integration.py)  │manager.py)      │                         │
└─────────────────┴─────────────────┴─────────────────┴─────────────────────────┘
                                     │
┌─────────────────────────────────────┴─────────────────────────────────────────┐
│                         External Integration Layer                           │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────────┤
│ External Model  │ Ollama Client   │    Hardware     │     Config              │
│   Connector     │ (ollama_client  │   Detector      │   (config.py)           │
│(external_model_ │     .py)        │  (hardware.py)  │                         │
│ connector.py)   │                 │                 │                         │
└─────────────────┴─────────────────┴─────────────────┴─────────────────────────┘
```

## Core System Components

### 1. Interface Layer

#### 1.1 CLI Interface (`cli.py`)
**Interface**: Command-line interface for system interaction
**Key Methods**:
- `main()` - Entry point for CLI commands
- `interactive_mode()` - Interactive conversation mode
- `process_command()` - Single command processing

**Data Exchange Format**:
```python
class CLICommand:
    command: str
    args: List[str]
    options: Dict[str, Any]
```

#### 1.2 GUI Interface (`gui.py`) 
**Interface**: Desktop graphical user interface
**Key Methods**:
- `run_gui()` - Launch GUI application
- `SovereignGUI.__init__()` - Initialize GUI components
- `update_display()` - Refresh GUI elements

**Data Exchange Format**:
```python
class GUIEvent:
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime
```

#### 1.3 Voice Interface (`voice_interface.py`)
**Interface**: Speech-to-text and text-to-speech functionality
**Key Methods**:
- `start_listening()` - Begin voice input capture
- `synthesize_speech()` - Convert text to speech
- `get_audio_devices()` - List available audio hardware

**Data Exchange Format**:
```python
class VoiceInput:
    text: str
    confidence: float
    audio_data: bytes
    duration: float
```

### 2. Service Management Layer (Lazy-Loaded)

#### 2.1 ServiceManager (`core/service_manager.py`)
**Interface**: Central service coordination and lazy loading
**Key Methods**:
- `get_service(service_name: str) -> Service`
- `initialize() -> None`
- `list_services() -> Dict[str, ServiceStatus]`
- `model() -> ModelService` (convenience accessor)
- `memory() -> MemoryService` (convenience accessor)
- `tool() -> ToolService` (convenience accessor)
- `screen() -> ScreenContextService` (convenience accessor)
- `voice() -> VoiceService` (convenience accessor)

**Lazy Loading Strategy**:
- Services instantiated only on first access
- Heavy imports deferred until actual service use
- Enables <1s cold startup performance

#### 2.2 ModelService (`services/model_service.py`)
**Interface**: Lazy-loaded model management and response generation
**Key Methods**:
- `query(prompt: str, model_type: str) -> ModelResponse`
- `get_available_models() -> List[str]`
- `get_health_check() -> Dict[str, Any]`

**Lazy Loading**:
- No heavy model imports at module level
- Models loaded on first query() call
- Subsequent calls reuse loaded components

#### 2.3 MemoryService (`services/memory_service.py`)
**Interface**: Memory coordination and search wrapper
**Key Methods**:
- `store(content: str, metadata: Dict) -> str`
- `search(query: str, limit: int) -> List[Memory]`
- `retrieve(memory_id: str) -> Memory`

#### 2.4 ToolService (`services/tool_service.py`)
**Interface**: Tool discovery, execution, and result processing
**Key Methods**:
- `discover_tools() -> List[Tool]`
- `execute(tool_name: str, params: Dict) -> ToolResult`
- `get_available_tools() -> List[str]`

#### 2.5 ScreenContextService (`services/screen_context_service.py`)
**Interface**: Lazy-loaded screen capture and OCR processing
**Key Methods**:
- `capture_and_analyse() -> ScreenCaptureResult`
- `get_health_check() -> Dict[str, Any]`

**Lazy Loading**:
- No cv2, pytesseract, torch imports at module level
- OCR/CV libraries loaded only on first capture_and_analyse() call
- Massive performance improvement: 67,000x faster subsequent calls

#### 2.6 VoiceService (`services/voice_service.py`)
**Interface**: Lazy-loaded voice input/output processing
**Key Methods**:
- `listen(timeout: float) -> VoiceResult`
- `speak(text: str) -> VoiceResult`
- `get_health_check() -> Dict[str, Any]`

**Lazy Loading**:
- No sounddevice, speech_recognition imports at module level
- Voice libraries loaded only on first listen() or speak() call
- Graceful degradation when voice libraries unavailable

### 3. Core Models Layer

#### 3.1 Model Orchestrator (`orchestrator.py`)
**Interface**: Central coordination hub for model selection and handoff
**Key Methods**:
- `process_query(user_input: str, context: QueryContext) -> OrchestrationResult`
- `determine_complexity(query: str) -> QueryComplexity`
- `handle_model_handoff() -> str`

**Data Exchange Format**:
```python
@dataclass
class QueryContext:
    user_input: str
    timestamp: datetime
    session_id: str
    previous_queries: List[str]
    conversation_history: List[Dict[str, Any]]
    screen_context: Optional[Dict[str, Any]]
    voice_context: Optional[Dict[str, Any]]
    user_preferences: Optional[Dict[str, Any]]

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

#### 3.2 Talker Model (`talker_model.py`)
**Interface**: Fast-response model for simple queries and real-time interaction
**Key Methods**:
- `generate_response(prompt: str) -> str`
- `initialize() -> None`
- `is_ready() -> bool`

**Data Exchange Format**:
```python
@dataclass
class TalkerRequest:
    prompt: str
    max_tokens: int
    temperature: float
    context: Optional[Dict]

@dataclass
class TalkerResponse:
    text: str
    tokens_used: int
    processing_time: float
    model_info: Dict[str, Any]
```

#### 3.3 Thinker Model (`thinker_model.py`)
**Interface**: Complex reasoning model for multi-step analysis and tool coordination
**Key Methods**:
- `process_complex_query(query: str, task_type: TaskType) -> str`
- `coordinate_tools(tools: List[str]) -> ToolResult`
- `analyze_context(context: Dict) -> AnalysisResult`

**Data Exchange Format**:
```python
@dataclass
class ThinkerRequest:
    query: str
    task_type: TaskType
    context: Dict[str, Any]
    available_tools: List[str]
    reasoning_depth: int

@dataclass
class ThinkerResponse:
    response: str
    reasoning_steps: List[str]
    tools_used: List[str]
    confidence: float
    processing_time: float
```

### 4. Memory Management Layer

#### 4.1 Memory Manager (`memory_manager.py`)
**Interface**: Central memory storage and retrieval system
**Key Methods**:
- `store_memory(memory: Memory) -> str`
- `retrieve_memories(query: str, limit: int) -> List[Memory]`
- `update_memory(memory_id: str, updates: Dict) -> bool`
- `delete_memory(memory_id: str) -> bool`

**Data Exchange Format**:
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
```

#### 4.2 Vector Search Engine (`vector_search_engine.py`)
**Interface**: Semantic search and similarity matching
**Key Methods**:
- `add_document(doc: Document) -> str`
- `search(query: str, k: int) -> List[SearchResult]`
- `update_index() -> None`
- `get_similar_documents(doc_id: str) -> List[Document]`

**Data Exchange Format**:
```python
@dataclass
class SearchResult:
    document: Document
    score: float
    highlights: List[str]
    metadata: Dict[str, Any]
```

#### 4.3 Embedding Service (`embedding_service.py`)
**Interface**: Local embedding generation for text
**Key Methods**:
- `generate_embedding(text: str) -> EmbeddingResponse`
- `generate_embeddings(requests: List[EmbeddingRequest]) -> List[EmbeddingResponse]`
- `get_model_info(model_type: EmbeddingModelType) -> ModelInfo`

**Data Exchange Format**:
```python
@dataclass
class EmbeddingRequest:
    text: str
    model_type: Optional[EmbeddingModelType]
    model_name: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class EmbeddingResponse:
    request_id: str
    embedding: Optional[np.ndarray]
    model_used: str
    processing_time: float
    tokens_processed: int
    success: bool
    error_message: Optional[str]
```

#### 4.4 Context Window Manager (`context_window_manager.py`)
**Interface**: Context size management and optimization
**Key Methods**:
- `optimize_context(context: str, max_tokens: int) -> OptimizedContext`
- `estimate_tokens(text: str) -> int`
- `truncate_context(context: str, target_size: int) -> str`

#### 4.5 Privacy Manager (`privacy_manager.py`)
**Interface**: Privacy protection and data anonymization
**Key Methods**:
- `anonymize_data(data: Dict) -> Dict`
- `check_privacy_compliance(operation: str) -> bool`
- `encrypt_sensitive_data(data: str) -> str`

### 5. Performance & Monitoring Layer

#### 5.1 Performance Monitor (`performance_monitor.py`)
**Interface**: Real-time system performance monitoring
**Key Methods**:
- `start_monitoring() -> None`
- `stop_monitoring() -> None`
- `get_current_metrics() -> PerformanceMetrics`
- `get_performance_report() -> Dict[str, Any]`

**Data Exchange Format**:
```python
@dataclass
class PerformanceMetrics:
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    disk_io: Dict[str, float]
    network_io: Dict[str, float]
    response_times: List[float]
    error_rates: Dict[str, float]
    timestamp: datetime
```

#### 5.2 Memory Leak Detector (`memory_leak_detector.py`)
**Interface**: Advanced memory leak detection and analysis
**Key Methods**:
- `start_monitoring() -> None`
- `detect_leaks() -> List[LeakDetectionResult]`
- `get_leak_summary() -> Dict[str, Any]`
- `force_cleanup() -> None`

#### 5.3 Automated Performance Testing (`automated_performance_testing.py`)
**Interface**: Automated optimization and crash recovery
**Key Methods**:
- `start_optimization_monitoring() -> None`
- `apply_optimization(optimization: OptimizationResult) -> bool`
- `detect_crashes() -> List[CrashEvent]`
- `recover_from_crash(crash: CrashEvent) -> RecoveryAction`

### 6. Tool Framework Layer

#### 6.1 Tool Discovery Engine (`tool_discovery_engine.py`)
**Interface**: Dynamic tool detection and registration
**Key Methods**:
- `discover_tools() -> List[Tool]`
- `register_tool(tool: Tool) -> bool`
- `get_available_tools() -> List[Tool]`

#### 6.2 Tool Execution Engine (`tool_execution_engine.py`)
**Interface**: Safe tool execution with sandboxing
**Key Methods**:
- `execute_tool(tool_call: ToolCall) -> ToolResult`
- `validate_tool_call(tool_call: ToolCall) -> bool`
- `get_execution_status(execution_id: str) -> ExecutionStatus`

**Data Exchange Format**:
```python
@dataclass
class ToolCall:
    tool_name: str
    parameters: Dict[str, Any]
    execution_id: str
    timeout: int
    sandbox_level: SandboxLevel

@dataclass
class ToolResult:
    execution_id: str
    success: bool
    result: Any
    error_message: Optional[str]
    execution_time: float
    resources_used: Dict[str, Any]
```

#### 6.3 Tool Result Processor (`tool_result_processor.py`)
**Interface**: Tool result processing and integration
**Key Methods**:
- `process_result(result: ToolResult) -> ProcessedResult`
- `format_for_model(result: ToolResult) -> str`
- `aggregate_results(results: List[ToolResult]) -> AggregatedResult`

### 7. Screen Context & Privacy Layer

#### 7.1 Screen Context Manager (`screen_context_manager.py`)
**Interface**: Screen capture and context extraction
**Key Methods**:
- `capture_screen() -> ScreenCapture`
- `extract_context(capture: ScreenCapture) -> ContextData`
- `get_active_applications() -> List[Application]`

**Data Exchange Format**:
```python
@dataclass
class ScreenCapture:
    image_data: bytes
    timestamp: datetime
    screen_resolution: Tuple[int, int]
    active_window: str
    metadata: Dict[str, Any]

@dataclass
class ContextData:
    text_content: str
    ui_elements: List[UIElement]
    applications: List[str]
    confidence_score: float
```

#### 7.2 Consent Manager (`consent_manager.py`)
**Interface**: User consent and permission management
**Key Methods**:
- `request_consent(operation: str) -> bool`
- `check_permission(operation: str) -> PermissionLevel`
- `update_consent_preferences(prefs: Dict) -> None`

### 8. External Integration Layer

#### 8.1 External Model Connector (`external_model_connector.py`)
**Interface**: Integration with external AI services
**Key Methods**:
- `connect_to_service(service: str, config: Dict) -> bool`
- `send_request(request: ExternalRequest) -> ExternalResponse`
- `get_service_status(service: str) -> ServiceStatus`

#### 8.2 Ollama Client (`ollama_client.py`)
**Interface**: Local Ollama model integration
**Key Methods**:
- `list_models() -> List[str]`
- `generate(model: str, prompt: str) -> str`
- `pull_model(model: str) -> bool`

## Data Flow Architecture

### Primary Data Flows

1. **User Query Processing Flow (Service-Based)**:
   ```
   User Input → Interface Layer → Core Shell → ServiceManager → Service (lazy-loaded) → Response Generation → Interface Layer → User
   ```

2. **Legacy User Query Processing Flow**:
   ```
   User Input → Interface Layer → Orchestrator → Model Selection → Response Generation → Interface Layer → User
   ```

3. **Service Lazy Loading Flow**:
   ```
   Service Request → ServiceManager.get_service() → First Access Check → Heavy Import Loading → Service Instantiation → Cached for Reuse
   ```

4. **Memory Storage Flow**:
   ```
   Conversation → Memory Manager → Embedding Service → Vector Search Engine → Persistent Storage
   ```

5. **Tool Execution Flow**:
   ```
   Tool Request → Tool Discovery → Tool Execution → Result Processing → Integration with Response
   ```

6. **Performance Monitoring Flow**:
   ```
   System Metrics → Performance Monitor → Memory Leak Detector → Automated Optimization → Recovery Actions
   ```

7. **Screen Context Flow (Lazy-Loaded)**:
   ```
   Screen Request → ScreenContextService.capture_and_analyse() → [First Call: Load OCR/CV] → Screen Capture → Context Extraction → Privacy Filtering → Context Integration → Query Enhancement
   ```

## Integration Points

### Inter-Component Communication

1. **Orchestrator ↔ Models**:
   - Protocol: Direct Python method calls
   - Data Format: QueryContext/Response objects
   - Error Handling: Exception propagation with graceful fallbacks

2. **Memory System ↔ Vector Search**:
   - Protocol: Embedding-based similarity search
   - Data Format: Vector arrays and metadata
   - Consistency: Eventual consistency with immediate writes

3. **Tool Framework ↔ Models**:
   - Protocol: JSON-based tool calls
   - Data Format: Structured tool definitions and results
   - Security: Sandboxed execution with permission checks

4. **Performance Layer ↔ All Components**:
   - Protocol: Observer pattern with metrics collection
   - Data Format: Standardized performance metrics
   - Frequency: Real-time monitoring with configurable intervals

### External Integrations

1. **Ollama Integration**:
   - Protocol: HTTP API calls
   - Data Format: JSON payloads
   - Error Handling: Connection retry with exponential backoff

2. **Hardware Integration**:
   - Protocol: System API calls (psutil, CUDA)
   - Data Format: Hardware metrics and capabilities
   - Monitoring: Continuous hardware health monitoring

## Configuration Management

### Configuration Architecture
- **Primary Config**: `config.py` - Central configuration management
- **Environment Variables**: Runtime configuration overrides
- **User Preferences**: Persistent user-specific settings
- **Model Configs**: Model-specific parameters and capabilities

### Configuration Data Flow
```
Environment Variables → Config Loader → Component Initialization → Runtime Updates
```

## Security & Privacy Architecture

### Security Layers
1. **Input Validation**: All user inputs sanitized and validated
2. **Tool Sandboxing**: Isolated execution environment for tools
3. **Memory Encryption**: Sensitive data encrypted at rest
4. **Network Security**: Secure communication protocols
5. **Access Control**: Role-based permission system

### Privacy Protection
1. **Data Anonymization**: Automatic PII detection and masking
2. **Consent Management**: Granular permission controls
3. **Local Processing**: All AI processing happens locally
4. **Audit Logging**: Comprehensive privacy audit trails

## Error Handling & Recovery

### Error Handling Strategy
- **Graceful Degradation**: System continues operating with reduced functionality
- **Circuit Breakers**: Prevent cascade failures in external integrations
- **Retry Logic**: Exponential backoff for transient failures
- **Fallback Mechanisms**: Alternative processing paths for critical functions

### Recovery Mechanisms
- **Automatic Restart**: Failed components restart automatically
- **State Restoration**: Critical state preserved and restored
- **Health Checks**: Continuous monitoring with automatic recovery
- **Emergency Protocols**: Safe shutdown and data preservation procedures

## Performance Characteristics

### Scalability Targets
- **Concurrent Users**: 1-10 users (single-machine deployment)
- **Memory Usage**: < 8GB RAM under normal operation
- **Response Time**: < 2 seconds for simple queries, < 30 seconds for complex analysis
- **Storage**: Efficient memory management with configurable retention

### Resource Management
- **CPU Optimization**: Multi-threading for I/O operations
- **Memory Management**: Intelligent caching with leak detection
- **GPU Utilization**: Automatic GPU detection and usage
- **Disk I/O**: Efficient database operations with indexing

## Testing Strategy

### Testing Levels
1. **Unit Tests**: Individual component functionality
2. **Integration Tests**: Inter-component communication
3. **End-to-End Tests**: Complete user workflows
4. **Performance Tests**: Load and stress testing
5. **Security Tests**: Vulnerability and penetration testing

### Testing Data Flows
- **Test Data Management**: Synthetic data generation for testing
- **Mock Services**: External service mocking for isolated testing
- **Performance Benchmarks**: Standardized performance test suites

---

This architecture document serves as the foundation for system integration and provides clear interfaces for all components. Each interface is designed for maintainability, testability, and extensibility while ensuring robust error handling and performance optimization. 