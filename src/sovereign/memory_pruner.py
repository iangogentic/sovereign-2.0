"""
Memory Pruning System for Sovereign AI Long-Term Memory (RAG) System

This module provides intelligent memory pruning capabilities to maintain optimal
performance and storage efficiency by removing outdated, irrelevant, or redundant
memory chunks while preserving valuable information.

Key Features:
- Multiple pruning strategies (time, usage, relevance, size-based)
- Configurable retention policies with database integration
- Automated and manual pruning operations
- Safe deletion with rollback capabilities
- Orphan cleanup and duplicate detection
- Performance monitoring and statistics
- Scheduled cleanup jobs
- User preference integration
"""

import logging
import time
import threading
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

from .memory_schema import MemorySchema, EntityType, AccessAction
from .config import Config

logger = logging.getLogger(__name__)


class PruningStrategy(Enum):
    """Memory pruning strategies"""
    TIME_BASED = "time_based"           # Remove old data based on timestamps
    USAGE_BASED = "usage_based"         # Remove unused/rarely accessed data
    RELEVANCE_BASED = "relevance_based" # Remove low-relevance content
    SIZE_BASED = "size_based"          # Maintain storage within limits
    DUPLICATE_REMOVAL = "duplicate_removal"  # Remove redundant content
    ORPHAN_CLEANUP = "orphan_cleanup"   # Remove dangling references
    COMPREHENSIVE = "comprehensive"     # Combined approach


class PruningPriority(Enum):
    """Priority levels for pruning operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RetentionPolicyType(Enum):
    """Types of retention policies"""
    CONVERSATION = "conversation"
    MESSAGE = "message"
    DOCUMENT = "document"
    CHUNK = "chunk"
    EMBEDDING = "embedding"
    ACCESS_LOG = "access_log"


@dataclass
class RetentionPolicy:
    """Retention policy configuration"""
    id: Optional[int] = None
    entity_type: RetentionPolicyType = RetentionPolicyType.CONVERSATION
    policy_name: str = "default"
    retention_days: int = 30
    auto_delete: bool = False
    max_items: Optional[int] = None
    min_relevance_score: Optional[float] = None
    preserve_favorites: bool = True
    created_at: Optional[datetime] = None
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'entity_type': self.entity_type.value,
            'policy_name': self.policy_name,
            'retention_days': self.retention_days,
            'auto_delete': self.auto_delete,
            'max_items': self.max_items,
            'min_relevance_score': self.min_relevance_score,
            'preserve_favorites': self.preserve_favorites,
            'is_active': self.is_active
        }


@dataclass
class PruningResult:
    """Result of a pruning operation"""
    strategy: PruningStrategy
    items_analyzed: int = 0
    items_deleted: int = 0
    storage_freed_mb: float = 0.0
    execution_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    deleted_items: Dict[str, List[int]] = field(default_factory=dict)
    success: bool = True
    
    def add_deleted_item(self, entity_type: str, item_id: int):
        """Add a deleted item to the result"""
        if entity_type not in self.deleted_items:
            self.deleted_items[entity_type] = []
        self.deleted_items[entity_type].append(item_id)
        self.items_deleted += 1


@dataclass
class PruningStats:
    """Statistics for pruning operations"""
    total_pruning_runs: int = 0
    total_items_deleted: int = 0
    total_storage_freed_mb: float = 0.0
    last_pruning_run: Optional[datetime] = None
    average_execution_time: float = 0.0
    strategy_usage: Dict[str, int] = field(default_factory=dict)
    error_count: int = 0
    
    def update_from_result(self, result: PruningResult):
        """Update stats from a pruning result"""
        self.total_pruning_runs += 1
        self.total_items_deleted += result.items_deleted
        self.total_storage_freed_mb += result.storage_freed_mb
        self.last_pruning_run = datetime.now()
        
        # Update average execution time
        total_time = self.average_execution_time * (self.total_pruning_runs - 1) + result.execution_time
        self.average_execution_time = total_time / self.total_pruning_runs
        
        # Update strategy usage
        strategy_name = result.strategy.value
        self.strategy_usage[strategy_name] = self.strategy_usage.get(strategy_name, 0) + 1
        
        if not result.success:
            self.error_count += 1


class MemoryPruner:
    """
    Main memory pruning engine.
    
    Provides intelligent memory pruning with multiple strategies,
    configurable policies, and comprehensive monitoring.
    """
    
    def __init__(self, config: Config, db_path: str):
        """
        Initialize the Memory Pruner.
        
        Args:
            config: Configuration object
            db_path: Path to the SQLite database
        """
        self.config = config
        self.db_path = db_path
        self.schema = MemorySchema(db_path)
        
        # Threading for background operations
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._lock = threading.RLock()
        
        # Performance tracking
        self.stats = PruningStats()
        
        # Cache for retention policies
        self._retention_policies: Dict[str, RetentionPolicy] = {}
        self._policies_loaded = False
        
        # Safety settings
        self.max_batch_size = 1000  # Maximum items to delete in one operation
        self.dry_run_mode = False   # Set to True for testing
        
        logger.info("Memory Pruner initialized")
    
    def close(self):
        """Clean up resources"""
        if self._executor:
            self._executor.shutdown(wait=False)
        self.schema.close()
        logger.info("Memory Pruner closed")
    
    # Retention Policy Management
    
    def create_retention_policy(self, policy: RetentionPolicy) -> int:
        """
        Create a new retention policy.
        
        Args:
            policy: Retention policy configuration
            
        Returns:
            Policy ID
        """
        try:
            with self._lock:
                conn = self.schema.get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO retention_policies (
                        entity_type, policy_name, retention_days, auto_delete, is_active
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    policy.entity_type.value,
                    policy.policy_name,
                    policy.retention_days,
                    policy.auto_delete,
                    policy.is_active
                ))
                
                policy_id = cursor.lastrowid
                conn.commit()
                
                # Update cache
                self._retention_policies[f"{policy.entity_type.value}:{policy.policy_name}"] = policy
                
                logger.info(f"Created retention policy {policy_id}: {policy.policy_name}")
                return policy_id
                
        except Exception as e:
            logger.error(f"Failed to create retention policy: {e}")
            raise
    
    def get_retention_policy(self, entity_type: RetentionPolicyType, policy_name: str = "default") -> Optional[RetentionPolicy]:
        """Get a retention policy by type and name"""
        try:
            self._load_retention_policies()
            
            key = f"{entity_type.value}:{policy_name}"
            return self._retention_policies.get(key)
            
        except Exception as e:
            logger.error(f"Failed to get retention policy: {e}")
            return None
    
    def _load_retention_policies(self):
        """Load retention policies from database"""
        if self._policies_loaded:
            return
        
        try:
            conn = self.schema.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, entity_type, policy_name, retention_days, auto_delete, 
                       created_at, is_active
                FROM retention_policies
                WHERE is_active = 1
            """)
            
            rows = cursor.fetchall()
            
            for row in rows:
                policy = RetentionPolicy(
                    id=row[0],
                    entity_type=RetentionPolicyType(row[1]),
                    policy_name=row[2],
                    retention_days=row[3],
                    auto_delete=bool(row[4]),
                    created_at=datetime.fromisoformat(row[5]) if row[5] else None,
                    is_active=bool(row[6])
                )
                
                key = f"{policy.entity_type.value}:{policy.policy_name}"
                self._retention_policies[key] = policy
            
            self._policies_loaded = True
            logger.info(f"Loaded {len(rows)} retention policies")
            
        except Exception as e:
            logger.error(f"Failed to load retention policies: {e}")
    
    # Core Pruning Methods
    
    async def prune_memory(self, 
                          strategy: PruningStrategy = PruningStrategy.COMPREHENSIVE,
                          dry_run: bool = False,
                          batch_size: int = None) -> PruningResult:
        """
        Execute memory pruning with the specified strategy.
        
        Args:
            strategy: Pruning strategy to use
            dry_run: If True, only simulate deletion
            batch_size: Maximum items to process in one batch
            
        Returns:
            PruningResult with details of the operation
        """
        start_time = time.time()
        
        result = PruningResult(strategy=strategy)
        original_dry_run = self.dry_run_mode
        
        try:
            # Set dry run mode
            self.dry_run_mode = dry_run
            
            if batch_size:
                self.max_batch_size = min(batch_size, self.max_batch_size)
            
            logger.info(f"Starting {strategy.value} pruning (dry_run={dry_run})")
            
            # Execute strategy-specific pruning
            if strategy == PruningStrategy.TIME_BASED:
                await self._prune_time_based(result)
            elif strategy == PruningStrategy.USAGE_BASED:
                await self._prune_usage_based(result)
            elif strategy == PruningStrategy.RELEVANCE_BASED:
                await self._prune_relevance_based(result)
            elif strategy == PruningStrategy.SIZE_BASED:
                await self._prune_size_based(result)
            elif strategy == PruningStrategy.DUPLICATE_REMOVAL:
                await self._prune_duplicates(result)
            elif strategy == PruningStrategy.ORPHAN_CLEANUP:
                await self._prune_orphans(result)
            elif strategy == PruningStrategy.COMPREHENSIVE:
                await self._prune_comprehensive(result)
            
            result.execution_time = time.time() - start_time
            
            # Update statistics
            self.stats.update_from_result(result)
            
            logger.info(f"Pruning completed: {result.items_deleted} items deleted, "
                       f"{result.storage_freed_mb:.2f}MB freed in {result.execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            result.execution_time = time.time() - start_time
            logger.error(f"Pruning failed: {e}")
            return result
        
        finally:
            self.dry_run_mode = original_dry_run
    
    async def _prune_time_based(self, result: PruningResult):
        """Prune old data based on retention policies"""
        self._load_retention_policies()
        
        # Prune conversations
        await self._prune_old_conversations(result)
        
        # Prune messages
        await self._prune_old_messages(result)
        
        # Prune documents
        await self._prune_old_documents(result)
        
        # Prune access logs
        await self._prune_old_access_logs(result)
    
    async def _prune_old_conversations(self, result: PruningResult):
        """Prune old conversations based on retention policy"""
        try:
            policy = self.get_retention_policy(RetentionPolicyType.CONVERSATION)
            if not policy:
                return
            
            cutoff_date = datetime.now() - timedelta(days=policy.retention_days)
            
            conn = self.schema.get_connection()
            cursor = conn.cursor()
            
            # Find old conversations
            cursor.execute("""
                SELECT id, title, started_at, total_messages
                FROM conversations 
                WHERE started_at < ? AND is_active = 1
                ORDER BY started_at ASC
                LIMIT ?
            """, (cutoff_date.isoformat(), self.max_batch_size))
            
            conversations = cursor.fetchall()
            result.items_analyzed += len(conversations)
            
            if not conversations:
                return
            
            logger.info(f"Found {len(conversations)} old conversations to prune")
            
            for conv in conversations:
                conv_id = conv[0]
                
                if not self.dry_run_mode:
                    # Delete conversation (cascade will handle messages)
                    cursor.execute("UPDATE conversations SET is_active = 0 WHERE id = ?", (conv_id,))
                
                result.add_deleted_item("conversations", conv_id)
            
            if not self.dry_run_mode:
                conn.commit()
            
        except Exception as e:
            result.errors.append(f"Failed to prune conversations: {e}")
            logger.error(f"Failed to prune conversations: {e}")
    
    async def _prune_old_messages(self, result: PruningResult):
        """Prune old messages based on retention policy"""
        try:
            policy = self.get_retention_policy(RetentionPolicyType.MESSAGE)
            if not policy:
                return
            
            cutoff_date = datetime.now() - timedelta(days=policy.retention_days)
            
            conn = self.schema.get_connection()
            cursor = conn.cursor()
            
            # Find old messages in active conversations
            cursor.execute("""
                SELECT m.id, m.conversation_id, m.created_at
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE m.created_at < ? AND c.is_active = 1
                ORDER BY m.created_at ASC
                LIMIT ?
            """, (cutoff_date.isoformat(), self.max_batch_size))
            
            messages = cursor.fetchall()
            result.items_analyzed += len(messages)
            
            if not messages:
                return
            
            logger.info(f"Found {len(messages)} old messages to prune")
            
            for msg in messages:
                msg_id = msg[0]
                
                if not self.dry_run_mode:
                    # Delete message
                    cursor.execute("DELETE FROM messages WHERE id = ?", (msg_id,))
                
                result.add_deleted_item("messages", msg_id)
            
            if not self.dry_run_mode:
                conn.commit()
            
        except Exception as e:
            result.errors.append(f"Failed to prune messages: {e}")
            logger.error(f"Failed to prune messages: {e}")
    
    async def _prune_old_documents(self, result: PruningResult):
        """Prune old unused documents"""
        try:
            policy = self.get_retention_policy(RetentionPolicyType.DOCUMENT)
            if not policy:
                return
            
            cutoff_date = datetime.now() - timedelta(days=policy.retention_days)
            
            conn = self.schema.get_connection()
            cursor = conn.cursor()
            
            # Find old documents not accessed recently
            cursor.execute("""
                SELECT d.id, d.title, d.created_at
                FROM documents d
                LEFT JOIN access_logs al ON al.entity_type = 'document' 
                    AND al.entity_id = d.id 
                    AND al.created_at > ?
                WHERE d.created_at < ? AND d.is_active = 1 AND al.id IS NULL
                LIMIT ?
            """, (cutoff_date.isoformat(), cutoff_date.isoformat(), self.max_batch_size))
            
            documents = cursor.fetchall()
            result.items_analyzed += len(documents)
            
            if not documents:
                return
            
            logger.info(f"Found {len(documents)} old documents to prune")
            
            for doc in documents:
                doc_id = doc[0]
                
                if not self.dry_run_mode:
                    # Soft delete document (cascade will handle chunks)
                    cursor.execute("UPDATE documents SET is_active = 0 WHERE id = ?", (doc_id,))
                
                result.add_deleted_item("documents", doc_id)
            
            if not self.dry_run_mode:
                conn.commit()
            
        except Exception as e:
            result.errors.append(f"Failed to prune documents: {e}")
            logger.error(f"Failed to prune documents: {e}")
    
    async def _prune_old_access_logs(self, result: PruningResult):
        """Prune old access logs"""
        try:
            policy = self.get_retention_policy(RetentionPolicyType.ACCESS_LOG)
            if not policy:
                # Default to 90 days for access logs
                retention_days = 90
            else:
                retention_days = policy.retention_days
            
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            conn = self.schema.get_connection()
            cursor = conn.cursor()
            
            # Count old access logs
            cursor.execute("""
                SELECT COUNT(*) FROM access_logs WHERE created_at < ?
            """, (cutoff_date.isoformat(),))
            
            count = cursor.fetchone()[0]
            result.items_analyzed += count
            
            if count == 0:
                return
            
            logger.info(f"Found {count} old access logs to prune")
            
            if not self.dry_run_mode:
                # Delete old access logs in batches
                cursor.execute("""
                    DELETE FROM access_logs 
                    WHERE id IN (
                        SELECT id FROM access_logs 
                        WHERE created_at < ? 
                        LIMIT ?
                    )
                """, (cutoff_date.isoformat(), self.max_batch_size))
                
                conn.commit()
            
            result.items_deleted += min(count, self.max_batch_size)
            
        except Exception as e:
            result.errors.append(f"Failed to prune access logs: {e}")
            logger.error(f"Failed to prune access logs: {e}")
    
    async def _prune_usage_based(self, result: PruningResult):
        """Prune unused or rarely accessed data"""
        try:
            conn = self.schema.get_connection()
            cursor = conn.cursor()
            
            # Find documents with no recent access
            thirty_days_ago = datetime.now() - timedelta(days=30)
            
            cursor.execute("""
                SELECT d.id, d.title
                FROM documents d
                LEFT JOIN access_logs al ON al.entity_type = 'document' 
                    AND al.entity_id = d.id 
                    AND al.created_at > ?
                WHERE d.is_active = 1 AND al.id IS NULL
                LIMIT ?
            """, (thirty_days_ago.isoformat(), self.max_batch_size))
            
            unused_docs = cursor.fetchall()
            result.items_analyzed += len(unused_docs)
            
            logger.info(f"Found {len(unused_docs)} unused documents")
            
            for doc in unused_docs:
                doc_id = doc[0]
                
                if not self.dry_run_mode:
                    cursor.execute("UPDATE documents SET is_active = 0 WHERE id = ?", (doc_id,))
                
                result.add_deleted_item("documents", doc_id)
            
            if not self.dry_run_mode:
                conn.commit()
            
        except Exception as e:
            result.errors.append(f"Failed usage-based pruning: {e}")
            logger.error(f"Failed usage-based pruning: {e}")
    
    async def _prune_relevance_based(self, result: PruningResult):
        """Prune low-relevance content based on feedback"""
        try:
            conn = self.schema.get_connection()
            cursor = conn.cursor()
            
            # Find chunks with negative feedback
            cursor.execute("""
                SELECT c.id, c.document_id, COUNT(f.id) as negative_feedback
                FROM chunks c
                LEFT JOIN feedback f ON f.chunk_id = c.id 
                    AND f.feedback_type IN ('irrelevant', 'not_helpful')
                WHERE c.is_active = 1
                GROUP BY c.id
                HAVING negative_feedback > 2
                LIMIT ?
            """, (self.max_batch_size,))
            
            low_relevance_chunks = cursor.fetchall()
            result.items_analyzed += len(low_relevance_chunks)
            
            logger.info(f"Found {len(low_relevance_chunks)} low-relevance chunks")
            
            for chunk in low_relevance_chunks:
                chunk_id = chunk[0]
                
                if not self.dry_run_mode:
                    cursor.execute("UPDATE chunks SET is_active = 0 WHERE id = ?", (chunk_id,))
                
                result.add_deleted_item("chunks", chunk_id)
            
            if not self.dry_run_mode:
                conn.commit()
            
        except Exception as e:
            result.errors.append(f"Failed relevance-based pruning: {e}")
            logger.error(f"Failed relevance-based pruning: {e}")
    
    async def _prune_size_based(self, result: PruningResult):
        """Prune data to maintain storage within limits"""
        try:
            # Get database size
            db_size_mb = self._get_database_size_mb()
            max_size_mb = getattr(self.config, 'max_database_size_mb', 1000)  # Default 1GB
            
            if db_size_mb <= max_size_mb:
                logger.info(f"Database size ({db_size_mb:.2f}MB) within limit ({max_size_mb}MB)")
                return
            
            logger.info(f"Database size ({db_size_mb:.2f}MB) exceeds limit ({max_size_mb}MB)")
            
            # Prune oldest data first
            await self._prune_time_based(result)
            
            # Check size again
            new_db_size_mb = self._get_database_size_mb()
            result.storage_freed_mb = db_size_mb - new_db_size_mb
            
        except Exception as e:
            result.errors.append(f"Failed size-based pruning: {e}")
            logger.error(f"Failed size-based pruning: {e}")
    
    async def _prune_duplicates(self, result: PruningResult):
        """Remove duplicate content"""
        try:
            conn = self.schema.get_connection()
            cursor = conn.cursor()
            
            # Find duplicate documents by content hash
            cursor.execute("""
                SELECT file_hash, COUNT(*) as count
                FROM documents 
                WHERE file_hash IS NOT NULL AND file_hash != '' AND is_active = 1
                GROUP BY file_hash 
                HAVING count > 1
                LIMIT ?
            """, (self.max_batch_size // 2,))
            
            duplicate_hashes = cursor.fetchall()
            
            for hash_row in duplicate_hashes:
                file_hash = hash_row[0]
                
                # Keep the newest, delete the rest
                cursor.execute("""
                    SELECT id, created_at 
                    FROM documents 
                    WHERE file_hash = ? AND is_active = 1 
                    ORDER BY created_at DESC
                """, (file_hash,))
                
                docs = cursor.fetchall()
                result.items_analyzed += len(docs)
                
                # Skip the first (newest) document
                for doc in docs[1:]:
                    doc_id = doc[0]
                    
                    if not self.dry_run_mode:
                        cursor.execute("UPDATE documents SET is_active = 0 WHERE id = ?", (doc_id,))
                    
                    result.add_deleted_item("documents", doc_id)
            
            if not self.dry_run_mode:
                conn.commit()
            
            logger.info(f"Removed {result.items_deleted} duplicate documents")
            
        except Exception as e:
            result.errors.append(f"Failed duplicate removal: {e}")
            logger.error(f"Failed duplicate removal: {e}")
    
    async def _prune_orphans(self, result: PruningResult):
        """Remove orphaned data (embeddings without chunks, etc.)"""
        try:
            conn = self.schema.get_connection()
            cursor = conn.cursor()
            
            # Find orphaned embeddings
            cursor.execute("""
                SELECT e.chunk_id
                FROM embeddings e
                LEFT JOIN chunks c ON e.chunk_id = c.id
                WHERE c.id IS NULL OR c.is_active = 0
                LIMIT ?
            """, (self.max_batch_size,))
            
            orphaned_embeddings = cursor.fetchall()
            result.items_analyzed += len(orphaned_embeddings)
            
            logger.info(f"Found {len(orphaned_embeddings)} orphaned embeddings")
            
            for emb in orphaned_embeddings:
                chunk_id = emb[0]
                
                if not self.dry_run_mode:
                    cursor.execute("DELETE FROM embeddings WHERE chunk_id = ?", (chunk_id,))
                
                result.add_deleted_item("embeddings", chunk_id)
            
            # Find orphaned context windows
            cursor.execute("""
                SELECT cw.id
                FROM context_windows cw
                LEFT JOIN chunks c ON cw.chunk_id = c.id
                WHERE c.id IS NULL OR c.is_active = 0
                LIMIT ?
            """, (self.max_batch_size,))
            
            orphaned_contexts = cursor.fetchall()
            result.items_analyzed += len(orphaned_contexts)
            
            for ctx in orphaned_contexts:
                ctx_id = ctx[0]
                
                if not self.dry_run_mode:
                    cursor.execute("DELETE FROM context_windows WHERE id = ?", (ctx_id,))
                
                result.add_deleted_item("context_windows", ctx_id)
            
            if not self.dry_run_mode:
                conn.commit()
            
            logger.info(f"Cleaned up {len(orphaned_embeddings) + len(orphaned_contexts)} orphaned records")
            
        except Exception as e:
            result.errors.append(f"Failed orphan cleanup: {e}")
            logger.error(f"Failed orphan cleanup: {e}")
    
    async def _prune_comprehensive(self, result: PruningResult):
        """Execute comprehensive pruning with all strategies"""
        logger.info("Starting comprehensive pruning")
        
        # Execute strategies in order of safety
        strategies = [
            ("Orphan cleanup", self._prune_orphans),
            ("Duplicate removal", self._prune_duplicates),
            ("Time-based pruning", self._prune_time_based),
            ("Usage-based pruning", self._prune_usage_based),
            ("Relevance-based pruning", self._prune_relevance_based),
            ("Size-based pruning", self._prune_size_based)
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                logger.info(f"Executing {strategy_name}")
                await strategy_func(result)
            except Exception as e:
                result.errors.append(f"{strategy_name} failed: {e}")
                logger.error(f"{strategy_name} failed: {e}")
    
    # Utility Methods
    
    def _get_database_size_mb(self) -> float:
        """Get database size in MB"""
        try:
            import os
            size_bytes = os.path.getsize(self.db_path)
            return size_bytes / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Failed to get database size: {e}")
            return 0.0
    
    def get_pruning_stats(self) -> Dict[str, Any]:
        """Get pruning statistics"""
        return {
            'total_pruning_runs': self.stats.total_pruning_runs,
            'total_items_deleted': self.stats.total_items_deleted,
            'total_storage_freed_mb': self.stats.total_storage_freed_mb,
            'last_pruning_run': self.stats.last_pruning_run.isoformat() if self.stats.last_pruning_run else None,
            'average_execution_time': self.stats.average_execution_time,
            'strategy_usage': self.stats.strategy_usage,
            'error_count': self.stats.error_count,
            'database_size_mb': self._get_database_size_mb()
        }
    
    def create_default_retention_policies(self):
        """Create default retention policies for all entity types"""
        default_policies = [
            RetentionPolicy(
                entity_type=RetentionPolicyType.CONVERSATION,
                policy_name="default",
                retention_days=365,  # 1 year
                auto_delete=False
            ),
            RetentionPolicy(
                entity_type=RetentionPolicyType.MESSAGE,
                policy_name="default",
                retention_days=180,  # 6 months
                auto_delete=False
            ),
            RetentionPolicy(
                entity_type=RetentionPolicyType.DOCUMENT,
                policy_name="default",
                retention_days=730,  # 2 years
                auto_delete=False
            ),
            RetentionPolicy(
                entity_type=RetentionPolicyType.ACCESS_LOG,
                policy_name="default",
                retention_days=90,   # 3 months
                auto_delete=True
            )
        ]
        
        for policy in default_policies:
            try:
                self.create_retention_policy(policy)
            except Exception as e:
                logger.warning(f"Failed to create default policy {policy.policy_name}: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PruningScheduler:
    """
    Scheduler for automated pruning operations.
    
    Manages periodic cleanup jobs and automated maintenance.
    """
    
    def __init__(self, pruner: MemoryPruner):
        """Initialize the pruning scheduler"""
        self.pruner = pruner
        self._running = False
        self._thread = None
        self._stop_event = threading.Event()
        
        # Default schedule: run comprehensive pruning daily at 2 AM
        self.schedule = {
            'comprehensive': {'hour': 2, 'minute': 0, 'interval_hours': 24},
            'orphan_cleanup': {'hour': 6, 'minute': 0, 'interval_hours': 6},
        }
    
    def start(self):
        """Start the pruning scheduler"""
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._thread.start()
        
        logger.info("Pruning scheduler started")
    
    def stop(self):
        """Stop the pruning scheduler"""
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=5)
        
        logger.info("Pruning scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        last_comprehensive = datetime.min
        last_orphan_cleanup = datetime.min
        
        while self._running and not self._stop_event.is_set():
            try:
                now = datetime.now()
                
                # Check if it's time for comprehensive pruning
                if self._should_run_job('comprehensive', last_comprehensive, now):
                    logger.info("Running scheduled comprehensive pruning")
                    
                    # Run async pruning in thread
                    import asyncio
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        result = loop.run_until_complete(
                            self.pruner.prune_memory(PruningStrategy.COMPREHENSIVE)
                        )
                        
                        logger.info(f"Scheduled pruning completed: {result.items_deleted} items deleted")
                        last_comprehensive = now
                        
                    except Exception as e:
                        logger.error(f"Scheduled comprehensive pruning failed: {e}")
                    finally:
                        loop.close()
                
                # Check if it's time for orphan cleanup
                if self._should_run_job('orphan_cleanup', last_orphan_cleanup, now):
                    logger.info("Running scheduled orphan cleanup")
                    
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        result = loop.run_until_complete(
                            self.pruner.prune_memory(PruningStrategy.ORPHAN_CLEANUP)
                        )
                        
                        logger.info(f"Scheduled orphan cleanup completed: {result.items_deleted} items cleaned")
                        last_orphan_cleanup = now
                        
                    except Exception as e:
                        logger.error(f"Scheduled orphan cleanup failed: {e}")
                    finally:
                        loop.close()
                
                # Sleep for 1 hour before next check
                self._stop_event.wait(3600)
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                self._stop_event.wait(60)  # Wait 1 minute on error
    
    def _should_run_job(self, job_name: str, last_run: datetime, now: datetime) -> bool:
        """Check if a job should run based on schedule"""
        if job_name not in self.schedule:
            return False
        
        schedule = self.schedule[job_name]
        interval_hours = schedule['interval_hours']
        
        # Check if enough time has passed
        time_since_last = now - last_run
        if time_since_last.total_seconds() < interval_hours * 3600:
            return False
        
        # Check if it's the right time of day
        target_hour = schedule['hour']
        target_minute = schedule['minute']
        
        if now.hour == target_hour and now.minute >= target_minute:
            return True
        
        return False 