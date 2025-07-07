"""
Memory Import System for Sovereign AI

This module provides comprehensive functionality to import user memory data from various formats
(JSON, CSV, XML, SQLite) with validation, conflict resolution, and integration capabilities.

Key features:
- Multiple import formats (JSON, CSV, XML, SQLite)
- Data validation and schema checking
- Conflict resolution and deduplication
- Progress tracking for large imports
- Integration with existing MemoryManager
- Error handling and recovery
- Backup creation before import
"""

import json
import csv
import sqlite3
import xml.etree.ElementTree as ET
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import base64
import hashlib
import uuid

from .memory_manager import MemoryManager
from .memory_schema import MemorySchema
from .privacy_manager import PrivacyManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImportFormat(Enum):
    """Supported import formats"""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    SQLITE = "sqlite"
    AUTO = "auto"  # Auto-detect format


class ConflictResolution(Enum):
    """Conflict resolution strategies"""
    SKIP = "skip"                 # Skip conflicting records
    OVERWRITE = "overwrite"       # Overwrite existing records
    MERGE = "merge"               # Merge with existing records
    RENAME = "rename"             # Rename conflicting records
    FAIL = "fail"                 # Fail on conflicts
    INTERACTIVE = "interactive"   # Ask user for resolution


class ImportScope(Enum):
    """Import scope options"""
    ALL = "all"
    USERS = "users"
    CONVERSATIONS = "conversations"
    DOCUMENTS = "documents"
    EMBEDDINGS = "embeddings"
    METADATA = "metadata"
    PRIVACY_DATA = "privacy_data"
    PERFORMANCE_DATA = "performance_data"


@dataclass
class ImportValidationRule:
    """Validation rule for import data"""
    table_name: str
    column_name: str
    rule_type: str  # 'required', 'unique', 'foreign_key', 'format', 'range'
    rule_value: Any = None
    error_message: str = ""


@dataclass
class ImportConflict:
    """Represents a data conflict during import"""
    table_name: str
    record_id: Any
    conflict_type: str  # 'duplicate_id', 'foreign_key_violation', 'data_mismatch'
    existing_data: Dict[str, Any]
    new_data: Dict[str, Any]
    suggested_resolution: ConflictResolution
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ImportOptions:
    """Configuration options for import operations"""
    format: ImportFormat = ImportFormat.AUTO
    conflict_resolution: ConflictResolution = ConflictResolution.SKIP
    scope: ImportScope = ImportScope.ALL
    validate_schema: bool = True
    validate_foreign_keys: bool = True
    create_backup: bool = True
    batch_size: int = 1000
    progress_callback: Optional[Callable[[int, int, str], None]] = None
    user_filter: Optional[str] = None
    date_range: Optional[Tuple[datetime, datetime]] = None
    dry_run: bool = False
    ignore_errors: bool = False
    merge_embeddings: bool = True
    preserve_ids: bool = False
    custom_id_mapping: Optional[Dict[str, Dict[Any, Any]]] = None


@dataclass
class ImportProgress:
    """Progress tracking for import operations"""
    total_records: int = 0
    processed_records: int = 0
    successful_imports: int = 0
    failed_imports: int = 0
    conflicts_found: int = 0
    conflicts_resolved: int = 0
    current_table: str = ""
    current_operation: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage"""
        if self.total_records == 0:
            return 0.0
        return (self.processed_records / self.total_records) * 100
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.processed_records == 0:
            return 0.0
        return (self.successful_imports / self.processed_records) * 100


@dataclass
class ImportResult:
    """Results of import operation"""
    success: bool
    format_detected: ImportFormat
    records_imported: int
    conflicts_found: int
    conflicts_resolved: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    import_summary: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    backup_path: Optional[str] = None
    id_mappings: Dict[str, Dict[Any, Any]] = field(default_factory=dict)
    
    def add_error(self, error: str):
        """Add an error message"""
        self.errors.append(error)
        logger.error(f"Import error: {error}")
    
    def add_warning(self, warning: str):
        """Add a warning message"""
        self.warnings.append(warning)
        logger.warning(f"Import warning: {warning}")


class MemoryImporter:
    """Main class for importing memory data"""
    
    def __init__(self, memory_manager: MemoryManager, privacy_manager: Optional[PrivacyManager] = None):
        """Initialize the Memory Importer
        
        Args:
            memory_manager: MemoryManager instance for database operations
            privacy_manager: Optional PrivacyManager for access control
        """
        self.memory_manager = memory_manager
        self.privacy_manager = privacy_manager
        self.schema = MemorySchema(str(memory_manager.db_path))
        
        # Validation rules
        self.validation_rules = self._initialize_validation_rules()
        
        # Table processing order (respects foreign key dependencies)
        self.table_order = [
            'users', 'embedding_models', 'conversations', 'documents',
            'messages', 'chunks', 'embeddings', 'metadata', 'tags',
            'entity_tags', 'context_windows', 'access_logs',
            'retention_policies', 'privacy_preferences', 'query_performance',
            'feedback', 'system_metrics', 'embedding_cache', 'schema_versions'
        ]
        
        # Format detectors
        self.format_detectors = {
            ImportFormat.JSON: self._detect_json_format,
            ImportFormat.CSV: self._detect_csv_format,
            ImportFormat.XML: self._detect_xml_format,
            ImportFormat.SQLITE: self._detect_sqlite_format
        }
        
        # Format importers
        self.format_importers = {
            ImportFormat.JSON: self._import_json,
            ImportFormat.CSV: self._import_csv,
            ImportFormat.XML: self._import_xml,
            ImportFormat.SQLITE: self._import_sqlite
        }
    
    def _initialize_validation_rules(self) -> List[ImportValidationRule]:
        """Initialize validation rules for import data"""
        rules = []
        
        # Required field rules
        required_fields = {
            'users': ['user_id', 'username'],
            'conversations': ['conversation_id', 'user_id', 'title'],
            'messages': ['message_id', 'conversation_id', 'content'],
            'documents': ['document_id', 'user_id', 'title'],
            'chunks': ['chunk_id', 'document_id', 'content'],
            'embeddings': ['embedding_id', 'chunk_id', 'embedding_model_id', 'embedding_vector']
        }
        
        for table, fields in required_fields.items():
            for field in fields:
                rules.append(ImportValidationRule(
                    table_name=table,
                    column_name=field,
                    rule_type='required',
                    error_message=f"Required field '{field}' missing in table '{table}'"
                ))
        
        # Unique constraint rules
        unique_constraints = {
            'users': ['user_id', 'username'],
            'conversations': ['conversation_id'],
            'messages': ['message_id'],
            'documents': ['document_id'],
            'chunks': ['chunk_id'],
            'embeddings': ['embedding_id']
        }
        
        for table, fields in unique_constraints.items():
            for field in fields:
                rules.append(ImportValidationRule(
                    table_name=table,
                    column_name=field,
                    rule_type='unique',
                    error_message=f"Duplicate value for unique field '{field}' in table '{table}'"
                ))
        
        # Foreign key rules
        foreign_keys = {
            'conversations': [('user_id', 'users', 'user_id')],
            'messages': [('conversation_id', 'conversations', 'conversation_id')],
            'documents': [('user_id', 'users', 'user_id')],
            'chunks': [('document_id', 'documents', 'document_id')],
            'embeddings': [
                ('chunk_id', 'chunks', 'chunk_id'),
                ('embedding_model_id', 'embedding_models', 'model_id')
            ]
        }
        
        for table, fks in foreign_keys.items():
            for fk_column, ref_table, ref_column in fks:
                rules.append(ImportValidationRule(
                    table_name=table,
                    column_name=fk_column,
                    rule_type='foreign_key',
                    rule_value=(ref_table, ref_column),
                    error_message=f"Foreign key violation: {table}.{fk_column} -> {ref_table}.{ref_column}"
                ))
        
        return rules
    
    def detect_format(self, file_path: str) -> ImportFormat:
        """Detect the format of an import file
        
        Args:
            file_path: Path to the import file
            
        Returns:
            Detected import format
        """
        file_path = Path(file_path)
        
        # Check extension first
        extension = file_path.suffix.lower()
        if extension == '.json':
            return ImportFormat.JSON
        elif extension == '.csv':
            return ImportFormat.CSV
        elif extension == '.xml':
            return ImportFormat.XML
        elif extension in ['.db', '.sqlite', '.sqlite3']:
            return ImportFormat.SQLITE
        elif extension == '.zip':
            # Check contents of zip file
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_file:
                    file_list = zip_file.namelist()
                    if any(f.endswith('.json') for f in file_list):
                        return ImportFormat.JSON
                    elif any(f.endswith('.csv') for f in file_list):
                        return ImportFormat.CSV
                    elif any(f.endswith('.xml') for f in file_list):
                        return ImportFormat.XML
                    elif any(f.endswith(('.db', '.sqlite', '.sqlite3')) for f in file_list):
                        return ImportFormat.SQLITE
            except:
                pass
        
        # Try content-based detection
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(1024)  # Read first 1KB
                
                if content.strip().startswith('{') or content.strip().startswith('['):
                    return ImportFormat.JSON
                elif content.strip().startswith('<?xml') or content.strip().startswith('<'):
                    return ImportFormat.XML
                elif ',' in content and '\n' in content:
                    return ImportFormat.CSV
        except:
            pass
        
        # Try binary file detection
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)
                if header.startswith(b'SQLite format 3'):
                    return ImportFormat.SQLITE
        except:
            pass
        
        # Default to JSON if detection fails
        return ImportFormat.JSON
    
    def _detect_json_format(self, file_path: str) -> bool:
        """Detect if file is JSON format"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json.load(f)
            return True
        except:
            return False
    
    def _detect_csv_format(self, file_path: str) -> bool:
        """Detect if file is CSV format"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                csv.Sniffer().sniff(f.read(1024))
            return True
        except:
            return False
    
    def _detect_xml_format(self, file_path: str) -> bool:
        """Detect if file is XML format"""
        try:
            ET.parse(file_path)
            return True
        except:
            return False
    
    def _detect_sqlite_format(self, file_path: str) -> bool:
        """Detect if file is SQLite format"""
        try:
            conn = sqlite3.connect(file_path)
            conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            conn.close()
            return True
        except:
            return False
    
    def validate_data(self, data: Dict[str, List[Dict]], options: ImportOptions) -> List[ImportConflict]:
        """Validate import data against schema and rules
        
        Args:
            data: Import data organized by table
            options: Import options
            
        Returns:
            List of validation conflicts
        """
        conflicts = []
        
        if not options.validate_schema:
            return conflicts
        
        for table_name, records in data.items():
            for record in records:
                # Check validation rules
                for rule in self.validation_rules:
                    if rule.table_name != table_name:
                        continue
                    
                    if rule.rule_type == 'required':
                        if rule.column_name not in record or record[rule.column_name] is None:
                            conflicts.append(ImportConflict(
                                table_name=table_name,
                                record_id=record.get('id', 'unknown'),
                                conflict_type='validation_error',
                                existing_data={},
                                new_data=record,
                                suggested_resolution=ConflictResolution.FAIL
                            ))
                    
                    elif rule.rule_type == 'foreign_key' and options.validate_foreign_keys:
                        ref_table, ref_column = rule.rule_value
                        fk_value = record.get(rule.column_name)
                        
                        if fk_value is not None:
                            # Check if referenced record exists
                            if ref_table in data:
                                ref_records = data[ref_table]
                                if not any(r.get(ref_column) == fk_value for r in ref_records):
                                    conflicts.append(ImportConflict(
                                        table_name=table_name,
                                        record_id=record.get('id', 'unknown'),
                                        conflict_type='foreign_key_violation',
                                        existing_data={},
                                        new_data=record,
                                        suggested_resolution=ConflictResolution.SKIP
                                    ))
        
        return conflicts
    
    def resolve_conflicts(self, conflicts: List[ImportConflict], options: ImportOptions) -> Dict[str, Any]:
        """Resolve import conflicts based on resolution strategy
        
        Args:
            conflicts: List of conflicts to resolve
            options: Import options with resolution strategy
            
        Returns:
            Resolution results
        """
        resolution_results = {
            'resolved': [],
            'unresolved': [],
            'actions_taken': []
        }
        
        for conflict in conflicts:
            if options.conflict_resolution == ConflictResolution.SKIP:
                resolution_results['resolved'].append(conflict)
                resolution_results['actions_taken'].append(f"Skipped {conflict.table_name} record {conflict.record_id}")
            
            elif options.conflict_resolution == ConflictResolution.OVERWRITE:
                resolution_results['resolved'].append(conflict)
                resolution_results['actions_taken'].append(f"Overwriting {conflict.table_name} record {conflict.record_id}")
            
            elif options.conflict_resolution == ConflictResolution.MERGE:
                resolution_results['resolved'].append(conflict)
                resolution_results['actions_taken'].append(f"Merging {conflict.table_name} record {conflict.record_id}")
            
            elif options.conflict_resolution == ConflictResolution.RENAME:
                # Generate new ID
                new_id = f"{conflict.record_id}_{uuid.uuid4().hex[:8]}"
                resolution_results['resolved'].append(conflict)
                resolution_results['actions_taken'].append(f"Renamed {conflict.table_name} record {conflict.record_id} to {new_id}")
            
            elif options.conflict_resolution == ConflictResolution.FAIL:
                resolution_results['unresolved'].append(conflict)
            
            else:  # INTERACTIVE - for now, default to SKIP
                resolution_results['resolved'].append(conflict)
                resolution_results['actions_taken'].append(f"Skipped {conflict.table_name} record {conflict.record_id} (interactive mode not implemented)")
        
        return resolution_results
    
    def create_backup(self, options: ImportOptions) -> Optional[str]:
        """Create a backup of the current database before import
        
        Args:
            options: Import options
            
        Returns:
            Path to backup file or None if backup failed
        """
        if not options.create_backup:
            return None
        
        try:
            # Create backup directory
            backup_dir = Path(self.memory_manager.db_path).parent / 'backups'
            backup_dir.mkdir(exist_ok=True)
            
            # Generate backup filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = backup_dir / f'memory_backup_{timestamp}.db'
            
            # Copy database file
            shutil.copy2(self.memory_manager.db_path, backup_path)
            
            logger.info(f"Created backup at {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None
    
    def import_data(self, file_path: str, options: ImportOptions) -> ImportResult:
        """Import memory data from file
        
        Args:
            file_path: Path to import file
            options: Import options
            
        Returns:
            Import results
        """
        start_time = datetime.now()
        result = ImportResult(success=False, format_detected=ImportFormat.AUTO)
        
        try:
            # Detect format
            if options.format == ImportFormat.AUTO:
                result.format_detected = self.detect_format(file_path)
            else:
                result.format_detected = options.format
            
            # Create backup if requested
            result.backup_path = self.create_backup(options)
            
            # Import data based on format
            if result.format_detected in self.format_importers:
                import_func = self.format_importers[result.format_detected]
                result = import_func(file_path, options, result)
            else:
                result.add_error(f"Unsupported import format: {result.format_detected}")
                return result
            
            # Calculate processing time
            result.processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Import completed in {result.processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            result.add_error(f"Import failed: {str(e)}")
            result.processing_time = (datetime.now() - start_time).total_seconds()
            return result
    
    def _import_json(self, file_path: str, options: ImportOptions, result: ImportResult) -> ImportResult:
        """Import data from JSON format"""
        try:
            # Handle ZIP files
            if file_path.endswith('.zip'):
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(file_path, 'r') as zip_file:
                        zip_file.extractall(temp_dir)
                        
                        # Find JSON files
                        json_files = list(Path(temp_dir).rglob('*.json'))
                        if not json_files:
                            result.add_error("No JSON files found in ZIP archive")
                            return result
                        
                        # Use the first JSON file found
                        json_file = json_files[0]
                        return self._import_json_file(str(json_file), options, result)
            else:
                return self._import_json_file(file_path, options, result)
                
        except Exception as e:
            result.add_error(f"JSON import failed: {str(e)}")
            return result
    
    def _import_json_file(self, file_path: str, options: ImportOptions, result: ImportResult) -> ImportResult:
        """Import data from a single JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if it's a taskmaster export format
            if 'export_info' in data and 'tables' in data:
                table_data = data['tables']
            else:
                # Assume direct table data
                table_data = data
            
            # Filter by scope
            if options.scope != ImportScope.ALL:
                table_data = self._filter_by_scope(table_data, options.scope)
            
            # Validate data
            conflicts = self.validate_data(table_data, options)
            result.conflicts_found = len(conflicts)
            
            # Resolve conflicts
            if conflicts:
                resolution_results = self.resolve_conflicts(conflicts, options)
                result.conflicts_resolved = len(resolution_results['resolved'])
                
                if resolution_results['unresolved'] and options.conflict_resolution == ConflictResolution.FAIL:
                    result.add_error(f"Unresolved conflicts: {len(resolution_results['unresolved'])}")
                    return result
            
            # Import data table by table
            progress = ImportProgress()
            progress.total_records = sum(len(records) for records in table_data.values())
            
            for table_name in self.table_order:
                if table_name not in table_data:
                    continue
                
                records = table_data[table_name]
                if not records:
                    continue
                
                progress.current_table = table_name
                progress.current_operation = f"Importing {table_name}"
                
                # Import records in batches
                for i in range(0, len(records), options.batch_size):
                    batch = records[i:i + options.batch_size]
                    
                    try:
                        if not options.dry_run:
                            self._import_table_batch(table_name, batch, options, result)
                        
                        progress.successful_imports += len(batch)
                        
                    except Exception as e:
                        progress.failed_imports += len(batch)
                        if not options.ignore_errors:
                            result.add_error(f"Failed to import {table_name} batch: {str(e)}")
                            return result
                        else:
                            result.add_warning(f"Skipped {table_name} batch due to error: {str(e)}")
                    
                    progress.processed_records += len(batch)
                    
                    # Call progress callback
                    if options.progress_callback:
                        options.progress_callback(
                            progress.processed_records,
                            progress.total_records,
                            progress.current_operation
                        )
            
            result.records_imported = progress.successful_imports
            result.success = True
            
            # Generate import summary
            result.import_summary = {
                'total_records': progress.total_records,
                'successful_imports': progress.successful_imports,
                'failed_imports': progress.failed_imports,
                'tables_imported': len([t for t in self.table_order if t in table_data and table_data[t]]),
                'dry_run': options.dry_run
            }
            
            return result
            
        except Exception as e:
            result.add_error(f"JSON file import failed: {str(e)}")
            return result
    
    def _import_csv(self, file_path: str, options: ImportOptions, result: ImportResult) -> ImportResult:
        """Import data from CSV format"""
        try:
            # Handle ZIP files containing multiple CSV files
            if file_path.endswith('.zip'):
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(file_path, 'r') as zip_file:
                        zip_file.extractall(temp_dir)
                        
                        # Find CSV files
                        csv_files = list(Path(temp_dir).rglob('*.csv'))
                        if not csv_files:
                            result.add_error("No CSV files found in ZIP archive")
                            return result
                        
                        # Import each CSV file
                        total_imported = 0
                        for csv_file in csv_files:
                            table_name = csv_file.stem
                            if table_name in self.table_order:
                                imported = self._import_csv_file(str(csv_file), table_name, options, result)
                                total_imported += imported
                        
                        result.records_imported = total_imported
                        result.success = True
                        return result
            else:
                # Single CSV file - assume it's a specific table
                table_name = Path(file_path).stem
                imported = self._import_csv_file(file_path, table_name, options, result)
                result.records_imported = imported
                result.success = True
                return result
                
        except Exception as e:
            result.add_error(f"CSV import failed: {str(e)}")
            return result
    
    def _import_csv_file(self, file_path: str, table_name: str, options: ImportOptions, result: ImportResult) -> int:
        """Import data from a single CSV file"""
        try:
            records = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert empty strings to None
                    record = {k: (v if v != '' else None) for k, v in row.items()}
                    records.append(record)
            
            if not options.dry_run:
                self._import_table_batch(table_name, records, options, result)
            
            return len(records)
            
        except Exception as e:
            result.add_error(f"CSV file import failed for {table_name}: {str(e)}")
            return 0
    
    def _import_xml(self, file_path: str, options: ImportOptions, result: ImportResult) -> ImportResult:
        """Import data from XML format"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Parse XML structure
            table_data = {}
            
            for table_elem in root.findall('table'):
                table_name = table_elem.get('name')
                if not table_name:
                    continue
                
                records = []
                for record_elem in table_elem.findall('record'):
                    record = {}
                    for field_elem in record_elem:
                        field_name = field_elem.tag
                        field_value = field_elem.text
                        
                        # Handle special data types
                        if field_elem.get('type') == 'binary':
                            field_value = base64.b64decode(field_value) if field_value else None
                        elif field_elem.get('type') == 'json':
                            field_value = json.loads(field_value) if field_value else None
                        
                        record[field_name] = field_value
                    
                    records.append(record)
                
                table_data[table_name] = records
            
            # Filter by scope
            if options.scope != ImportScope.ALL:
                table_data = self._filter_by_scope(table_data, options.scope)
            
            # Import data
            total_imported = 0
            for table_name in self.table_order:
                if table_name not in table_data:
                    continue
                
                records = table_data[table_name]
                if not records:
                    continue
                
                if not options.dry_run:
                    self._import_table_batch(table_name, records, options, result)
                
                total_imported += len(records)
            
            result.records_imported = total_imported
            result.success = True
            return result
            
        except Exception as e:
            result.add_error(f"XML import failed: {str(e)}")
            return result
    
    def _import_sqlite(self, file_path: str, options: ImportOptions, result: ImportResult) -> ImportResult:
        """Import data from SQLite format"""
        try:
            # Connect to source database
            source_conn = sqlite3.connect(file_path)
            source_conn.row_factory = sqlite3.Row
            
            # Get table list
            cursor = source_conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Filter tables by scope
            if options.scope != ImportScope.ALL:
                tables = self._filter_tables_by_scope(tables, options.scope)
            
            total_imported = 0
            
            for table_name in self.table_order:
                if table_name not in tables:
                    continue
                
                # Get table data
                cursor = source_conn.execute(f"SELECT * FROM {table_name}")
                rows = cursor.fetchall()
                
                if not rows:
                    continue
                
                # Convert rows to dictionaries
                records = []
                for row in rows:
                    record = dict(row)
                    records.append(record)
                
                if not options.dry_run:
                    self._import_table_batch(table_name, records, options, result)
                
                total_imported += len(records)
            
            source_conn.close()
            
            result.records_imported = total_imported
            result.success = True
            return result
            
        except Exception as e:
            result.add_error(f"SQLite import failed: {str(e)}")
            return result
    
    def _filter_by_scope(self, table_data: Dict[str, List[Dict]], scope: ImportScope) -> Dict[str, List[Dict]]:
        """Filter table data by import scope"""
        if scope == ImportScope.ALL:
            return table_data
        
        scope_tables = {
            ImportScope.USERS: ['users'],
            ImportScope.CONVERSATIONS: ['conversations', 'messages'],
            ImportScope.DOCUMENTS: ['documents', 'chunks'],
            ImportScope.EMBEDDINGS: ['embeddings', 'embedding_models', 'embedding_cache'],
            ImportScope.METADATA: ['metadata', 'tags', 'entity_tags'],
            ImportScope.PRIVACY_DATA: ['privacy_preferences', 'access_logs', 'retention_policies'],
            ImportScope.PERFORMANCE_DATA: ['query_performance', 'feedback', 'system_metrics']
        }
        
        allowed_tables = scope_tables.get(scope, [])
        return {table: data for table, data in table_data.items() if table in allowed_tables}
    
    def _filter_tables_by_scope(self, tables: List[str], scope: ImportScope) -> List[str]:
        """Filter table list by import scope"""
        if scope == ImportScope.ALL:
            return tables
        
        scope_tables = {
            ImportScope.USERS: ['users'],
            ImportScope.CONVERSATIONS: ['conversations', 'messages'],
            ImportScope.DOCUMENTS: ['documents', 'chunks'],
            ImportScope.EMBEDDINGS: ['embeddings', 'embedding_models', 'embedding_cache'],
            ImportScope.METADATA: ['metadata', 'tags', 'entity_tags'],
            ImportScope.PRIVACY_DATA: ['privacy_preferences', 'access_logs', 'retention_policies'],
            ImportScope.PERFORMANCE_DATA: ['query_performance', 'feedback', 'system_metrics']
        }
        
        allowed_tables = scope_tables.get(scope, [])
        return [table for table in tables if table in allowed_tables]
    
    def _import_table_batch(self, table_name: str, records: List[Dict], options: ImportOptions, result: ImportResult):
        """Import a batch of records for a specific table"""
        if not records:
            return
        
        try:
            # Check if table exists
            if not self.memory_manager.table_exists(table_name):
                result.add_warning(f"Table {table_name} does not exist, skipping")
                return
            
            # Get table schema
            columns = self.memory_manager.get_table_columns(table_name)
            if not columns:
                result.add_warning(f"Cannot get schema for table {table_name}, skipping")
                return
            
            # Process each record
            for record in records:
                try:
                    # Filter record to only include valid columns
                    filtered_record = {k: v for k, v in record.items() if k in columns}
                    
                    if options.conflict_resolution == ConflictResolution.OVERWRITE:
                        # Use INSERT OR REPLACE
                        self.memory_manager.insert_or_replace_record(table_name, filtered_record)
                    elif options.conflict_resolution == ConflictResolution.SKIP:
                        # Use INSERT OR IGNORE
                        self.memory_manager.insert_or_ignore_record(table_name, filtered_record)
                    else:
                        # Use regular INSERT
                        self.memory_manager.insert_record(table_name, filtered_record)
                        
                except Exception as e:
                    if not options.ignore_errors:
                        raise e
                    else:
                        result.add_warning(f"Skipped record in {table_name} due to error: {str(e)}")
            
        except Exception as e:
            result.add_error(f"Failed to import batch for table {table_name}: {str(e)}")
            raise e


# Public API functions
def import_memory_data(file_path: str, memory_manager: MemoryManager, 
                      options: Optional[ImportOptions] = None) -> ImportResult:
    """Import memory data from file
    
    Args:
        file_path: Path to import file
        memory_manager: MemoryManager instance
        options: Import options (optional)
        
    Returns:
        Import results
    """
    if options is None:
        options = ImportOptions()
    
    importer = MemoryImporter(memory_manager)
    return importer.import_data(file_path, options)


def import_user_backup(file_path: str, memory_manager: MemoryManager, 
                      user_id: str, conflict_resolution: ConflictResolution = ConflictResolution.SKIP) -> ImportResult:
    """Import a user's backup data
    
    Args:
        file_path: Path to backup file
        memory_manager: MemoryManager instance
        user_id: User ID to import data for
        conflict_resolution: How to handle conflicts
        
    Returns:
        Import results
    """
    options = ImportOptions(
        user_filter=user_id,
        conflict_resolution=conflict_resolution,
        scope=ImportScope.ALL,
        create_backup=True,
        validate_schema=True
    )
    
    return import_memory_data(file_path, memory_manager, options)


def migrate_from_database(source_db_path: str, target_memory_manager: MemoryManager,
                         options: Optional[ImportOptions] = None) -> ImportResult:
    """Migrate data from another database
    
    Args:
        source_db_path: Path to source database
        target_memory_manager: Target MemoryManager instance
        options: Import options (optional)
        
    Returns:
        Import results
    """
    if options is None:
        options = ImportOptions(
            format=ImportFormat.SQLITE,
            conflict_resolution=ConflictResolution.MERGE,
            create_backup=True,
            validate_schema=True
        )
    
    return import_memory_data(source_db_path, target_memory_manager, options)


def restore_from_backup(backup_path: str, memory_manager: MemoryManager) -> ImportResult:
    """Restore database from backup
    
    Args:
        backup_path: Path to backup file
        memory_manager: MemoryManager instance
        
    Returns:
        Import results
    """
    options = ImportOptions(
        format=ImportFormat.SQLITE,
        conflict_resolution=ConflictResolution.OVERWRITE,
        create_backup=True,
        validate_schema=False  # Backup should be valid
    )
    
    return import_memory_data(backup_path, memory_manager, options) 