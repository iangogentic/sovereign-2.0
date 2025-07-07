"""
Tests for Memory Import System - Comprehensive Test Suite

This module tests the memory import functionality including:
- Multiple import formats (JSON, CSV, XML, SQLite)
- Data validation and schema checking
- Conflict resolution strategies
- Integration with MemoryManager
- Error handling and recovery
- Progress tracking
- Backup creation
"""

import pytest
import tempfile
import shutil
import json
import csv
import xml.etree.ElementTree as ET
import sqlite3
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import base64

from src.sovereign.memory_importer import (
    MemoryImporter, ImportFormat, ConflictResolution, ImportScope,
    ImportOptions, ImportResult, ImportConflict, ImportProgress,
    ImportValidationRule, import_memory_data, import_user_backup,
    migrate_from_database, restore_from_backup
)
from src.sovereign.memory_manager import MemoryManager
from src.sovereign.privacy_manager import PrivacyManager


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_memory_manager():
    """Create a mock MemoryManager for testing"""
    mock_manager = Mock(spec=MemoryManager)
    mock_manager.db_path = "/tmp/test.db"
    mock_manager.table_exists.return_value = True
    mock_manager.get_table_columns.return_value = ['id', 'name', 'data']
    mock_manager.insert_record.return_value = True
    mock_manager.insert_or_replace_record.return_value = True
    mock_manager.insert_or_ignore_record.return_value = True
    return mock_manager


@pytest.fixture
def mock_privacy_manager():
    """Create a mock PrivacyManager for testing"""
    mock_manager = Mock(spec=PrivacyManager)
    mock_manager.can_access_data.return_value = True
    return mock_manager


@pytest.fixture
def sample_json_data():
    """Create sample JSON data for testing"""
    return {
        "export_info": {
            "format": "json",
            "version": "1.0",
            "timestamp": "2024-01-15T10:00:00Z",
            "total_records": 3
        },
        "tables": {
            "users": [
                {
                    "user_id": "user1",
                    "username": "testuser1",
                    "email": "test1@example.com",
                    "created_at": "2024-01-01T00:00:00Z"
                },
                {
                    "user_id": "user2",
                    "username": "testuser2",
                    "email": "test2@example.com",
                    "created_at": "2024-01-02T00:00:00Z"
                }
            ],
            "conversations": [
                {
                    "conversation_id": "conv1",
                    "user_id": "user1",
                    "title": "Test Conversation",
                    "created_at": "2024-01-01T01:00:00Z"
                }
            ]
        }
    }


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing"""
    return {
        "users": [
            ["user_id", "username", "email", "created_at"],
            ["user1", "testuser1", "test1@example.com", "2024-01-01T00:00:00Z"],
            ["user2", "testuser2", "test2@example.com", "2024-01-02T00:00:00Z"]
        ],
        "conversations": [
            ["conversation_id", "user_id", "title", "created_at"],
            ["conv1", "user1", "Test Conversation", "2024-01-01T01:00:00Z"]
        ]
    }


class TestImportFormat:
    """Test import format enumeration"""
    
    def test_format_values(self):
        """Test format enumeration values"""
        assert ImportFormat.JSON.value == "json"
        assert ImportFormat.CSV.value == "csv"
        assert ImportFormat.XML.value == "xml"
        assert ImportFormat.SQLITE.value == "sqlite"
        assert ImportFormat.AUTO.value == "auto"


class TestConflictResolution:
    """Test conflict resolution strategies"""
    
    def test_resolution_values(self):
        """Test conflict resolution enumeration values"""
        assert ConflictResolution.SKIP.value == "skip"
        assert ConflictResolution.OVERWRITE.value == "overwrite"
        assert ConflictResolution.MERGE.value == "merge"
        assert ConflictResolution.RENAME.value == "rename"
        assert ConflictResolution.FAIL.value == "fail"
        assert ConflictResolution.INTERACTIVE.value == "interactive"


class TestImportScope:
    """Test import scope options"""
    
    def test_scope_values(self):
        """Test import scope enumeration values"""
        assert ImportScope.ALL.value == "all"
        assert ImportScope.USERS.value == "users"
        assert ImportScope.CONVERSATIONS.value == "conversations"
        assert ImportScope.DOCUMENTS.value == "documents"
        assert ImportScope.EMBEDDINGS.value == "embeddings"
        assert ImportScope.METADATA.value == "metadata"
        assert ImportScope.PRIVACY_DATA.value == "privacy_data"
        assert ImportScope.PERFORMANCE_DATA.value == "performance_data"


class TestImportOptions:
    """Test import options data class"""
    
    def test_default_options(self):
        """Test default import options"""
        options = ImportOptions()
        assert options.format == ImportFormat.AUTO
        assert options.conflict_resolution == ConflictResolution.SKIP
        assert options.scope == ImportScope.ALL
        assert options.validate_schema is True
        assert options.validate_foreign_keys is True
        assert options.create_backup is True
        assert options.batch_size == 1000
        assert options.dry_run is False
        assert options.ignore_errors is False
    
    def test_custom_options(self):
        """Test custom import options"""
        options = ImportOptions(
            format=ImportFormat.JSON,
            conflict_resolution=ConflictResolution.OVERWRITE,
            scope=ImportScope.USERS,
            batch_size=500,
            dry_run=True
        )
        assert options.format == ImportFormat.JSON
        assert options.conflict_resolution == ConflictResolution.OVERWRITE
        assert options.scope == ImportScope.USERS
        assert options.batch_size == 500
        assert options.dry_run is True


class TestImportProgress:
    """Test import progress tracking"""
    
    def test_progress_initialization(self):
        """Test progress initialization"""
        progress = ImportProgress()
        assert progress.total_records == 0
        assert progress.processed_records == 0
        assert progress.successful_imports == 0
        assert progress.failed_imports == 0
        assert progress.conflicts_found == 0
        assert progress.conflicts_resolved == 0
        assert progress.current_table == ""
        assert progress.current_operation == ""
        assert isinstance(progress.start_time, datetime)
    
    def test_completion_percentage(self):
        """Test completion percentage calculation"""
        progress = ImportProgress()
        progress.total_records = 100
        progress.processed_records = 25
        assert progress.completion_percentage == 25.0
        
        progress.processed_records = 0
        assert progress.completion_percentage == 0.0
        
        progress.total_records = 0
        assert progress.completion_percentage == 0.0
    
    def test_success_rate(self):
        """Test success rate calculation"""
        progress = ImportProgress()
        progress.processed_records = 100
        progress.successful_imports = 85
        assert progress.success_rate == 85.0
        
        progress.processed_records = 0
        assert progress.success_rate == 0.0


class TestImportResult:
    """Test import result data class"""
    
    def test_result_initialization(self):
        """Test result initialization"""
        result = ImportResult(success=True, format_detected=ImportFormat.JSON)
        assert result.success is True
        assert result.format_detected == ImportFormat.JSON
        assert result.records_imported == 0
        assert result.conflicts_found == 0
        assert result.conflicts_resolved == 0
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        assert result.processing_time == 0.0
        assert result.backup_path is None
    
    def test_add_error(self):
        """Test adding error messages"""
        result = ImportResult(success=False, format_detected=ImportFormat.JSON)
        result.add_error("Test error")
        assert len(result.errors) == 1
        assert "Test error" in result.errors[0]
    
    def test_add_warning(self):
        """Test adding warning messages"""
        result = ImportResult(success=True, format_detected=ImportFormat.JSON)
        result.add_warning("Test warning")
        assert len(result.warnings) == 1
        assert "Test warning" in result.warnings[0]


class TestMemoryImporter:
    """Test the main MemoryImporter class"""
    
    def test_importer_initialization(self, mock_memory_manager, mock_privacy_manager):
        """Test importer initialization"""
        importer = MemoryImporter(mock_memory_manager, mock_privacy_manager)
        assert importer.memory_manager == mock_memory_manager
        assert importer.privacy_manager == mock_privacy_manager
        assert len(importer.validation_rules) > 0
        assert len(importer.table_order) > 0
        assert len(importer.format_detectors) == 4
        assert len(importer.format_importers) == 4
    
    def test_validation_rules_initialization(self, mock_memory_manager):
        """Test validation rules initialization"""
        importer = MemoryImporter(mock_memory_manager)
        rules = importer.validation_rules
        
        # Check that we have required field rules
        required_rules = [r for r in rules if r.rule_type == 'required']
        assert len(required_rules) > 0
        
        # Check that we have unique constraint rules
        unique_rules = [r for r in rules if r.rule_type == 'unique']
        assert len(unique_rules) > 0
        
        # Check that we have foreign key rules
        fk_rules = [r for r in rules if r.rule_type == 'foreign_key']
        assert len(fk_rules) > 0
    
    def test_format_detection_json(self, mock_memory_manager, temp_dir, sample_json_data):
        """Test JSON format detection"""
        importer = MemoryImporter(mock_memory_manager)
        
        # Create JSON file
        json_file = Path(temp_dir) / "test.json"
        with open(json_file, 'w') as f:
            json.dump(sample_json_data, f)
        
        format_detected = importer.detect_format(str(json_file))
        assert format_detected == ImportFormat.JSON
    
    def test_format_detection_csv(self, mock_memory_manager, temp_dir, sample_csv_data):
        """Test CSV format detection"""
        importer = MemoryImporter(mock_memory_manager)
        
        # Create CSV file
        csv_file = Path(temp_dir) / "test.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in sample_csv_data["users"]:
                writer.writerow(row)
        
        format_detected = importer.detect_format(str(csv_file))
        assert format_detected == ImportFormat.CSV
    
    def test_format_detection_xml(self, mock_memory_manager, temp_dir):
        """Test XML format detection"""
        importer = MemoryImporter(mock_memory_manager)
        
        # Create XML file
        xml_file = Path(temp_dir) / "test.xml"
        root = ET.Element("export")
        table = ET.SubElement(root, "table", name="users")
        record = ET.SubElement(table, "record")
        user_id = ET.SubElement(record, "user_id")
        user_id.text = "user1"
        
        tree = ET.ElementTree(root)
        tree.write(xml_file, encoding='utf-8', xml_declaration=True)
        
        format_detected = importer.detect_format(str(xml_file))
        assert format_detected == ImportFormat.XML
    
    def test_format_detection_sqlite(self, mock_memory_manager, temp_dir):
        """Test SQLite format detection"""
        importer = MemoryImporter(mock_memory_manager)
        
        # Create SQLite file
        sqlite_file = Path(temp_dir) / "test.db"
        conn = sqlite3.connect(sqlite_file)
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO users (name) VALUES ('test')")
        conn.commit()
        conn.close()
        
        format_detected = importer.detect_format(str(sqlite_file))
        assert format_detected == ImportFormat.SQLITE
    
    def test_format_detection_zip(self, mock_memory_manager, temp_dir, sample_json_data):
        """Test ZIP format detection"""
        importer = MemoryImporter(mock_memory_manager)
        
        # Create JSON file
        json_file = Path(temp_dir) / "data.json"
        with open(json_file, 'w') as f:
            json.dump(sample_json_data, f)
        
        # Create ZIP file
        zip_file = Path(temp_dir) / "test.zip"
        with zipfile.ZipFile(zip_file, 'w') as zf:
            zf.write(json_file, "data.json")
        
        format_detected = importer.detect_format(str(zip_file))
        assert format_detected == ImportFormat.JSON
    
    def test_data_validation(self, mock_memory_manager):
        """Test data validation"""
        importer = MemoryImporter(mock_memory_manager)
        
        # Valid data
        valid_data = {
            "users": [
                {"user_id": "user1", "username": "test1"},
                {"user_id": "user2", "username": "test2"}
            ],
            "conversations": [
                {"conversation_id": "conv1", "user_id": "user1", "title": "Test"}
            ]
        }
        
        options = ImportOptions(validate_schema=True, validate_foreign_keys=True)
        conflicts = importer.validate_data(valid_data, options)
        
        # Should have no conflicts for valid data
        assert len(conflicts) == 0
    
    def test_data_validation_missing_required(self, mock_memory_manager):
        """Test validation with missing required fields"""
        importer = MemoryImporter(mock_memory_manager)
        
        # Invalid data - missing required fields
        invalid_data = {
            "users": [
                {"user_id": "user1"},  # Missing username
                {"username": "test2"}  # Missing user_id
            ]
        }
        
        options = ImportOptions(validate_schema=True)
        conflicts = importer.validate_data(invalid_data, options)
        
        # Should have conflicts for missing required fields
        assert len(conflicts) > 0
        assert any(c.conflict_type == 'validation_error' for c in conflicts)
    
    def test_data_validation_foreign_key_violation(self, mock_memory_manager):
        """Test validation with foreign key violations"""
        importer = MemoryImporter(mock_memory_manager)
        
        # Invalid data - foreign key violation
        invalid_data = {
            "users": [
                {"user_id": "user1", "username": "test1"}
            ],
            "conversations": [
                {"conversation_id": "conv1", "user_id": "user2", "title": "Test"}  # user2 doesn't exist
            ]
        }
        
        options = ImportOptions(validate_schema=True, validate_foreign_keys=True)
        conflicts = importer.validate_data(invalid_data, options)
        
        # Should have conflicts for foreign key violations
        assert len(conflicts) > 0
        assert any(c.conflict_type == 'foreign_key_violation' for c in conflicts)
    
    def test_conflict_resolution_skip(self, mock_memory_manager):
        """Test conflict resolution with SKIP strategy"""
        importer = MemoryImporter(mock_memory_manager)
        
        conflicts = [
            ImportConflict(
                table_name="users",
                record_id="user1",
                conflict_type="duplicate_id",
                existing_data={"user_id": "user1", "username": "old"},
                new_data={"user_id": "user1", "username": "new"},
                suggested_resolution=ConflictResolution.SKIP
            )
        ]
        
        options = ImportOptions(conflict_resolution=ConflictResolution.SKIP)
        results = importer.resolve_conflicts(conflicts, options)
        
        assert len(results['resolved']) == 1
        assert len(results['unresolved']) == 0
        assert "Skipped" in results['actions_taken'][0]
    
    def test_conflict_resolution_overwrite(self, mock_memory_manager):
        """Test conflict resolution with OVERWRITE strategy"""
        importer = MemoryImporter(mock_memory_manager)
        
        conflicts = [
            ImportConflict(
                table_name="users",
                record_id="user1",
                conflict_type="duplicate_id",
                existing_data={"user_id": "user1", "username": "old"},
                new_data={"user_id": "user1", "username": "new"},
                suggested_resolution=ConflictResolution.OVERWRITE
            )
        ]
        
        options = ImportOptions(conflict_resolution=ConflictResolution.OVERWRITE)
        results = importer.resolve_conflicts(conflicts, options)
        
        assert len(results['resolved']) == 1
        assert len(results['unresolved']) == 0
        assert "Overwriting" in results['actions_taken'][0]
    
    def test_conflict_resolution_fail(self, mock_memory_manager):
        """Test conflict resolution with FAIL strategy"""
        importer = MemoryImporter(mock_memory_manager)
        
        conflicts = [
            ImportConflict(
                table_name="users",
                record_id="user1",
                conflict_type="duplicate_id",
                existing_data={"user_id": "user1", "username": "old"},
                new_data={"user_id": "user1", "username": "new"},
                suggested_resolution=ConflictResolution.FAIL
            )
        ]
        
        options = ImportOptions(conflict_resolution=ConflictResolution.FAIL)
        results = importer.resolve_conflicts(conflicts, options)
        
        assert len(results['resolved']) == 0
        assert len(results['unresolved']) == 1
    
    @patch('shutil.copy2')
    def test_create_backup(self, mock_copy, mock_memory_manager):
        """Test backup creation"""
        importer = MemoryImporter(mock_memory_manager)
        
        options = ImportOptions(create_backup=True)
        backup_path = importer.create_backup(options)
        
        assert backup_path is not None
        assert "backup" in backup_path
        mock_copy.assert_called_once()
    
    def test_create_backup_disabled(self, mock_memory_manager):
        """Test backup creation when disabled"""
        importer = MemoryImporter(mock_memory_manager)
        
        options = ImportOptions(create_backup=False)
        backup_path = importer.create_backup(options)
        
        assert backup_path is None
    
    def test_json_import_success(self, mock_memory_manager, temp_dir, sample_json_data):
        """Test successful JSON import"""
        importer = MemoryImporter(mock_memory_manager)
        
        # Create JSON file
        json_file = Path(temp_dir) / "test.json"
        with open(json_file, 'w') as f:
            json.dump(sample_json_data, f)
        
        options = ImportOptions(format=ImportFormat.JSON, create_backup=False)
        result = importer.import_data(str(json_file), options)
        
        assert result.success is True
        assert result.format_detected == ImportFormat.JSON
        assert result.records_imported > 0
        assert len(result.errors) == 0
    
    def test_json_import_zip(self, mock_memory_manager, temp_dir, sample_json_data):
        """Test JSON import from ZIP file"""
        importer = MemoryImporter(mock_memory_manager)
        
        # Create JSON file
        json_file = Path(temp_dir) / "data.json"
        with open(json_file, 'w') as f:
            json.dump(sample_json_data, f)
        
        # Create ZIP file
        zip_file = Path(temp_dir) / "test.zip"
        with zipfile.ZipFile(zip_file, 'w') as zf:
            zf.write(json_file, "data.json")
        
        options = ImportOptions(format=ImportFormat.JSON, create_backup=False)
        result = importer.import_data(str(zip_file), options)
        
        assert result.success is True
        assert result.format_detected == ImportFormat.JSON
        assert result.records_imported > 0
    
    def test_csv_import_success(self, mock_memory_manager, temp_dir, sample_csv_data):
        """Test successful CSV import"""
        importer = MemoryImporter(mock_memory_manager)
        
        # Create CSV files
        for table_name, rows in sample_csv_data.items():
            csv_file = Path(temp_dir) / f"{table_name}.csv"
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                for row in rows:
                    writer.writerow(row)
        
        # Create ZIP file with CSV files
        zip_file = Path(temp_dir) / "test.zip"
        with zipfile.ZipFile(zip_file, 'w') as zf:
            for table_name in sample_csv_data:
                csv_file = Path(temp_dir) / f"{table_name}.csv"
                zf.write(csv_file, f"{table_name}.csv")
        
        options = ImportOptions(format=ImportFormat.CSV, create_backup=False)
        result = importer.import_data(str(zip_file), options)
        
        assert result.success is True
        assert result.format_detected == ImportFormat.CSV
        assert result.records_imported > 0
    
    def test_xml_import_success(self, mock_memory_manager, temp_dir):
        """Test successful XML import"""
        importer = MemoryImporter(mock_memory_manager)
        
        # Create XML file
        xml_file = Path(temp_dir) / "test.xml"
        root = ET.Element("export")
        
        # Add users table
        users_table = ET.SubElement(root, "table", name="users")
        user_record = ET.SubElement(users_table, "record")
        user_id = ET.SubElement(user_record, "user_id")
        user_id.text = "user1"
        username = ET.SubElement(user_record, "username")
        username.text = "testuser"
        
        tree = ET.ElementTree(root)
        tree.write(xml_file, encoding='utf-8', xml_declaration=True)
        
        options = ImportOptions(format=ImportFormat.XML, create_backup=False)
        result = importer.import_data(str(xml_file), options)
        
        assert result.success is True
        assert result.format_detected == ImportFormat.XML
        assert result.records_imported > 0
    
    def test_sqlite_import_success(self, mock_memory_manager, temp_dir):
        """Test successful SQLite import"""
        importer = MemoryImporter(mock_memory_manager)
        
        # Create SQLite file
        sqlite_file = Path(temp_dir) / "test.db"
        conn = sqlite3.connect(sqlite_file)
        conn.execute("CREATE TABLE users (user_id TEXT PRIMARY KEY, username TEXT)")
        conn.execute("INSERT INTO users (user_id, username) VALUES ('user1', 'testuser')")
        conn.commit()
        conn.close()
        
        options = ImportOptions(format=ImportFormat.SQLITE, create_backup=False)
        result = importer.import_data(str(sqlite_file), options)
        
        assert result.success is True
        assert result.format_detected == ImportFormat.SQLITE
        assert result.records_imported > 0
    
    def test_import_with_scope_filter(self, mock_memory_manager, temp_dir, sample_json_data):
        """Test import with scope filtering"""
        importer = MemoryImporter(mock_memory_manager)
        
        # Create JSON file
        json_file = Path(temp_dir) / "test.json"
        with open(json_file, 'w') as f:
            json.dump(sample_json_data, f)
        
        # Import only users
        options = ImportOptions(
            format=ImportFormat.JSON,
            scope=ImportScope.USERS,
            create_backup=False
        )
        result = importer.import_data(str(json_file), options)
        
        assert result.success is True
        # Should have processed only users table
        assert result.records_imported == 2  # 2 users
    
    def test_import_dry_run(self, mock_memory_manager, temp_dir, sample_json_data):
        """Test dry run import"""
        importer = MemoryImporter(mock_memory_manager)
        
        # Create JSON file
        json_file = Path(temp_dir) / "test.json"
        with open(json_file, 'w') as f:
            json.dump(sample_json_data, f)
        
        options = ImportOptions(
            format=ImportFormat.JSON,
            dry_run=True,
            create_backup=False
        )
        result = importer.import_data(str(json_file), options)
        
        assert result.success is True
        assert result.import_summary['dry_run'] is True
        
        # Should not have called actual import methods
        mock_memory_manager.insert_record.assert_not_called()
    
    def test_import_with_progress_callback(self, mock_memory_manager, temp_dir, sample_json_data):
        """Test import with progress callback"""
        importer = MemoryImporter(mock_memory_manager)
        
        # Create JSON file
        json_file = Path(temp_dir) / "test.json"
        with open(json_file, 'w') as f:
            json.dump(sample_json_data, f)
        
        progress_calls = []
        
        def progress_callback(processed, total, operation):
            progress_calls.append((processed, total, operation))
        
        options = ImportOptions(
            format=ImportFormat.JSON,
            progress_callback=progress_callback,
            create_backup=False
        )
        result = importer.import_data(str(json_file), options)
        
        assert result.success is True
        assert len(progress_calls) > 0
    
    def test_import_error_handling(self, mock_memory_manager, temp_dir):
        """Test import error handling"""
        importer = MemoryImporter(mock_memory_manager)
        
        # Create invalid JSON file
        json_file = Path(temp_dir) / "invalid.json"
        with open(json_file, 'w') as f:
            f.write("invalid json content")
        
        options = ImportOptions(format=ImportFormat.JSON, create_backup=False)
        result = importer.import_data(str(json_file), options)
        
        assert result.success is False
        assert len(result.errors) > 0
    
    def test_import_ignore_errors(self, mock_memory_manager, temp_dir, sample_json_data):
        """Test import with ignore errors option"""
        importer = MemoryImporter(mock_memory_manager)
        
        # Mock insert_record to raise exception
        mock_memory_manager.insert_record.side_effect = Exception("Test error")
        
        # Create JSON file
        json_file = Path(temp_dir) / "test.json"
        with open(json_file, 'w') as f:
            json.dump(sample_json_data, f)
        
        options = ImportOptions(
            format=ImportFormat.JSON,
            ignore_errors=True,
            create_backup=False
        )
        result = importer.import_data(str(json_file), options)
        
        assert result.success is True
        assert len(result.warnings) > 0


class TestPublicAPIFunctions:
    """Test public API functions"""
    
    def test_import_memory_data(self, mock_memory_manager, temp_dir, sample_json_data):
        """Test import_memory_data function"""
        # Create JSON file
        json_file = Path(temp_dir) / "test.json"
        with open(json_file, 'w') as f:
            json.dump(sample_json_data, f)
        
        result = import_memory_data(str(json_file), mock_memory_manager)
        
        assert isinstance(result, ImportResult)
        assert result.format_detected == ImportFormat.JSON
    
    def test_import_user_backup(self, mock_memory_manager, temp_dir, sample_json_data):
        """Test import_user_backup function"""
        # Create JSON file
        json_file = Path(temp_dir) / "test.json"
        with open(json_file, 'w') as f:
            json.dump(sample_json_data, f)
        
        result = import_user_backup(
            str(json_file),
            mock_memory_manager,
            "user1",
            ConflictResolution.SKIP
        )
        
        assert isinstance(result, ImportResult)
        assert result.format_detected == ImportFormat.JSON
    
    def test_migrate_from_database(self, mock_memory_manager, temp_dir):
        """Test migrate_from_database function"""
        # Create SQLite file
        sqlite_file = Path(temp_dir) / "source.db"
        conn = sqlite3.connect(sqlite_file)
        conn.execute("CREATE TABLE users (user_id TEXT PRIMARY KEY, username TEXT)")
        conn.execute("INSERT INTO users (user_id, username) VALUES ('user1', 'testuser')")
        conn.commit()
        conn.close()
        
        result = migrate_from_database(str(sqlite_file), mock_memory_manager)
        
        assert isinstance(result, ImportResult)
        assert result.format_detected == ImportFormat.SQLITE
    
    def test_restore_from_backup(self, mock_memory_manager, temp_dir):
        """Test restore_from_backup function"""
        # Create SQLite backup file
        backup_file = Path(temp_dir) / "backup.db"
        conn = sqlite3.connect(backup_file)
        conn.execute("CREATE TABLE users (user_id TEXT PRIMARY KEY, username TEXT)")
        conn.execute("INSERT INTO users (user_id, username) VALUES ('user1', 'testuser')")
        conn.commit()
        conn.close()
        
        result = restore_from_backup(str(backup_file), mock_memory_manager)
        
        assert isinstance(result, ImportResult)
        assert result.format_detected == ImportFormat.SQLITE


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_file(self, mock_memory_manager, temp_dir):
        """Test import of empty file"""
        importer = MemoryImporter(mock_memory_manager)
        
        # Create empty JSON file
        json_file = Path(temp_dir) / "empty.json"
        with open(json_file, 'w') as f:
            f.write("{}")
        
        options = ImportOptions(format=ImportFormat.JSON, create_backup=False)
        result = importer.import_data(str(json_file), options)
        
        assert result.success is True
        assert result.records_imported == 0
    
    def test_nonexistent_file(self, mock_memory_manager):
        """Test import of non-existent file"""
        importer = MemoryImporter(mock_memory_manager)
        
        options = ImportOptions(format=ImportFormat.JSON, create_backup=False)
        result = importer.import_data("/nonexistent/file.json", options)
        
        assert result.success is False
        assert len(result.errors) > 0
    
    def test_unsupported_format(self, mock_memory_manager, temp_dir):
        """Test import with unsupported format"""
        importer = MemoryImporter(mock_memory_manager)
        
        # Create text file
        text_file = Path(temp_dir) / "test.txt"
        with open(text_file, 'w') as f:
            f.write("Not a supported format")
        
        # Mock format detection to return unsupported format
        with patch.object(importer, 'detect_format') as mock_detect:
            mock_detect.return_value = "unsupported"
            
            options = ImportOptions(format=ImportFormat.AUTO, create_backup=False)
            result = importer.import_data(str(text_file), options)
            
            assert result.success is False
            assert len(result.errors) > 0
    
    def test_table_not_exists(self, mock_memory_manager, temp_dir, sample_json_data):
        """Test import when table doesn't exist"""
        importer = MemoryImporter(mock_memory_manager)
        
        # Mock table_exists to return False
        mock_memory_manager.table_exists.return_value = False
        
        # Create JSON file
        json_file = Path(temp_dir) / "test.json"
        with open(json_file, 'w') as f:
            json.dump(sample_json_data, f)
        
        options = ImportOptions(format=ImportFormat.JSON, create_backup=False)
        result = importer.import_data(str(json_file), options)
        
        assert result.success is True
        assert len(result.warnings) > 0
        assert any("does not exist" in warning for warning in result.warnings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 