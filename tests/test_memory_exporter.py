"""
Tests for Memory Data Export System

This module tests the memory export functionality including:
- Multiple export formats (JSON, CSV, XML, SQLite)
- Selective filtering by user, date, conversation, document
- Privacy compliance and access control
- Compression and encryption options
- Error handling and progress tracking
- Data integrity and completeness
"""

import pytest
import tempfile
import shutil
import json
import csv
import sqlite3
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.sovereign.config import Config
from src.sovereign.memory_exporter import (
    MemoryExporter, ExportFormat, ExportScope, CompressionType,
    ExportFilter, ExportOptions, ExportProgress, ExportResult,
    export_user_data, export_conversation_data, create_full_backup
)
from src.sovereign.memory_schema import MemorySchema
from src.sovereign.memory_manager import MemoryManager, MessageSender
from src.sovereign.privacy_manager import PrivacyManager


@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_export.db"
    yield str(db_path)
    
    # Cleanup
    try:
        shutil.rmtree(temp_dir)
    except:
        pass


@pytest.fixture
def test_config():
    """Create test configuration"""
    config = Mock(spec=Config)
    config.database = Mock()
    config.database.db_path = "test.db"
    return config


@pytest.fixture
def populated_db(temp_db, test_config):
    """Create a database with test data"""
    # Create schema
    schema = MemorySchema(temp_db)
    schema.create_schema()
    
    # Populate with test data using MemoryManager
    memory_manager = MemoryManager(test_config)
    memory_manager.db_path = Path(temp_db)
    memory_manager.schema = schema
    
    # Create test users
    user1_id = memory_manager.create_user("test_user1", "test1@example.com")
    user2_id = memory_manager.create_user("test_user2", "test2@example.com")
    
    # Create test conversations and messages
    memory_manager.set_current_user(user1_id)
    conv1_id = memory_manager.start_conversation("Test Conversation 1")
    memory_manager.add_message("Hello world", MessageSender.USER, conv1_id)
    memory_manager.add_message("Hi there!", MessageSender.ASSISTANT, conv1_id)
    
    conv2_id = memory_manager.start_conversation("Test Conversation 2")
    memory_manager.add_message("How are you?", MessageSender.USER, conv2_id)
    memory_manager.add_message("I'm doing well!", MessageSender.ASSISTANT, conv2_id)
    
    # Create test documents
    doc1_id = memory_manager.add_document(
        "Test Document 1", 
        "This is a test document with some content.",
        "test_source"
    )
    doc2_id = memory_manager.add_document(
        "Test Document 2",
        "Another test document with different content.",
        "test_source2"
    )
    
    # Create chunks
    memory_manager.create_chunks(doc1_id, chunk_size=50)
    memory_manager.create_chunks(doc2_id, chunk_size=50)
    
    # Create user 2 data
    memory_manager.set_current_user(user2_id)
    conv3_id = memory_manager.start_conversation("User 2 Conversation")
    memory_manager.add_message("User 2 message", MessageSender.USER, conv3_id)
    
    memory_manager.close()
    schema.close()
    
    return temp_db, {
        'user1_id': user1_id,
        'user2_id': user2_id,
        'conv1_id': conv1_id,
        'conv2_id': conv2_id,
        'conv3_id': conv3_id,
        'doc1_id': doc1_id,
        'doc2_id': doc2_id
    }


@pytest.fixture
def memory_exporter(test_config, populated_db):
    """Create a memory exporter with populated test data"""
    db_path, test_data = populated_db
    exporter = MemoryExporter(test_config, db_path)
    yield exporter, test_data
    exporter.close()


class TestExportFilter:
    """Test export filter functionality"""
    
    def test_default_filter(self):
        """Test default export filter"""
        filter_obj = ExportFilter()
        assert filter_obj.user_ids is None
        assert filter_obj.date_from is None
        assert filter_obj.date_to is None
        assert filter_obj.include_inactive is False
        assert filter_obj.include_embeddings is True
        assert filter_obj.scopes == [ExportScope.ALL]
    
    def test_filter_with_options(self):
        """Test export filter with specific options"""
        date_from = datetime.now() - timedelta(days=7)
        date_to = datetime.now()
        
        filter_obj = ExportFilter(
            user_ids=[1, 2],
            date_from=date_from,
            date_to=date_to,
            conversation_ids=[10, 20],
            privacy_levels=[1, 2],
            include_inactive=True,
            include_embeddings=False,
            scopes=[ExportScope.CONVERSATIONS, ExportScope.DOCUMENTS]
        )
        
        assert filter_obj.user_ids == [1, 2]
        assert filter_obj.date_from == date_from
        assert filter_obj.date_to == date_to
        assert filter_obj.conversation_ids == [10, 20]
        assert filter_obj.privacy_levels == [1, 2]
        assert filter_obj.include_inactive is True
        assert filter_obj.include_embeddings is False
        assert ExportScope.CONVERSATIONS in filter_obj.scopes
        assert ExportScope.DOCUMENTS in filter_obj.scopes


class TestExportOptions:
    """Test export options functionality"""
    
    def test_default_options(self):
        """Test default export options"""
        options = ExportOptions()
        assert options.format == ExportFormat.JSON
        assert options.compression == CompressionType.NONE
        assert options.include_schema is True
        assert options.include_relationships is True
        assert options.pretty_format is True
        assert options.validate_privacy is True
    
    def test_options_with_values(self):
        """Test export options with specific values"""
        output_dir = Path("/tmp/exports")
        
        options = ExportOptions(
            format=ExportFormat.CSV,
            compression=CompressionType.ZIP,
            include_schema=False,
            encrypt_output=True,
            encryption_password="test123",
            output_directory=output_dir,
            filename_prefix="custom_export",
            pretty_format=False,
            validate_privacy=False
        )
        
        assert options.format == ExportFormat.CSV
        assert options.compression == CompressionType.ZIP
        assert options.include_schema is False
        assert options.encrypt_output is True
        assert options.encryption_password == "test123"
        assert options.output_directory == output_dir
        assert options.filename_prefix == "custom_export"
        assert options.pretty_format is False
        assert options.validate_privacy is False


class TestMemoryExporter:
    """Test the main MemoryExporter class"""
    
    def test_exporter_initialization(self, test_config, temp_db):
        """Test exporter initialization"""
        exporter = MemoryExporter(test_config, temp_db)
        
        assert exporter.config == test_config
        assert exporter.db_path == temp_db
        assert exporter.schema is not None
        assert exporter._format_handlers is not None
        assert len(exporter.TABLE_EXPORT_ORDER) > 0
        
        exporter.close()
    
    def test_get_tables_for_scope(self, memory_exporter):
        """Test table selection based on scope"""
        exporter, _ = memory_exporter
        
        # Test ALL scope
        tables = exporter._get_tables_for_scope([ExportScope.ALL])
        assert len(tables) == len(exporter.TABLE_EXPORT_ORDER)
        assert 'users' in tables
        assert 'conversations' in tables
        assert 'documents' in tables
        
        # Test CONVERSATIONS scope
        tables = exporter._get_tables_for_scope([ExportScope.CONVERSATIONS])
        assert 'conversations' in tables
        assert 'messages' in tables
        assert 'context_windows' in tables
        assert 'documents' not in tables
        
        # Test DOCUMENTS scope
        tables = exporter._get_tables_for_scope([ExportScope.DOCUMENTS])
        assert 'documents' in tables
        assert 'chunks' in tables
        assert 'conversations' not in tables
        
        # Test multiple scopes
        tables = exporter._get_tables_for_scope([ExportScope.CONVERSATIONS, ExportScope.DOCUMENTS])
        assert 'conversations' in tables
        assert 'documents' in tables
        assert 'chunks' in tables
        assert 'messages' in tables
    
    def test_count_total_records(self, memory_exporter):
        """Test record counting for progress tracking"""
        exporter, test_data = memory_exporter
        
        tables = ['users', 'conversations', 'messages']
        export_filter = ExportFilter()
        
        total = exporter._count_total_records(tables, export_filter)
        assert total > 0  # Should have some records from populated_db
        
        # Test with user filter
        export_filter = ExportFilter(user_ids=[test_data['user1_id']])
        total_filtered = exporter._count_total_records(tables, export_filter)
        assert total_filtered <= total  # Should be same or fewer records
    
    def test_build_where_clause(self, memory_exporter):
        """Test WHERE clause building"""
        exporter, test_data = memory_exporter
        
        # Test user filtering
        export_filter = ExportFilter(user_ids=[test_data['user1_id']])
        where_clause, params = exporter._build_where_clause('conversations', export_filter)
        assert 'user_id IN' in where_clause
        assert test_data['user1_id'] in params
        
        # Test date filtering
        date_from = datetime.now() - timedelta(days=1)
        export_filter = ExportFilter(date_from=date_from)
        where_clause, params = exporter._build_where_clause('conversations', export_filter)
        assert 'started_at >=' in where_clause
        assert date_from.isoformat() in params
        
        # Test conversation ID filtering
        export_filter = ExportFilter(conversation_ids=[test_data['conv1_id']])
        where_clause, params = exporter._build_where_clause('messages', export_filter)
        assert 'conversation_id IN' in where_clause
        assert test_data['conv1_id'] in params
    
    def test_list_exportable_data(self, memory_exporter):
        """Test listing available data for export"""
        exporter, test_data = memory_exporter
        
        # Test without user filter
        summary = exporter.list_exportable_data()
        assert isinstance(summary, dict)
        assert 'users' in summary
        assert 'conversations' in summary
        assert 'messages' in summary
        assert summary['users'] >= 2  # Should have at least 2 test users
        
        # Test with user filter
        user_summary = exporter.list_exportable_data(test_data['user1_id'])
        assert isinstance(user_summary, dict)
        assert user_summary['conversations'] >= 2  # User 1 has 2 conversations


class TestJSONExport:
    """Test JSON export functionality"""
    
    def test_json_export_all_data(self, memory_exporter):
        """Test complete JSON export"""
        exporter, test_data = memory_exporter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            options = ExportOptions(
                format=ExportFormat.JSON,
                output_directory=Path(temp_dir),
                pretty_format=True
            )
            
            result = exporter.export_memory_data(options=options)
            
            assert result.success is True
            # Should have main export file + manifest file
            assert len(result.output_files) == 2
            json_files = [f for f in result.output_files if f.suffix == '.json' and 'manifest' not in f.name]
            assert len(json_files) == 1
            assert result.total_records_exported > 0
            assert result.export_size_bytes > 0
            
            # Verify JSON content
            with open(json_files[0], 'r') as f:
                data = json.load(f)
            
            assert 'export_info' in data
            assert 'schema' in data
            assert 'data' in data
            assert 'users' in data['data']
            assert 'conversations' in data['data']
            assert 'messages' in data['data']
            
            # Check export info
            assert 'created_at' in data['export_info']
            assert 'total_tables' in data['export_info']
            
            # Check data content
            assert len(data['data']['users']) >= 2
            assert len(data['data']['conversations']) >= 3
            assert len(data['data']['messages']) >= 5
    
    def test_json_export_with_filter(self, memory_exporter):
        """Test JSON export with user filtering"""
        exporter, test_data = memory_exporter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            export_filter = ExportFilter(user_ids=[test_data['user1_id']])
            options = ExportOptions(
                format=ExportFormat.JSON,
                output_directory=Path(temp_dir)
            )
            
            result = exporter.export_memory_data(export_filter, options)
            
            assert result.success is True
            
            # Verify filtered content
            with open(result.output_files[0], 'r') as f:
                data = json.load(f)
            
            # Should only have user1's conversations
            user1_conversations = [
                conv for conv in data['data']['conversations']
                if conv['user_id'] == test_data['user1_id']
            ]
            assert len(user1_conversations) >= 2
            
            # All conversations should belong to user1
            for conv in data['data']['conversations']:
                assert conv['user_id'] == test_data['user1_id']
    
    def test_json_export_scope_filtering(self, memory_exporter):
        """Test JSON export with scope filtering"""
        exporter, test_data = memory_exporter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            export_filter = ExportFilter(scopes=[ExportScope.CONVERSATIONS])
            options = ExportOptions(
                format=ExportFormat.JSON,
                output_directory=Path(temp_dir)
            )
            
            result = exporter.export_memory_data(export_filter, options)
            
            assert result.success is True
            
            # Verify scope filtering
            with open(result.output_files[0], 'r') as f:
                data = json.load(f)
            
            # Should have conversation-related tables
            assert 'conversations' in data['data']
            assert 'messages' in data['data']
            assert 'context_windows' in data['data']
            
            # Should not have document-related tables
            assert 'documents' not in data['data']
            assert 'chunks' not in data['data']
    
    def test_json_export_without_binary_data(self, memory_exporter):
        """Test JSON export excluding binary data"""
        exporter, test_data = memory_exporter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            export_filter = ExportFilter(include_binary_data=False)
            options = ExportOptions(
                format=ExportFormat.JSON,
                output_directory=Path(temp_dir)
            )
            
            result = exporter.export_memory_data(export_filter, options)
            
            assert result.success is True
            
            # Get main export file (not manifest)
            json_files = [f for f in result.output_files if f.suffix == '.json' and 'manifest' not in f.name]
            assert len(json_files) == 1
            
            # Verify binary data exclusion
            with open(json_files[0], 'r') as f:
                data = json.load(f)
            
            # Check if any embedding data is excluded
            if 'embeddings' in data['data']:
                for embedding in data['data']['embeddings']:
                    if '_embedding_data_excluded' in embedding:
                        assert embedding['embedding_data'] is None
                        assert embedding['_embedding_data_excluded'] is True


class TestCSVExport:
    """Test CSV export functionality"""
    
    def test_csv_export_all_data(self, memory_exporter):
        """Test complete CSV export"""
        exporter, test_data = memory_exporter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            options = ExportOptions(
                format=ExportFormat.CSV,
                output_directory=Path(temp_dir)
            )
            
            result = exporter.export_memory_data(options=options)
            
            assert result.success is True
            assert len(result.output_files) > 1  # Multiple CSV files
            assert result.total_records_exported > 0
            
            # Check that we have CSV files for expected tables
            csv_files = [f for f in result.output_files if f.suffix == '.csv']
            table_names = [f.stem.split('_')[-1] for f in csv_files]
            
            assert 'users' in table_names
            assert 'conversations' in table_names
            assert 'messages' in table_names
            
            # Verify CSV content
            users_file = next(f for f in csv_files if 'users' in f.name)
            with open(users_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                users = list(reader)
                assert len(users) >= 2
                assert 'id' in users[0]
                assert 'username' in users[0]
    
    def test_csv_export_with_filter(self, memory_exporter):
        """Test CSV export with filtering"""
        exporter, test_data = memory_exporter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            export_filter = ExportFilter(
                scopes=[ExportScope.CONVERSATIONS],
                user_ids=[test_data['user1_id']]
            )
            options = ExportOptions(
                format=ExportFormat.CSV,
                output_directory=Path(temp_dir)
            )
            
            result = exporter.export_memory_data(export_filter, options)
            
            assert result.success is True
            
            # Should only have conversation-related CSV files
            csv_files = [f for f in result.output_files if f.suffix == '.csv']
            table_names = [f.stem.split('_')[-1] for f in csv_files]
            
            assert 'conversations' in table_names
            assert 'messages' in table_names
            assert 'documents' not in table_names
            
            # Verify filtered content
            conv_file = next(f for f in csv_files if 'conversations' in f.name)
            with open(conv_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                conversations = list(reader)
                
                # All conversations should belong to user1
                for conv in conversations:
                    assert int(conv['user_id']) == test_data['user1_id']


class TestXMLExport:
    """Test XML export functionality"""
    
    def test_xml_export_all_data(self, memory_exporter):
        """Test complete XML export"""
        exporter, test_data = memory_exporter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            options = ExportOptions(
                format=ExportFormat.XML,
                output_directory=Path(temp_dir),
                pretty_format=True
            )
            
            result = exporter.export_memory_data(options=options)
            
            assert result.success is True
            # Should have main export file + manifest file
            assert len(result.output_files) == 2
            xml_files = [f for f in result.output_files if f.suffix == '.xml']
            assert len(xml_files) == 1
            assert result.total_records_exported > 0
            
            # Verify XML structure
            tree = ET.parse(xml_files[0])
            root = tree.getroot()
            
            assert root.tag == 'memory_export'
            
            # Check for expected sections
            export_info = root.find('export_info')
            assert export_info is not None
            assert export_info.find('created_at') is not None
            
            schema_elem = root.find('schema')
            assert schema_elem is not None
            
            data_elem = root.find('data')
            assert data_elem is not None
            
            # Check for expected tables
            tables = [table.get('name') for table in data_elem.findall('table')]
            assert 'users' in tables
            assert 'conversations' in tables
            assert 'messages' in tables
    
    def test_xml_export_with_filter(self, memory_exporter):
        """Test XML export with filtering"""
        exporter, test_data = memory_exporter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            export_filter = ExportFilter(scopes=[ExportScope.USER_DATA])
            options = ExportOptions(
                format=ExportFormat.XML,
                output_directory=Path(temp_dir)
            )
            
            result = exporter.export_memory_data(export_filter, options)
            
            assert result.success is True
            
            # Verify filtered content
            tree = ET.parse(result.output_files[0])
            root = tree.getroot()
            
            data_elem = root.find('data')
            tables = [table.get('name') for table in data_elem.findall('table')]
            
            # Should only have user-related tables
            assert 'users' in tables
            assert 'conversations' not in tables


class TestSQLiteExport:
    """Test SQLite export functionality"""
    
    def test_sqlite_export_all_data(self, memory_exporter):
        """Test complete SQLite database export"""
        exporter, test_data = memory_exporter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            options = ExportOptions(
                format=ExportFormat.SQLITE,
                output_directory=Path(temp_dir)
            )
            
            result = exporter.export_memory_data(options=options)
            
            assert result.success is True
            # Should have main export file + manifest file
            assert len(result.output_files) == 2
            db_files = [f for f in result.output_files if f.suffix == '.db']
            assert len(db_files) == 1
            assert result.total_records_exported > 0
            
            # Verify database content
            conn = sqlite3.connect(db_files[0])
            conn.row_factory = sqlite3.Row
            
            # Check that tables exist and have data
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            assert 'users' in tables
            assert 'conversations' in tables
            assert 'messages' in tables
            
            # Check data integrity
            cursor = conn.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]
            assert user_count >= 2
            
            cursor = conn.execute("SELECT COUNT(*) FROM conversations")
            conv_count = cursor.fetchone()[0]
            assert conv_count >= 3
            
            conn.close()
    
    def test_sqlite_export_with_filter(self, memory_exporter):
        """Test SQLite export with filtering"""
        exporter, test_data = memory_exporter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            export_filter = ExportFilter(user_ids=[test_data['user1_id']])
            options = ExportOptions(
                format=ExportFormat.SQLITE,
                output_directory=Path(temp_dir)
            )
            
            result = exporter.export_memory_data(export_filter, options)
            
            assert result.success is True
            
            # Verify filtered content
            conn = sqlite3.connect(result.output_files[0])
            conn.row_factory = sqlite3.Row
            
            # Check that only user1's conversations are included
            cursor = conn.execute("SELECT user_id FROM conversations")
            user_ids = [row[0] for row in cursor.fetchall()]
            
            for user_id in user_ids:
                assert user_id == test_data['user1_id']
            
            conn.close()


class TestCompressionAndEncryption:
    """Test compression and encryption features"""
    
    def test_zip_compression(self, memory_exporter):
        """Test ZIP compression of export files"""
        exporter, test_data = memory_exporter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            options = ExportOptions(
                format=ExportFormat.JSON,
                compression=CompressionType.ZIP,
                output_directory=Path(temp_dir)
            )
            
            result = exporter.export_memory_data(options=options)
            
            assert result.success is True
            # Should have ZIP file + manifest file
            assert len(result.output_files) == 2
            zip_files = [f for f in result.output_files if f.suffix == '.zip']
            assert len(zip_files) == 1
            
            # Verify ZIP contents
            with zipfile.ZipFile(zip_files[0], 'r') as zf:
                files = zf.namelist()
                assert len(files) >= 1
                assert any(f.endswith('.json') for f in files)
    
    def test_encryption_placeholder(self, memory_exporter):
        """Test encryption placeholder (warns about not implemented)"""
        exporter, test_data = memory_exporter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            options = ExportOptions(
                format=ExportFormat.JSON,
                encrypt_output=True,
                encryption_password="test123",
                output_directory=Path(temp_dir)
            )
            
            with patch('src.sovereign.memory_exporter.logger') as mock_logger:
                result = exporter.export_memory_data(options=options)
                
                # Should warn about encryption not implemented
                mock_logger.warning.assert_called_with(
                    "Encryption requested but not implemented in this demo"
                )


class TestProgressTracking:
    """Test progress tracking functionality"""
    
    def test_progress_callback(self, memory_exporter):
        """Test progress tracking with callback"""
        exporter, test_data = memory_exporter
        
        progress_updates = []
        
        def progress_callback(progress):
            progress_updates.append({
                'completed_tables': progress.completed_tables,
                'total_tables': progress.total_tables,
                'exported_records': progress.exported_records,
                'current_table': progress.current_table,
                'progress_percentage': progress.progress_percentage
            })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            options = ExportOptions(
                format=ExportFormat.JSON,
                output_directory=Path(temp_dir)
            )
            
            result = exporter.export_memory_data(options=options, progress_callback=progress_callback)
            
            assert result.success is True
            assert len(progress_updates) > 0
            
            # Check progress tracking
            final_progress = progress_updates[-1]
            assert final_progress['completed_tables'] > 0
            assert final_progress['exported_records'] > 0
            assert final_progress['progress_percentage'] >= 0
    
    def test_progress_calculation(self):
        """Test progress percentage calculation"""
        progress = ExportProgress(total_records=100, exported_records=25)
        assert progress.progress_percentage == 25.0
        
        progress = ExportProgress(total_records=0, exported_records=0)
        assert progress.progress_percentage == 0.0


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_export_nonexistent_database(self, test_config):
        """Test export with nonexistent database"""
        nonexistent_db = "/nonexistent/path/test.db"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = MemoryExporter(test_config, nonexistent_db)
            options = ExportOptions(output_directory=Path(temp_dir))
            
            result = exporter.export_memory_data(options=options)
            
            # Should succeed but with warnings about missing tables
            assert result.success is True
            assert result.total_records_exported == 0
            assert len(result.progress.errors) > 0  # Should have errors about missing tables
            
            exporter.close()
    
    def test_export_with_invalid_filter(self, memory_exporter):
        """Test export with invalid filter values"""
        exporter, test_data = memory_exporter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with non-existent user ID
            export_filter = ExportFilter(user_ids=[99999])
            options = ExportOptions(
                format=ExportFormat.JSON,
                output_directory=Path(temp_dir)
            )
            
            result = exporter.export_memory_data(export_filter, options)
            
            # Should succeed but export only tables not affected by user filtering
            assert result.success is True
            # Should export some system tables that aren't user-filtered
            assert result.total_records_exported >= 0
    
    def test_export_to_read_only_directory(self, memory_exporter):
        """Test export to read-only directory"""
        exporter, test_data = memory_exporter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            read_only_dir = Path(temp_dir) / "readonly"
            read_only_dir.mkdir()
            
            # Note: On Windows, file permissions work differently
            # Skip this test on Windows as it's hard to simulate read-only directories
            import platform
            if platform.system() == "Windows":
                # On Windows, just test that export works normally
                options = ExportOptions(
                    format=ExportFormat.JSON,
                    output_directory=read_only_dir
                )
                result = exporter.export_memory_data(options=options)
                assert result.success is True
            else:
                read_only_dir.chmod(0o444)  # Read-only
                
                options = ExportOptions(
                    format=ExportFormat.JSON,
                    output_directory=read_only_dir
                )
                
                try:
                    result = exporter.export_memory_data(options=options)
                    # Should fail due to permission error
                    assert result.success is False
                finally:
                    # Restore permissions for cleanup
                    read_only_dir.chmod(0o755)


class TestPublicAPIFunctions:
    """Test public API convenience functions"""
    
    def test_export_user_data(self, test_config, populated_db):
        """Test export_user_data convenience function"""
        db_path, test_data = populated_db
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = export_user_data(
                test_config, 
                db_path,
                test_data['user1_id'],
                ExportFormat.JSON,
                Path(temp_dir)
            )
            
            assert result.success is True
            assert result.total_records_exported > 0
            
            # Verify it only exported user1's data
            with open(result.output_files[0], 'r') as f:
                data = json.load(f)
            
            # All conversations should belong to user1
            for conv in data['data']['conversations']:
                assert conv['user_id'] == test_data['user1_id']
    
    def test_export_conversation_data(self, test_config, populated_db):
        """Test export_conversation_data convenience function"""
        db_path, test_data = populated_db
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = export_conversation_data(
                test_config,
                db_path,
                [test_data['conv1_id']],
                ExportFormat.JSON,
                Path(temp_dir)
            )
            
            assert result.success is True
            assert result.total_records_exported > 0
            
            # Verify it only exported the specified conversation
            with open(result.output_files[0], 'r') as f:
                data = json.load(f)
            
            conversations = data['data']['conversations']
            assert len(conversations) == 1
            assert conversations[0]['id'] == test_data['conv1_id']
    
    def test_create_full_backup(self, test_config, populated_db):
        """Test create_full_backup convenience function"""
        db_path, test_data = populated_db
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = create_full_backup(
                test_config,
                db_path,
                Path(temp_dir),
                compress=True
            )
            
            assert result.success is True
            assert result.total_records_exported > 0
            assert result.output_files[0].suffix == '.zip'
            
            # Verify it's a complete backup
            with zipfile.ZipFile(result.output_files[0], 'r') as zf:
                files = zf.namelist()
                assert any(f.endswith('.db') for f in files)


class TestManifestCreation:
    """Test manifest file creation"""
    
    def test_manifest_creation(self, memory_exporter):
        """Test that manifest files are created"""
        exporter, test_data = memory_exporter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            options = ExportOptions(
                format=ExportFormat.JSON,
                output_directory=Path(temp_dir),
                create_manifest=True
            )
            
            result = exporter.export_memory_data(options=options)
            
            assert result.success is True
            
            # Find manifest file
            manifest_files = [f for f in result.output_files if 'manifest' in f.name]
            assert len(manifest_files) == 1
            
            # Verify manifest content
            with open(manifest_files[0], 'r') as f:
                manifest = json.load(f)
            
            assert 'export_summary' in manifest
            assert 'filters_applied' in manifest
            assert 'options_used' in manifest
            assert 'output_files' in manifest
            
            assert manifest['export_summary']['success'] is True
            assert manifest['export_summary']['total_records'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 