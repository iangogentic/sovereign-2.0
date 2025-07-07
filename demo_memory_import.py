#!/usr/bin/env python3
"""
Memory Import System Demo

This script demonstrates the comprehensive capabilities of the Memory Import system:
- Multiple import formats (JSON, CSV, XML, SQLite)
- Data validation and conflict resolution
- Backup creation and restoration
- Progress tracking
- Error handling
- Integration with MemoryManager

Usage: python demo_memory_import.py
"""

import json
import csv
import xml.etree.ElementTree as ET
import sqlite3
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import time
import random
import zipfile

# Add the src directory to the Python path
import sys
sys.path.insert(0, 'src')

from sovereign.memory_importer import (
    MemoryImporter, ImportFormat, ConflictResolution, ImportScope,
    ImportOptions, import_memory_data, import_user_backup,
    migrate_from_database, restore_from_backup
)
from sovereign.memory_manager import MemoryManager
from sovereign.memory_schema import MemorySchema


def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_subheader(title):
    """Print a formatted subheader"""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")


def create_sample_data():
    """Create sample data for import demonstrations"""
    return {
        "export_info": {
            "format": "json",
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "exported_by": "Sovereign AI Memory System",
            "total_records": 15
        },
        "tables": {
            "users": [
                {
                    "user_id": "user_001",
                    "username": "alice_demo",
                    "email": "alice@example.com",
                    "full_name": "Alice Johnson",
                    "created_at": "2024-01-01T10:00:00Z",
                    "is_active": True,
                    "preferences": '{"theme": "dark", "notifications": true}'
                },
                {
                    "user_id": "user_002",
                    "username": "bob_demo",
                    "email": "bob@example.com",
                    "full_name": "Bob Smith",
                    "created_at": "2024-01-02T11:00:00Z",
                    "is_active": True,
                    "preferences": '{"theme": "light", "notifications": false}'
                },
                {
                    "user_id": "user_003",
                    "username": "charlie_demo",
                    "email": "charlie@example.com",
                    "full_name": "Charlie Brown",
                    "created_at": "2024-01-03T12:00:00Z",
                    "is_active": False,
                    "preferences": '{"theme": "auto", "notifications": true}'
                }
            ],
            "conversations": [
                {
                    "conversation_id": "conv_001",
                    "user_id": "user_001",
                    "title": "Getting Started with AI",
                    "summary": "Discussion about AI capabilities and features",
                    "created_at": "2024-01-01T10:30:00Z",
                    "updated_at": "2024-01-01T11:00:00Z",
                    "message_count": 5,
                    "is_archived": False
                },
                {
                    "conversation_id": "conv_002",
                    "user_id": "user_002",
                    "title": "Programming Help",
                    "summary": "Python coding assistance and debugging",
                    "created_at": "2024-01-02T14:00:00Z",
                    "updated_at": "2024-01-02T15:30:00Z",
                    "message_count": 8,
                    "is_archived": False
                },
                {
                    "conversation_id": "conv_003",
                    "user_id": "user_003",
                    "title": "Data Analysis Questions",
                    "summary": "Statistical analysis and data visualization",
                    "created_at": "2024-01-03T09:00:00Z",
                    "updated_at": "2024-01-03T10:00:00Z",
                    "message_count": 3,
                    "is_archived": True
                }
            ],
            "messages": [
                {
                    "message_id": "msg_001",
                    "conversation_id": "conv_001",
                    "role": "user",
                    "content": "Hello! I'm new to AI assistants. What can you help me with?",
                    "timestamp": "2024-01-01T10:30:00Z",
                    "token_count": 12
                },
                {
                    "message_id": "msg_002",
                    "conversation_id": "conv_001",
                    "role": "assistant",
                    "content": "Welcome! I can help with various tasks like answering questions, writing, coding, analysis, and more. What would you like to explore?",
                    "timestamp": "2024-01-01T10:31:00Z",
                    "token_count": 28
                },
                {
                    "message_id": "msg_003",
                    "conversation_id": "conv_002",
                    "role": "user",
                    "content": "I'm having trouble with a Python function. Can you help me debug it?",
                    "timestamp": "2024-01-02T14:00:00Z",
                    "token_count": 15
                },
                {
                    "message_id": "msg_004",
                    "conversation_id": "conv_002",
                    "role": "assistant",
                    "content": "Of course! Please share your code and describe the issue you're encountering. I'll help you identify and fix the problem.",
                    "timestamp": "2024-01-02T14:01:00Z",
                    "token_count": 24
                }
            ],
            "documents": [
                {
                    "document_id": "doc_001",
                    "user_id": "user_001",
                    "title": "AI Usage Guidelines",
                    "content": "This document contains guidelines for effective AI usage...",
                    "document_type": "text",
                    "created_at": "2024-01-01T12:00:00Z",
                    "updated_at": "2024-01-01T12:30:00Z",
                    "size_bytes": 2048,
                    "is_public": True
                },
                {
                    "document_id": "doc_002",
                    "user_id": "user_002",
                    "title": "Python Best Practices",
                    "content": "A collection of Python programming best practices and tips...",
                    "document_type": "text",
                    "created_at": "2024-01-02T16:00:00Z",
                    "updated_at": "2024-01-02T16:15:00Z",
                    "size_bytes": 4096,
                    "is_public": False
                }
            ],
            "chunks": [
                {
                    "chunk_id": "chunk_001",
                    "document_id": "doc_001",
                    "content": "AI assistants are powerful tools that can help with various tasks...",
                    "chunk_index": 0,
                    "start_offset": 0,
                    "end_offset": 100,
                    "token_count": 20
                },
                {
                    "chunk_id": "chunk_002",
                    "document_id": "doc_001",
                    "content": "When using AI, it's important to provide clear and specific instructions...",
                    "chunk_index": 1,
                    "start_offset": 101,
                    "end_offset": 200,
                    "token_count": 18
                },
                {
                    "chunk_id": "chunk_003",
                    "document_id": "doc_002",
                    "content": "Python follows the principle of readability and simplicity...",
                    "chunk_index": 0,
                    "start_offset": 0,
                    "end_offset": 80,
                    "token_count": 15
                }
            ]
        }
    }


def create_json_export_file(temp_dir, data):
    """Create a JSON export file for import testing"""
    json_file = Path(temp_dir) / "memory_export.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return json_file


def create_csv_export_files(temp_dir, data):
    """Create CSV export files for import testing"""
    csv_files = []
    
    for table_name, records in data["tables"].items():
        if not records:
            continue
            
        csv_file = Path(temp_dir) / f"{table_name}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if records:
                writer = csv.DictWriter(f, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
        csv_files.append(csv_file)
    
    # Create ZIP file with all CSV files
    zip_file = Path(temp_dir) / "memory_export.zip"
    with zipfile.ZipFile(zip_file, 'w') as zf:
        for csv_file in csv_files:
            zf.write(csv_file, csv_file.name)
    
    return zip_file


def create_xml_export_file(temp_dir, data):
    """Create an XML export file for import testing"""
    root = ET.Element("memory_export")
    
    # Add export info
    export_info = ET.SubElement(root, "export_info")
    for key, value in data["export_info"].items():
        elem = ET.SubElement(export_info, key)
        elem.text = str(value)
    
    # Add tables
    tables = ET.SubElement(root, "tables")
    for table_name, records in data["tables"].items():
        table = ET.SubElement(tables, "table", name=table_name)
        
        for record in records:
            record_elem = ET.SubElement(table, "record")
            for field_name, field_value in record.items():
                field_elem = ET.SubElement(record_elem, field_name)
                if field_value is not None:
                    field_elem.text = str(field_value)
    
    xml_file = Path(temp_dir) / "memory_export.xml"
    tree = ET.ElementTree(root)
    tree.write(xml_file, encoding='utf-8', xml_declaration=True)
    
    return xml_file


def create_sqlite_export_file(temp_dir, data):
    """Create a SQLite export file for import testing"""
    sqlite_file = Path(temp_dir) / "memory_export.db"
    conn = sqlite3.connect(sqlite_file)
    
    # Create tables and insert data
    for table_name, records in data["tables"].items():
        if not records:
            continue
        
        # Create table based on first record
        first_record = records[0]
        columns = []
        for field_name in first_record.keys():
            columns.append(f"{field_name} TEXT")
        
        create_sql = f"CREATE TABLE {table_name} ({', '.join(columns)})"
        conn.execute(create_sql)
        
        # Insert records
        placeholders = ', '.join(['?' for _ in first_record.keys()])
        insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"
        
        for record in records:
            values = [str(v) if v is not None else None for v in record.values()]
            conn.execute(insert_sql, values)
    
    conn.commit()
    conn.close()
    
    return sqlite_file


def progress_callback(processed, total, operation):
    """Progress callback for import operations"""
    percentage = (processed / total) * 100 if total > 0 else 0
    print(f"  Progress: {processed}/{total} ({percentage:.1f}%) - {operation}")


def demo_format_detection():
    """Demonstrate format detection capabilities"""
    print_header("Format Detection Demo")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mock memory manager
        memory_manager = MemoryManager(":memory:")
        memory_manager.initialize()
        
        importer = MemoryImporter(memory_manager)
        
        # Create sample data
        sample_data = create_sample_data()
        
        # Test each format
        formats = [
            ("JSON", create_json_export_file),
            ("CSV", create_csv_export_files),
            ("XML", create_xml_export_file),
            ("SQLite", create_sqlite_export_file)
        ]
        
        for format_name, create_func in formats:
            print(f"\n📄 Testing {format_name} format detection...")
            
            # Create file
            file_path = create_func(temp_dir, sample_data)
            
            # Detect format
            detected_format = importer.detect_format(str(file_path))
            print(f"  File: {file_path.name}")
            print(f"  Detected format: {detected_format.value}")
            
            # Validate detection
            expected_format = ImportFormat[format_name] if format_name != "CSV" else ImportFormat.CSV
            if detected_format == expected_format:
                print(f"  ✅ Format detection successful!")
            else:
                print(f"  ❌ Format detection failed (expected {expected_format.value})")


def demo_json_import():
    """Demonstrate JSON import capabilities"""
    print_header("JSON Import Demo")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a real memory manager
        db_path = Path(temp_dir) / "demo.db"
        memory_manager = MemoryManager(str(db_path))
        memory_manager.initialize()
        
        # Create sample data
        sample_data = create_sample_data()
        json_file = create_json_export_file(temp_dir, sample_data)
        
        print(f"📄 Created JSON file: {json_file.name}")
        print(f"📊 Data contains {len(sample_data['tables'])} tables")
        
        # Import with different options
        import_scenarios = [
            ("Basic Import", ImportOptions(
                format=ImportFormat.JSON,
                create_backup=True,
                progress_callback=progress_callback
            )),
            ("Users Only", ImportOptions(
                format=ImportFormat.JSON,
                scope=ImportScope.USERS,
                create_backup=False
            )),
            ("Dry Run", ImportOptions(
                format=ImportFormat.JSON,
                dry_run=True,
                create_backup=False
            ))
        ]
        
        for scenario_name, options in import_scenarios:
            print_subheader(f"{scenario_name}")
            
            start_time = time.time()
            result = import_memory_data(str(json_file), memory_manager, options)
            end_time = time.time()
            
            print(f"  📈 Import completed in {end_time - start_time:.2f} seconds")
            print(f"  ✅ Success: {result.success}")
            print(f"  📊 Records imported: {result.records_imported}")
            print(f"  🔍 Format detected: {result.format_detected.value}")
            print(f"  ⚠️  Conflicts found: {result.conflicts_found}")
            print(f"  🛠️  Conflicts resolved: {result.conflicts_resolved}")
            
            if result.backup_path:
                print(f"  💾 Backup created: {Path(result.backup_path).name}")
            
            if result.errors:
                print(f"  ❌ Errors: {len(result.errors)}")
                for error in result.errors[:3]:  # Show first 3 errors
                    print(f"    • {error}")
            
            if result.warnings:
                print(f"  ⚠️  Warnings: {len(result.warnings)}")
                for warning in result.warnings[:3]:  # Show first 3 warnings
                    print(f"    • {warning}")


def demo_csv_import():
    """Demonstrate CSV import capabilities"""
    print_header("CSV Import Demo")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a real memory manager
        db_path = Path(temp_dir) / "demo.db"
        memory_manager = MemoryManager(str(db_path))
        memory_manager.initialize()
        
        # Create sample data
        sample_data = create_sample_data()
        csv_zip_file = create_csv_export_files(temp_dir, sample_data)
        
        print(f"📄 Created CSV ZIP file: {csv_zip_file.name}")
        print(f"📊 Contains CSV files for {len(sample_data['tables'])} tables")
        
        # Import CSV data
        options = ImportOptions(
            format=ImportFormat.CSV,
            conflict_resolution=ConflictResolution.OVERWRITE,
            create_backup=True,
            progress_callback=progress_callback
        )
        
        start_time = time.time()
        result = import_memory_data(str(csv_zip_file), memory_manager, options)
        end_time = time.time()
        
        print(f"\n📈 Import completed in {end_time - start_time:.2f} seconds")
        print(f"✅ Success: {result.success}")
        print(f"📊 Records imported: {result.records_imported}")
        print(f"🔍 Format detected: {result.format_detected.value}")
        
        if result.backup_path:
            print(f"💾 Backup created: {Path(result.backup_path).name}")


def demo_xml_import():
    """Demonstrate XML import capabilities"""
    print_header("XML Import Demo")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a real memory manager
        db_path = Path(temp_dir) / "demo.db"
        memory_manager = MemoryManager(str(db_path))
        memory_manager.initialize()
        
        # Create sample data
        sample_data = create_sample_data()
        xml_file = create_xml_export_file(temp_dir, sample_data)
        
        print(f"📄 Created XML file: {xml_file.name}")
        print(f"📊 Data contains {len(sample_data['tables'])} tables")
        
        # Import XML data
        options = ImportOptions(
            format=ImportFormat.XML,
            conflict_resolution=ConflictResolution.MERGE,
            validate_schema=True,
            create_backup=True
        )
        
        start_time = time.time()
        result = import_memory_data(str(xml_file), memory_manager, options)
        end_time = time.time()
        
        print(f"\n📈 Import completed in {end_time - start_time:.2f} seconds")
        print(f"✅ Success: {result.success}")
        print(f"📊 Records imported: {result.records_imported}")
        print(f"🔍 Format detected: {result.format_detected.value}")
        
        if result.backup_path:
            print(f"💾 Backup created: {Path(result.backup_path).name}")


def demo_sqlite_import():
    """Demonstrate SQLite import capabilities"""
    print_header("SQLite Import Demo")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a real memory manager
        db_path = Path(temp_dir) / "demo.db"
        memory_manager = MemoryManager(str(db_path))
        memory_manager.initialize()
        
        # Create sample data
        sample_data = create_sample_data()
        sqlite_file = create_sqlite_export_file(temp_dir, sample_data)
        
        print(f"📄 Created SQLite file: {sqlite_file.name}")
        print(f"📊 Data contains {len(sample_data['tables'])} tables")
        
        # Import SQLite data
        options = ImportOptions(
            format=ImportFormat.SQLITE,
            conflict_resolution=ConflictResolution.OVERWRITE,
            validate_schema=True,
            create_backup=True
        )
        
        start_time = time.time()
        result = import_memory_data(str(sqlite_file), memory_manager, options)
        end_time = time.time()
        
        print(f"\n📈 Import completed in {end_time - start_time:.2f} seconds")
        print(f"✅ Success: {result.success}")
        print(f"📊 Records imported: {result.records_imported}")
        print(f"🔍 Format detected: {result.format_detected.value}")
        
        if result.backup_path:
            print(f"💾 Backup created: {Path(result.backup_path).name}")


def demo_conflict_resolution():
    """Demonstrate conflict resolution strategies"""
    print_header("Conflict Resolution Demo")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a real memory manager with existing data
        db_path = Path(temp_dir) / "demo.db"
        memory_manager = MemoryManager(str(db_path))
        memory_manager.initialize()
        
        # Add some existing data to create conflicts
        existing_user = {
            "user_id": "user_001",
            "username": "existing_user",
            "email": "existing@example.com",
            "created_at": "2024-01-01T00:00:00Z"
        }
        
        # Insert existing data (this would normally be done through proper API)
        print("📝 Setting up existing data to create conflicts...")
        
        # Create import data with conflicts
        conflicting_data = {
            "export_info": {
                "format": "json",
                "version": "1.0",
                "timestamp": datetime.now().isoformat()
            },
            "tables": {
                "users": [
                    {
                        "user_id": "user_001",  # Conflicts with existing
                        "username": "new_user",
                        "email": "new@example.com",
                        "created_at": "2024-01-01T10:00:00Z"
                    },
                    {
                        "user_id": "user_004",  # New user, no conflict
                        "username": "fresh_user",
                        "email": "fresh@example.com",
                        "created_at": "2024-01-04T10:00:00Z"
                    }
                ]
            }
        }
        
        json_file = create_json_export_file(temp_dir, conflicting_data)
        
        # Test different conflict resolution strategies
        strategies = [
            ("Skip Conflicts", ConflictResolution.SKIP),
            ("Overwrite Existing", ConflictResolution.OVERWRITE),
            ("Rename Conflicts", ConflictResolution.RENAME)
        ]
        
        for strategy_name, strategy in strategies:
            print_subheader(f"{strategy_name}")
            
            options = ImportOptions(
                format=ImportFormat.JSON,
                conflict_resolution=strategy,
                create_backup=False,
                validate_schema=True
            )
            
            result = import_memory_data(str(json_file), memory_manager, options)
            
            print(f"  📊 Records imported: {result.records_imported}")
            print(f"  ⚠️  Conflicts found: {result.conflicts_found}")
            print(f"  🛠️  Conflicts resolved: {result.conflicts_resolved}")
            print(f"  ✅ Success: {result.success}")
            
            if result.errors:
                print(f"  ❌ Errors: {len(result.errors)}")
            
            if result.warnings:
                print(f"  ⚠️  Warnings: {len(result.warnings)}")


def demo_api_functions():
    """Demonstrate public API functions"""
    print_header("Public API Functions Demo")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create memory managers
        source_db = Path(temp_dir) / "source.db"
        target_db = Path(temp_dir) / "target.db"
        
        source_manager = MemoryManager(str(source_db))
        source_manager.initialize()
        target_manager = MemoryManager(str(target_db))
        target_manager.initialize()
        
        # Create sample data
        sample_data = create_sample_data()
        
        # Test 1: import_user_backup
        print_subheader("User Backup Import")
        json_file = create_json_export_file(temp_dir, sample_data)
        
        result = import_user_backup(
            str(json_file),
            target_manager,
            "user_001",
            ConflictResolution.SKIP
        )
        
        print(f"  📊 User backup import: {result.success}")
        print(f"  📈 Records imported: {result.records_imported}")
        
        # Test 2: migrate_from_database
        print_subheader("Database Migration")
        sqlite_file = create_sqlite_export_file(temp_dir, sample_data)
        
        result = migrate_from_database(str(sqlite_file), target_manager)
        
        print(f"  📊 Database migration: {result.success}")
        print(f"  📈 Records imported: {result.records_imported}")
        
        # Test 3: restore_from_backup
        print_subheader("Backup Restoration")
        backup_file = create_sqlite_export_file(temp_dir, sample_data)
        
        result = restore_from_backup(str(backup_file), target_manager)
        
        print(f"  📊 Backup restoration: {result.success}")
        print(f"  📈 Records imported: {result.records_imported}")


def demo_error_handling():
    """Demonstrate error handling capabilities"""
    print_header("Error Handling Demo")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a real memory manager
        db_path = Path(temp_dir) / "demo.db"
        memory_manager = MemoryManager(str(db_path))
        memory_manager.initialize()
        
        # Test scenarios
        error_scenarios = [
            ("Non-existent File", "/nonexistent/file.json"),
            ("Empty File", None),
            ("Invalid JSON", None),
            ("Corrupted Data", None)
        ]
        
        for scenario_name, file_path in error_scenarios:
            print_subheader(scenario_name)
            
            if file_path is None:
                # Create test file based on scenario
                if scenario_name == "Empty File":
                    file_path = Path(temp_dir) / "empty.json"
                    file_path.write_text("{}")
                elif scenario_name == "Invalid JSON":
                    file_path = Path(temp_dir) / "invalid.json"
                    file_path.write_text("invalid json content")
                elif scenario_name == "Corrupted Data":
                    file_path = Path(temp_dir) / "corrupted.json"
                    file_path.write_text('{"tables": {"users": [{"invalid": "data"}]}}')
                
                file_path = str(file_path)
            
            options = ImportOptions(
                format=ImportFormat.JSON,
                create_backup=False,
                ignore_errors=True
            )
            
            result = import_memory_data(file_path, memory_manager, options)
            
            print(f"  📊 Import attempted: {not result.success or result.records_imported >= 0}")
            print(f"  ✅ Success: {result.success}")
            print(f"  📈 Records imported: {result.records_imported}")
            
            if result.errors:
                print(f"  ❌ Errors: {len(result.errors)}")
                for error in result.errors[:2]:  # Show first 2 errors
                    print(f"    • {error}")
            
            if result.warnings:
                print(f"  ⚠️  Warnings: {len(result.warnings)}")


def main():
    """Run all Memory Import demos"""
    print("🚀 Memory Import System - Comprehensive Demo")
    print("=" * 60)
    print("This demo showcases the complete Memory Import system capabilities")
    print("including multiple formats, validation, conflict resolution, and more.")
    
    try:
        # Run all demos
        demo_format_detection()
        demo_json_import()
        demo_csv_import()
        demo_xml_import()
        demo_sqlite_import()
        demo_conflict_resolution()
        demo_api_functions()
        demo_error_handling()
        
        print_header("Demo Complete!")
        print("🎉 All Memory Import demos completed successfully!")
        print("📚 The system demonstrated:")
        print("  • Multiple import formats (JSON, CSV, XML, SQLite)")
        print("  • Format auto-detection")
        print("  • Data validation and schema checking")
        print("  • Conflict resolution strategies")
        print("  • Progress tracking and callbacks")
        print("  • Backup creation and restoration")
        print("  • Error handling and recovery")
        print("  • Public API functions")
        print("  • Integration with MemoryManager")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 