#!/usr/bin/env python3
"""
Memory Export System Demonstration

This demo script showcases the comprehensive memory export functionality
including various formats, filtering options, and use cases.
"""

import sys
import tempfile
import shutil
import json
import csv
import sqlite3
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sovereign.config import Config
from sovereign.memory_exporter import (
    MemoryExporter, ExportFormat, ExportScope, CompressionType,
    ExportFilter, ExportOptions, export_user_data, export_conversation_data,
    create_full_backup
)
from sovereign.memory_schema import MemorySchema
from sovereign.memory_manager import MemoryManager, MessageSender


def create_sample_data(db_path: str) -> dict:
    """Create sample data for demonstration"""
    print("Creating sample memory data...")
    
    # Create configuration
    config = Config()
    config.database.db_path = db_path
    
    # Initialize schema
    schema = MemorySchema(db_path)
    schema.create_schema()
    
    # Create memory manager
    memory_manager = MemoryManager(config)
    memory_manager.db_path = Path(db_path)
    memory_manager.schema = schema
    
    try:
        # Create sample users
        user1_id = memory_manager.create_user("alice_smith", "alice@example.com")
        user2_id = memory_manager.create_user("bob_jones", "bob@example.com")
        user3_id = memory_manager.create_user("carol_white", "carol@example.com")
        
        # User 1: AI Research Conversations
        memory_manager.set_current_user(user1_id)
        
        conv1_id = memory_manager.start_conversation("AI Ethics Discussion", privacy_level=2)
        memory_manager.add_message(
            "What are the main ethical considerations in AI development?",
            MessageSender.USER, conv1_id
        )
        memory_manager.add_message(
            "Key ethical considerations include fairness, transparency, privacy, "
            "accountability, and preventing harm. These principles should guide "
            "AI system design and deployment.",
            MessageSender.ASSISTANT, conv1_id
        )
        memory_manager.add_message(
            "How can we ensure AI systems are fair and unbiased?",
            MessageSender.USER, conv1_id
        )
        memory_manager.add_message(
            "Ensuring fairness requires diverse training data, bias testing, "
            "regular audits, and inclusive development teams. It's an ongoing process.",
            MessageSender.ASSISTANT, conv1_id
        )
        
        conv2_id = memory_manager.start_conversation("Machine Learning Techniques")
        memory_manager.add_message(
            "Explain the difference between supervised and unsupervised learning.",
            MessageSender.USER, conv2_id
        )
        memory_manager.add_message(
            "Supervised learning uses labeled data to learn patterns, while "
            "unsupervised learning finds hidden structures in unlabeled data. "
            "Supervised is like learning with a teacher, unsupervised is like "
            "discovering patterns on your own.",
            MessageSender.ASSISTANT, conv2_id
        )
        
        # User 2: Technology Help
        memory_manager.set_current_user(user2_id)
        
        conv3_id = memory_manager.start_conversation("Python Programming Help")
        memory_manager.add_message(
            "How do I handle exceptions in Python?",
            MessageSender.USER, conv3_id
        )
        memory_manager.add_message(
            "Use try-except blocks: try: risky_code() except Exception as e: handle_error(e)",
            MessageSender.ASSISTANT, conv3_id
        )
        
        conv4_id = memory_manager.start_conversation("Database Design Questions")
        memory_manager.add_message(
            "What's the difference between SQL and NoSQL databases?",
            MessageSender.USER, conv4_id
        )
        memory_manager.add_message(
            "SQL databases are structured and use tables with relationships, "
            "while NoSQL databases are more flexible and can store unstructured data.",
            MessageSender.ASSISTANT, conv4_id
        )
        
        # User 3: Personal Assistant
        memory_manager.set_current_user(user3_id)
        
        conv5_id = memory_manager.start_conversation("Travel Planning", privacy_level=3)
        memory_manager.add_message(
            "I'm planning a trip to Japan. What should I know?",
            MessageSender.USER, conv5_id
        )
        memory_manager.add_message(
            "Japan offers amazing culture, food, and technology. Key tips: learn basic "
            "Japanese phrases, carry cash, bow for greetings, and try local cuisine.",
            MessageSender.ASSISTANT, conv5_id
        )
        
        # Add documents
        memory_manager.set_current_user(user1_id)
        doc1_id = memory_manager.add_document(
            "AI Ethics Guidelines",
            "Artificial Intelligence ethics involves ensuring AI systems are developed "
            "and deployed responsibly. Key principles include: 1) Transparency - AI "
            "decisions should be explainable, 2) Fairness - avoiding bias and discrimination, "
            "3) Privacy - protecting user data, 4) Accountability - clear responsibility "
            "for AI outcomes, 5) Safety - preventing harm to humans and society.",
            "research_paper",
            privacy_level=1
        )
        
        doc2_id = memory_manager.add_document(
            "Machine Learning Basics",
            "Machine Learning is a subset of AI that enables computers to learn from "
            "data without explicit programming. Types include: Supervised Learning "
            "(learns from labeled examples), Unsupervised Learning (finds patterns in "
            "unlabeled data), and Reinforcement Learning (learns through trial and error "
            "with rewards). Common algorithms include neural networks, decision trees, "
            "and support vector machines.",
            "tutorial",
            privacy_level=1
        )
        
        memory_manager.set_current_user(user2_id)
        doc3_id = memory_manager.add_document(
            "Python Best Practices",
            "Python coding best practices: 1) Use meaningful variable names, "
            "2) Follow PEP 8 style guide, 3) Write docstrings for functions, "
            "4) Use virtual environments, 5) Handle exceptions properly, "
            "6) Write tests for your code, 7) Use type hints for clarity.",
            "guide",
            privacy_level=1
        )
        
        # Create chunks for documents
        memory_manager.create_chunks(doc1_id, chunk_size=100, overlap_size=20)
        memory_manager.create_chunks(doc2_id, chunk_size=100, overlap_size=20)
        memory_manager.create_chunks(doc3_id, chunk_size=100, overlap_size=20)
        
        print(f"✅ Created sample data:")
        print(f"   - Users: 3 (Alice, Bob, Carol)")
        print(f"   - Conversations: 5")
        print(f"   - Messages: 10+")
        print(f"   - Documents: 3")
        print(f"   - Chunks: Multiple per document")
        
        return {
            'user1_id': user1_id,
            'user2_id': user2_id,
            'user3_id': user3_id,
            'conv1_id': conv1_id,
            'conv2_id': conv2_id,
            'conv3_id': conv3_id,
            'conv4_id': conv4_id,
            'conv5_id': conv5_id,
            'doc1_id': doc1_id,
            'doc2_id': doc2_id,
            'doc3_id': doc3_id
        }
    
    finally:
        memory_manager.close()
        schema.close()


def demo_export_formats(config: Config, db_path: str, output_dir: Path):
    """Demonstrate different export formats"""
    print("\n" + "="*60)
    print("DEMO 1: Export Formats (JSON, CSV, XML, SQLite)")
    print("="*60)
    
    with MemoryExporter(config, db_path) as exporter:
        
        # 1. JSON Export (Complete)
        print("\n📄 JSON Export (Complete Database)...")
        options = ExportOptions(
            format=ExportFormat.JSON,
            output_directory=output_dir,
            filename_prefix="complete_json_export",
            pretty_format=True,
            include_schema=True
        )
        
        result = exporter.export_memory_data(options=options)
        
        print(f"   ✅ Success: {result.success}")
        print(f"   📁 Output: {result.output_files[0].name}")
        print(f"   📊 Records: {result.total_records_exported}")
        print(f"   💾 Size: {result.export_size_bytes / 1024:.1f} KB")
        print(f"   ⏱️  Time: {result.execution_time:.2f}s")
        print(f"   📋 Tables: {', '.join(result.tables_exported[:5])}{'...' if len(result.tables_exported) > 5 else ''}")
        
        # 2. CSV Export (Separate files per table)
        print("\n📊 CSV Export (Separate Files)...")
        export_filter = ExportFilter(include_binary_data=False)
        options = ExportOptions(
            format=ExportFormat.CSV,
            output_directory=output_dir,
            filename_prefix="csv_export"
        )
        
        result = exporter.export_memory_data(export_filter, options)
        
        print(f"   ✅ Success: {result.success}")
        print(f"   📁 Files: {len(result.output_files)} CSV files")
        print(f"   📊 Records: {result.total_records_exported}")
        print(f"   💾 Total Size: {result.export_size_bytes / 1024:.1f} KB")
        
        # Show sample CSV files
        for file_path in result.output_files[:3]:
            print(f"      - {file_path.name}")
        
        # 3. XML Export
        print("\n🏷️  XML Export...")
        options = ExportOptions(
            format=ExportFormat.XML,
            output_directory=output_dir,
            filename_prefix="xml_export",
            pretty_format=True
        )
        
        result = exporter.export_memory_data(options=options)
        
        print(f"   ✅ Success: {result.success}")
        print(f"   📁 Output: {result.output_files[0].name}")
        print(f"   📊 Records: {result.total_records_exported}")
        print(f"   💾 Size: {result.export_size_bytes / 1024:.1f} KB")
        
        # 4. SQLite Export (Complete database copy)
        print("\n🗄️  SQLite Export (Database Copy)...")
        options = ExportOptions(
            format=ExportFormat.SQLITE,
            output_directory=output_dir,
            filename_prefix="sqlite_backup"
        )
        
        result = exporter.export_memory_data(options=options)
        
        print(f"   ✅ Success: {result.success}")
        print(f"   📁 Output: {result.output_files[0].name}")
        print(f"   📊 Records: {result.total_records_exported}")
        print(f"   💾 Size: {result.export_size_bytes / 1024:.1f} KB")


def demo_selective_filtering(config: Config, db_path: str, output_dir: Path, test_data: dict):
    """Demonstrate selective export filtering"""
    print("\n" + "="*60)
    print("DEMO 2: Selective Export Filtering")
    print("="*60)
    
    with MemoryExporter(config, db_path) as exporter:
        
        # 1. Export by User
        print("\n👤 Export User 1 Data Only (Alice)...")
        export_filter = ExportFilter(user_ids=[test_data['user1_id']])
        options = ExportOptions(
            format=ExportFormat.JSON,
            output_directory=output_dir,
            filename_prefix="user_alice_export"
        )
        
        result = exporter.export_memory_data(export_filter, options)
        
        print(f"   ✅ Success: {result.success}")
        print(f"   📊 Records: {result.total_records_exported}")
        print(f"   💾 Size: {result.export_size_bytes / 1024:.1f} KB")
        
        # Verify user filtering
        with open(result.output_files[0], 'r') as f:
            data = json.load(f)
        
        user_conversations = len(data['data']['conversations'])
        print(f"   💬 Alice's Conversations: {user_conversations}")
        
        # 2. Export by Date Range (Recent data only)
        print("\n📅 Export Recent Data (Last 24 hours)...")
        date_from = datetime.now() - timedelta(hours=24)
        export_filter = ExportFilter(
            date_from=date_from,
            scopes=[ExportScope.CONVERSATIONS, ExportScope.DOCUMENTS]
        )
        options = ExportOptions(
            format=ExportFormat.JSON,
            output_directory=output_dir,
            filename_prefix="recent_data_export"
        )
        
        result = exporter.export_memory_data(export_filter, options)
        
        print(f"   ✅ Success: {result.success}")
        print(f"   📊 Records: {result.total_records_exported}")
        print(f"   💾 Size: {result.export_size_bytes / 1024:.1f} KB")
        
        # 3. Export Specific Conversations
        print("\n💬 Export Specific Conversations...")
        export_filter = ExportFilter(
            conversation_ids=[test_data['conv1_id'], test_data['conv3_id']],
            scopes=[ExportScope.CONVERSATIONS]
        )
        options = ExportOptions(
            format=ExportFormat.CSV,
            output_directory=output_dir,
            filename_prefix="specific_conversations"
        )
        
        result = exporter.export_memory_data(export_filter, options)
        
        print(f"   ✅ Success: {result.success}")
        print(f"   📊 Records: {result.total_records_exported}")
        print(f"   📁 Files: {len(result.output_files)}")
        
        # 4. Export by Scope (Documents Only)
        print("\n📚 Export Documents Only...")
        export_filter = ExportFilter(scopes=[ExportScope.DOCUMENTS])
        options = ExportOptions(
            format=ExportFormat.JSON,
            output_directory=output_dir,
            filename_prefix="documents_only"
        )
        
        result = exporter.export_memory_data(export_filter, options)
        
        print(f"   ✅ Success: {result.success}")
        print(f"   📊 Records: {result.total_records_exported}")
        
        # Show what tables were exported
        with open(result.output_files[0], 'r') as f:
            data = json.load(f)
        
        exported_tables = list(data['data'].keys())
        print(f"   📋 Tables: {', '.join(exported_tables)}")
        
        # 5. Export by Privacy Level
        print("\n🔒 Export Public Data Only (Privacy Level 1)...")
        export_filter = ExportFilter(privacy_levels=[1])
        options = ExportOptions(
            format=ExportFormat.JSON,
            output_directory=output_dir,
            filename_prefix="public_data_only"
        )
        
        result = exporter.export_memory_data(export_filter, options)
        
        print(f"   ✅ Success: {result.success}")
        print(f"   📊 Records: {result.total_records_exported}")


def demo_advanced_options(config: Config, db_path: str, output_dir: Path):
    """Demonstrate advanced export options"""
    print("\n" + "="*60)
    print("DEMO 3: Advanced Export Options")
    print("="*60)
    
    with MemoryExporter(config, db_path) as exporter:
        
        # 1. Compressed Export
        print("\n📦 Compressed Export (ZIP)...")
        options = ExportOptions(
            format=ExportFormat.CSV,
            compression=CompressionType.ZIP,
            output_directory=output_dir,
            filename_prefix="compressed_export"
        )
        
        result = exporter.export_memory_data(options=options)
        
        print(f"   ✅ Success: {result.success}")
        print(f"   📁 Output: {result.output_files[0].name}")
        print(f"   💾 Size: {result.export_size_bytes / 1024:.1f} KB (compressed)")
        
        # Examine ZIP contents
        with zipfile.ZipFile(result.output_files[0], 'r') as zf:
            files = zf.namelist()
            print(f"   📋 Contains: {len(files)} files")
            for file_name in files[:3]:
                print(f"      - {file_name}")
            if len(files) > 3:
                print(f"      ... and {len(files) - 3} more")
        
        # 2. Export without Binary Data
        print("\n🚫 Export Excluding Binary Data...")
        export_filter = ExportFilter(include_binary_data=False)
        options = ExportOptions(
            format=ExportFormat.JSON,
            output_directory=output_dir,
            filename_prefix="no_binary_export",
            include_binary_data=False
        )
        
        result = exporter.export_memory_data(export_filter, options)
        
        print(f"   ✅ Success: {result.success}")
        print(f"   📊 Records: {result.total_records_exported}")
        print(f"   💾 Size: {result.export_size_bytes / 1024:.1f} KB")
        print("   ℹ️  Binary data (embeddings) excluded for portability")
        
        # 3. Export with Record Limits
        print("\n🔢 Export with Record Limits (Max 10 per table)...")
        export_filter = ExportFilter(max_records_per_table=10)
        options = ExportOptions(
            format=ExportFormat.JSON,
            output_directory=output_dir,
            filename_prefix="limited_export"
        )
        
        result = exporter.export_memory_data(export_filter, options)
        
        print(f"   ✅ Success: {result.success}")
        print(f"   📊 Records: {result.total_records_exported}")
        print(f"   💾 Size: {result.export_size_bytes / 1024:.1f} KB")
        
        # 4. Export with Progress Tracking
        print("\n📊 Export with Progress Tracking...")
        
        progress_updates = []
        
        def progress_callback(progress):
            progress_updates.append(progress.progress_percentage)
            if progress.current_table:
                print(f"   📋 Processing: {progress.current_table} "
                     f"({progress.completed_tables}/{progress.total_tables} tables, "
                     f"{progress.progress_percentage:.1f}%)")
        
        options = ExportOptions(
            format=ExportFormat.JSON,
            output_directory=output_dir,
            filename_prefix="progress_tracked_export"
        )
        
        result = exporter.export_memory_data(options=options, progress_callback=progress_callback)
        
        print(f"   ✅ Success: {result.success}")
        print(f"   📊 Final Progress: {len(progress_updates)} updates")
        print(f"   📈 Completion: {max(progress_updates) if progress_updates else 0:.1f}%")


def demo_convenience_functions(config: Config, db_path: str, output_dir: Path, test_data: dict):
    """Demonstrate convenience API functions"""
    print("\n" + "="*60)
    print("DEMO 4: Convenience API Functions")
    print("="*60)
    
    # 1. Export User Data
    print("\n👤 export_user_data() - Alice's Complete Data...")
    result = export_user_data(
        config, 
        db_path, 
        test_data['user1_id'],
        ExportFormat.JSON,
        output_dir
    )
    
    print(f"   ✅ Success: {result.success}")
    print(f"   📊 Records: {result.total_records_exported}")
    print(f"   📁 Output: {result.output_files[0].name}")
    
    # 2. Export Specific Conversations
    print("\n💬 export_conversation_data() - Specific Conversations...")
    conversation_ids = [test_data['conv1_id'], test_data['conv2_id']]
    result = export_conversation_data(
        config,
        db_path,
        conversation_ids,
        ExportFormat.CSV,
        output_dir
    )
    
    print(f"   ✅ Success: {result.success}")
    print(f"   📊 Records: {result.total_records_exported}")
    print(f"   📁 Files: {len(result.output_files)}")
    
    # 3. Create Full Backup
    print("\n💾 create_full_backup() - Complete System Backup...")
    result = create_full_backup(
        config,
        db_path,
        output_dir,
        compress=True
    )
    
    print(f"   ✅ Success: {result.success}")
    print(f"   📊 Records: {result.total_records_exported}")
    print(f"   📁 Output: {result.output_files[0].name}")
    print(f"   💾 Size: {result.export_size_bytes / 1024:.1f} KB")
    print(f"   ⏱️  Time: {result.execution_time:.2f}s")


def demo_data_analysis(output_dir: Path):
    """Analyze exported data to show functionality"""
    print("\n" + "="*60)
    print("DEMO 5: Data Analysis from Exports")
    print("="*60)
    
    # Find the complete JSON export
    json_files = list(output_dir.glob("*complete_json_export*.json"))
    if not json_files:
        print("❌ No complete JSON export found for analysis")
        return
    
    json_file = json_files[0]
    print(f"\n📊 Analyzing: {json_file.name}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Export metadata
    export_info = data.get('export_info', {})
    print(f"\n📋 Export Information:")
    print(f"   🕐 Created: {export_info.get('created_at', 'Unknown')}")
    print(f"   📊 Tables: {export_info.get('total_tables', 0)}")
    print(f"   🔧 Version: {export_info.get('exporter_version', 'Unknown')}")
    
    # Data summary
    exported_data = data.get('data', {})
    print(f"\n📈 Data Summary:")
    
    for table_name, table_data in exported_data.items():
        if isinstance(table_data, list):
            print(f"   📋 {table_name}: {len(table_data)} records")
    
    # User analysis
    users = exported_data.get('users', [])
    conversations = exported_data.get('conversations', [])
    messages = exported_data.get('messages', [])
    documents = exported_data.get('documents', [])
    
    print(f"\n👥 User Analysis:")
    for user in users:
        user_id = user['id']
        username = user['username']
        user_convs = [c for c in conversations if c['user_id'] == user_id]
        user_docs = [d for d in documents if d.get('created_by') == user_id]
        
        print(f"   👤 {username} (ID: {user_id}):")
        print(f"      💬 Conversations: {len(user_convs)}")
        print(f"      📚 Documents: {len(user_docs)}")
    
    # Conversation analysis
    print(f"\n💬 Conversation Analysis:")
    for conv in conversations[:3]:  # Show first 3
        conv_messages = [m for m in messages if m['conversation_id'] == conv['id']]
        print(f"   💬 \"{conv['title']}\":")
        print(f"      📊 Messages: {len(conv_messages)}")
        print(f"      🔒 Privacy: Level {conv.get('privacy_level', 1)}")
        print(f"      📅 Started: {conv.get('started_at', 'Unknown')}")
    
    if len(conversations) > 3:
        print(f"   ... and {len(conversations) - 3} more conversations")
    
    # Schema analysis
    schema_info = data.get('schema', {})
    if schema_info:
        print(f"\n🏗️  Schema Information:")
        print(f"   📋 Tables with schema info: {len(schema_info)}")
        
        # Show sample table schema
        for table_name, table_schema in list(schema_info.items())[:2]:
            columns = table_schema.get('columns', [])
            foreign_keys = table_schema.get('foreign_keys', [])
            print(f"   📋 {table_name}:")
            print(f"      🏛️  Columns: {len(columns)}")
            print(f"      🔗 Foreign Keys: {len(foreign_keys)}")


def demo_file_format_examples(output_dir: Path):
    """Show examples of different file formats"""
    print("\n" + "="*60)
    print("DEMO 6: Export Format Examples")
    print("="*60)
    
    # JSON Example
    json_files = list(output_dir.glob("*complete_json_export*.json"))
    if json_files:
        print("\n📄 JSON Format Sample:")
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        
        # Show a sample user record
        users = data.get('data', {}).get('users', [])
        if users:
            user = users[0]
            print("   Sample User Record (JSON):")
            print(f"   {json.dumps(user, indent=4)}")
    
    # CSV Example
    csv_files = list(output_dir.glob("*csv_export*users.csv"))
    if csv_files:
        print("\n📊 CSV Format Sample:")
        with open(csv_files[0], 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                print("   Sample User Record (CSV):")
                for key, value in rows[0].items():
                    print(f"     {key}: {value}")
    
    # XML Example
    xml_files = list(output_dir.glob("*xml_export*.xml"))
    if xml_files:
        print("\n🏷️  XML Format Sample:")
        with open(xml_files[0], 'r') as f:
            content = f.read()
        
        # Show first few lines
        lines = content.split('\n')[:15]
        print("   XML Structure Sample:")
        for line in lines:
            if line.strip():
                print(f"   {line}")
        print("   ...")
    
    # SQLite Example
    sqlite_files = list(output_dir.glob("*sqlite_backup*.db"))
    if sqlite_files:
        print("\n🗄️  SQLite Format Sample:")
        conn = sqlite3.connect(sqlite_files[0])
        conn.row_factory = sqlite3.Row
        
        # Show table list
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"   Database Tables ({len(tables)}):")
        for table in tables[:10]:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"     📋 {table}: {count} records")
        
        if len(tables) > 10:
            print(f"     ... and {len(tables) - 10} more tables")
        
        conn.close()


def main():
    """Main demonstration function"""
    print("🚀 Memory Export System Demonstration")
    print("=====================================")
    print("This demo showcases the comprehensive memory export functionality")
    print("including various formats, filtering options, and use cases.\n")
    
    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        db_path = str(temp_path / "demo_memory.db")
        output_dir = temp_path / "exports"
        output_dir.mkdir()
        
        print(f"📁 Working Directory: {temp_dir}")
        print(f"🗄️  Database: {db_path}")
        print(f"📤 Export Directory: {output_dir}")
        
        # Create sample data
        test_data = create_sample_data(db_path)
        
        # Create configuration
        config = Config()
        config.database.db_path = db_path
        
        try:
            # Run demonstrations
            demo_export_formats(config, db_path, output_dir)
            demo_selective_filtering(config, db_path, output_dir, test_data)
            demo_advanced_options(config, db_path, output_dir)
            demo_convenience_functions(config, db_path, output_dir, test_data)
            demo_data_analysis(output_dir)
            demo_file_format_examples(output_dir)
            
            # Final summary
            print("\n" + "="*60)
            print("DEMONSTRATION SUMMARY")
            print("="*60)
            
            export_files = list(output_dir.glob("*"))
            total_size = sum(f.stat().st_size for f in export_files if f.is_file())
            
            print(f"\n📊 Export Results:")
            print(f"   📁 Files Created: {len(export_files)}")
            print(f"   💾 Total Size: {total_size / 1024:.1f} KB")
            print(f"   📋 Formats Demonstrated: JSON, CSV, XML, SQLite")
            print(f"   🎯 Features Shown:")
            print(f"      ✅ Multiple export formats")
            print(f"      ✅ Selective filtering (user, date, scope, privacy)")
            print(f"      ✅ Compression and packaging")
            print(f"      ✅ Progress tracking")
            print(f"      ✅ Privacy compliance")
            print(f"      ✅ Data integrity preservation")
            print(f"      ✅ Convenience API functions")
            
            print(f"\n🎉 All demonstrations completed successfully!")
            print(f"📁 Export files are in: {output_dir}")
            
            # Show file listing
            print(f"\n📋 Generated Files:")
            for file_path in sorted(export_files):
                if file_path.is_file():
                    size_kb = file_path.stat().st_size / 1024
                    print(f"   📄 {file_path.name} ({size_kb:.1f} KB)")
        
        except Exception as e:
            print(f"\n❌ Demonstration failed: {e}")
            raise


if __name__ == "__main__":
    main() 