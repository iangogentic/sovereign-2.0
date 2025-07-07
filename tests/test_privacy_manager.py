"""
Tests for Privacy Controls and Access Management System
"""

import pytest
import tempfile
import shutil
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

from src.sovereign.privacy_manager import (
    PrivacyManager, EncryptionManager, UserRole, PrivacyLevel, 
    ConsentType, AuditAction, EncryptionStatus,
    UserPermission, PrivacySetting, ConsentRecord, AuditLogEntry
)
from src.sovereign.memory_schema import MemorySchema


class MockConfig:
    def __init__(self):
        self.privacy_settings = {
            "encryption_enabled": True,
            "audit_logging": True,
            "require_consent": True
        }


@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_privacy.db")
    yield db_path
    
    # Improved cleanup with retry mechanism
    import gc
    import time
    
    # Force garbage collection to close any remaining connections
    gc.collect()
    time.sleep(0.1)  # Small delay to allow file handles to be released
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            shutil.rmtree(temp_dir)
            break
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(0.2)  # Wait a bit longer
                gc.collect()  # Force another garbage collection
            else:
                # If all retries fail, try to remove just the database file
                try:
                    os.remove(db_path)
                except:
                    pass  # Ignore if we can't remove it


@pytest.fixture
def privacy_manager(temp_db):
    """Create a privacy manager for testing"""
    config = {
        "master_password": "test_password_123",
        "enable_audit_logging": True,
        "default_privacy_level": "internal"
    }
    
    manager = PrivacyManager(config, temp_db)
    yield manager
    
    # Ensure manager is properly closed before cleanup
    try:
        manager.close()
    except:
        pass  # Ignore if already closed


class TestEncryptionManager:
    """Test encryption functionality"""
    
    def test_encryption_manager_initialization(self):
        """Test encryption manager can be initialized"""
        manager = EncryptionManager()
        assert manager.master_password is not None
        assert manager._fernet is not None
    
    def test_encrypt_decrypt_string(self):
        """Test string encryption and decryption"""
        manager = EncryptionManager("test_password")
        original_data = "sensitive information"
        
        encrypted = manager.encrypt_data(original_data)
        assert isinstance(encrypted, bytes)
        assert encrypted != original_data.encode()
        
        decrypted = manager.decrypt_data(encrypted)
        assert decrypted == original_data
    
    def test_encrypt_decrypt_bytes(self):
        """Test bytes encryption and decryption"""
        manager = EncryptionManager("test_password")
        original_data = b"binary data"
        
        encrypted = manager.encrypt_data(original_data)
        assert isinstance(encrypted, bytes)
        assert encrypted != original_data
        
        decrypted = manager.decrypt_data(encrypted)
        assert decrypted == original_data.decode()
    
    def test_encrypt_decrypt_json(self):
        """Test JSON encryption and decryption"""
        manager = EncryptionManager("test_password")
        original_data = {"key": "value", "number": 123, "list": [1, 2, 3]}
        
        encrypted = manager.encrypt_json(original_data)
        assert isinstance(encrypted, bytes)
        
        decrypted = manager.decrypt_json(encrypted)
        assert decrypted == original_data


class TestPrivacyManager:
    """Test privacy manager functionality"""
    
    def test_privacy_manager_initialization(self, privacy_manager):
        """Test privacy manager initializes correctly"""
        assert privacy_manager.db_path is not None
        assert privacy_manager.encryption_manager is not None
        assert privacy_manager.default_settings is not None
    
    def test_privacy_tables_created(self, privacy_manager):
        """Test that all privacy tables are created"""
        with privacy_manager._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check that all privacy tables exist
            tables = [
                'user_permissions', 'privacy_settings', 'consent_records',
                'audit_log', 'encryption_keys', 'encrypted_data'
            ]
            
            for table in tables:
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name=?
                """, (table,))
                assert cursor.fetchone() is not None, f"Table {table} not found"


class TestUserPermissions:
    """Test user permission management"""
    
    def test_grant_permission(self, privacy_manager):
        """Test granting permissions to a user"""
        user_id = 1
        role = UserRole.USER
        permissions = ["read_conversations", "write_messages"]
        
        permission_id = privacy_manager.grant_permission(user_id, role, permissions)
        assert permission_id is not None
        
        # Verify permission was granted
        user_perms = privacy_manager.get_user_permissions(user_id)
        assert user_perms is not None
        assert user_perms.role == role
        assert user_perms.permissions == permissions
    
    def test_grant_admin_permission(self, privacy_manager):
        """Test granting admin permissions"""
        user_id = 2
        role = UserRole.ADMIN
        permissions = ["*"]  # Admin gets all permissions
        
        privacy_manager.grant_permission(user_id, role, permissions)
        
        user_perms = privacy_manager.get_user_permissions(user_id)
        assert user_perms.role == UserRole.ADMIN
    
    def test_revoke_permission(self, privacy_manager):
        """Test revoking user permissions"""
        user_id = 1
        
        # Grant permission first
        privacy_manager.grant_permission(user_id, UserRole.USER, ["read_data"])
        
        # Verify permission exists
        assert privacy_manager.get_user_permissions(user_id) is not None
        
        # Revoke permission
        success = privacy_manager.revoke_permission(user_id)
        assert success
        
        # Verify permission is revoked
        assert privacy_manager.get_user_permissions(user_id) is None
    
    def test_check_permission(self, privacy_manager):
        """Test permission checking"""
        user_id = 1
        
        # Grant specific permissions
        privacy_manager.grant_permission(
            user_id, UserRole.USER, ["read_conversations", "write_messages"]
        )
        
        # Test permission checks
        assert privacy_manager.check_permission(user_id, "read_conversations")
        assert privacy_manager.check_permission(user_id, "write_messages")
        assert not privacy_manager.check_permission(user_id, "delete_data")
    
    def test_admin_has_all_permissions(self, privacy_manager):
        """Test that admin role has all permissions"""
        user_id = 2
        
        privacy_manager.grant_permission(user_id, UserRole.ADMIN, ["*"])
        
        # Admin should have any permission
        assert privacy_manager.check_permission(user_id, "any_permission")
        assert privacy_manager.check_permission(user_id, "delete_everything")
        assert privacy_manager.check_permission(user_id, "super_secret_action")
    
    def test_expired_permissions(self, privacy_manager):
        """Test that expired permissions are not valid"""
        user_id = 1
        expires_at = datetime.now() - timedelta(hours=1)  # Expired 1 hour ago
        
        privacy_manager.grant_permission(
            user_id, UserRole.USER, ["read_data"], expires_at=expires_at
        )
        
        # Permission should not be valid
        assert not privacy_manager.check_permission(user_id, "read_data")


class TestPrivacySettings:
    """Test privacy settings management"""
    
    def test_set_privacy_setting(self, privacy_manager):
        """Test setting privacy settings"""
        user_id = 1
        setting_name = "data_collection_enabled"
        setting_value = False
        
        success = privacy_manager.set_privacy_setting(
            user_id, setting_name, setting_value, PrivacyLevel.INTERNAL
        )
        assert success
        
        # Verify setting was stored
        retrieved_value = privacy_manager.get_privacy_setting(user_id, setting_name)
        assert retrieved_value == setting_value
    
    def test_encrypted_privacy_setting(self, privacy_manager):
        """Test setting encrypted privacy settings"""
        user_id = 1
        setting_name = "secret_preference"
        setting_value = "very_secret_data"
        
        success = privacy_manager.set_privacy_setting(
            user_id, setting_name, setting_value, PrivacyLevel.CONFIDENTIAL
        )
        assert success
        
        # Verify setting can be retrieved and decrypted
        retrieved_value = privacy_manager.get_privacy_setting(user_id, setting_name)
        assert retrieved_value == setting_value
    
    def test_get_all_privacy_settings(self, privacy_manager):
        """Test getting all privacy settings"""
        user_id = 1
        
        # Set multiple settings
        privacy_manager.set_privacy_setting(user_id, "setting1", "value1")
        privacy_manager.set_privacy_setting(user_id, "setting2", "value2")
        
        all_settings = privacy_manager.get_all_privacy_settings(user_id)
        
        # Should include our settings plus defaults
        assert "setting1" in all_settings
        assert "setting2" in all_settings
        assert all_settings["setting1"] == "value1"
        assert all_settings["setting2"] == "value2"
        
        # Should include default settings
        assert "data_collection_enabled" in all_settings
    
    def test_default_privacy_settings(self, privacy_manager):
        """Test default privacy settings are returned"""
        user_id = 1
        
        # Get setting that hasn't been set
        value = privacy_manager.get_privacy_setting(user_id, "data_collection_enabled")
        assert value == privacy_manager.default_settings["data_collection_enabled"]


class TestConsentManagement:
    """Test user consent management"""
    
    def test_record_consent(self, privacy_manager):
        """Test recording user consent"""
        user_id = 1
        consent_type = ConsentType.DATA_COLLECTION
        
        consent_id = privacy_manager.record_consent(user_id, consent_type, True)
        assert consent_id is not None
        
        # Verify consent was recorded
        consent = privacy_manager.get_consent(user_id, consent_type)
        assert consent is not None
        assert consent.granted is True
        assert consent.consent_type == consent_type
    
    def test_withdraw_consent(self, privacy_manager):
        """Test withdrawing consent"""
        user_id = 1
        consent_type = ConsentType.DATA_PROCESSING
        
        # Grant consent first
        privacy_manager.record_consent(user_id, consent_type, True)
        assert privacy_manager.check_consent(user_id, consent_type)
        
        # Withdraw consent
        privacy_manager.record_consent(user_id, consent_type, False)
        assert not privacy_manager.check_consent(user_id, consent_type)
    
    def test_consent_with_expiration(self, privacy_manager):
        """Test consent with expiration date"""
        user_id = 1
        consent_type = ConsentType.ANALYTICS
        expires_at = datetime.now() + timedelta(days=30)
        
        privacy_manager.record_consent(
            user_id, consent_type, True, expires_at=expires_at
        )
        
        # Should be valid now
        assert privacy_manager.check_consent(user_id, consent_type)
        
        # Test with past expiration
        past_expiration = datetime.now() - timedelta(days=1)
        privacy_manager.record_consent(
            user_id, ConsentType.MARKETING, True, expires_at=past_expiration
        )
        
        # Should not be valid
        assert not privacy_manager.check_consent(user_id, ConsentType.MARKETING)
    
    def test_consent_metadata(self, privacy_manager):
        """Test consent with metadata"""
        user_id = 1
        consent_type = ConsentType.PERSONALIZATION
        metadata = {"source": "settings_page", "ip": "127.0.0.1"}
        
        privacy_manager.record_consent(
            user_id, consent_type, True, metadata=metadata
        )
        
        consent = privacy_manager.get_consent(user_id, consent_type)
        assert consent.metadata == metadata


class TestAuditLogging:
    """Test audit logging functionality"""
    
    def test_audit_log_creation(self, privacy_manager):
        """Test that audit logs are created"""
        user_id = 1
        
        # Perform an action that should be audited
        privacy_manager.grant_permission(user_id, UserRole.USER, ["read_data"])
        
        # Check audit log
        logs = privacy_manager.get_audit_log(user_id=user_id)
        assert len(logs) > 0
        
        # Find the permission grant log
        permission_logs = [log for log in logs if log.action == AuditAction.SETTING_CHANGE]
        assert len(permission_logs) > 0
    
    def test_audit_log_filtering(self, privacy_manager):
        """Test audit log filtering"""
        user_id = 1
        
        # Create different types of audit entries
        privacy_manager.record_consent(user_id, ConsentType.DATA_COLLECTION, True)
        privacy_manager.set_privacy_setting(user_id, "test_setting", "test_value")
        
        # Filter by action
        consent_logs = privacy_manager.get_audit_log(action=AuditAction.CONSENT_CHANGE)
        setting_logs = privacy_manager.get_audit_log(action=AuditAction.SETTING_CHANGE)
        
        assert len(consent_logs) > 0
        assert len(setting_logs) > 0
        
        # Verify action types
        for log in consent_logs:
            assert log.action == AuditAction.CONSENT_CHANGE
        
        for log in setting_logs:
            assert log.action == AuditAction.SETTING_CHANGE
    
    def test_audit_log_date_filtering(self, privacy_manager):
        """Test audit log date filtering"""
        user_id = 1
        
        # Create an audit entry
        privacy_manager.record_consent(user_id, ConsentType.DATA_COLLECTION, True)
        
        # Filter by date range
        now = datetime.now()
        start_date = now - timedelta(hours=1)
        end_date = now + timedelta(hours=1)
        
        logs = privacy_manager.get_audit_log(
            start_date=start_date, end_date=end_date
        )
        assert len(logs) > 0
        
        # All logs should be within date range
        for log in logs:
            assert start_date <= log.timestamp <= end_date


class TestDataEncryption:
    """Test data encryption functionality"""
    
    def test_encrypt_user_data(self, privacy_manager):
        """Test encrypting user data"""
        user_id = 1
        table_name = "messages"
        record_id = 123
        field_name = "content"
        data = "This is sensitive message content"
        
        success = privacy_manager.encrypt_user_data(
            user_id, table_name, record_id, field_name, data
        )
        assert success
        
        # Verify we can decrypt it back
        decrypted = privacy_manager.decrypt_user_data(
            user_id, table_name, record_id, field_name
        )
        assert decrypted == data
    
    def test_decrypt_nonexistent_data(self, privacy_manager):
        """Test decrypting data that doesn't exist"""
        user_id = 1
        decrypted = privacy_manager.decrypt_user_data(
            user_id, "nonexistent", 999, "field"
        )
        assert decrypted is None


class TestDataAccessControl:
    """Test data access control"""
    
    def test_can_access_data(self, privacy_manager):
        """Test data access permissions"""
        user_id = 1
        
        # Grant permission
        privacy_manager.grant_permission(
            user_id, UserRole.USER, ["read_conversations"]
        )
        
        # Should be able to access
        assert privacy_manager.can_access_data(
            user_id, "conversations", "123", AuditAction.READ
        )
        
        # Should not be able to delete without permission
        assert not privacy_manager.can_access_data(
            user_id, "conversations", "123", AuditAction.DELETE
        )
    
    def test_access_with_consent_requirement(self, privacy_manager):
        """Test access control with consent requirements"""
        user_id = 1
        
        # Grant permission and set consent requirement
        privacy_manager.grant_permission(
            user_id, UserRole.USER, ["read_conversations"]
        )
        privacy_manager.set_privacy_setting(user_id, "require_consent", True)
        
        # Without consent, should not be able to access
        assert not privacy_manager.can_access_data(
            user_id, "conversations", "123", AuditAction.READ
        )
        
        # With consent, should be able to access
        privacy_manager.record_consent(user_id, ConsentType.DATA_PROCESSING, True)
        assert privacy_manager.can_access_data(
            user_id, "conversations", "123", AuditAction.READ
        )
    
    def test_access_data_with_audit(self, privacy_manager):
        """Test data access with automatic audit logging"""
        user_id = 1
        
        # Grant permission
        privacy_manager.grant_permission(
            user_id, UserRole.USER, ["read_conversations"]
        )
        
        # Access data
        success = privacy_manager.access_data_with_audit(
            user_id, "conversations", "123", AuditAction.READ
        )
        assert success
        
        # Check audit log
        logs = privacy_manager.get_audit_log(user_id=user_id, action=AuditAction.READ)
        assert len(logs) > 0
        
        # Verify audit details
        log = logs[0]
        assert log.resource_type == "conversations"
        assert log.resource_id == "123"
        assert log.success is True


class TestUtilityMethods:
    """Test utility methods"""
    
    def test_privacy_summary(self, privacy_manager):
        """Test getting privacy summary"""
        user_id = 1
        
        # Set up user data
        privacy_manager.grant_permission(user_id, UserRole.USER, ["read_data"])
        privacy_manager.set_privacy_setting(user_id, "test_setting", "test_value")
        privacy_manager.record_consent(user_id, ConsentType.DATA_COLLECTION, True)
        
        summary = privacy_manager.get_privacy_summary(user_id)
        
        assert summary["user_id"] == user_id
        assert summary["role"] == "user"
        assert "read_data" in summary["permissions"]
        assert "test_setting" in summary["privacy_settings"]
        assert summary["consent_status"]["data_collection"] is True
    
    def test_cleanup_expired_data(self, privacy_manager):
        """Test cleanup of expired data"""
        user_id = 1
        
        # Create expired permission
        expired_time = datetime.now() - timedelta(hours=1)
        privacy_manager.grant_permission(
            user_id, UserRole.USER, ["read_data"], expires_at=expired_time
        )
        
        # Verify permission exists initially
        assert privacy_manager.get_user_permissions(user_id) is not None
        
        # Cleanup expired data
        results = privacy_manager.cleanup_expired_data()
        assert results["expired_permissions"] > 0
        
        # Verify permission is gone
        assert privacy_manager.get_user_permissions(user_id) is None


class TestIntegrationScenarios:
    """Test integration scenarios"""
    
    def test_complete_user_privacy_workflow(self, privacy_manager):
        """Test complete privacy workflow for a user"""
        user_id = 1
        
        # 1. Grant user permissions
        privacy_manager.grant_permission(
            user_id, UserRole.USER, ["read_conversations", "write_messages"]
        )
        
        # 2. Set privacy preferences
        privacy_manager.set_privacy_setting(user_id, "data_collection_enabled", True)
        privacy_manager.set_privacy_setting(user_id, "analytics_enabled", False)
        
        # 3. Record consent
        privacy_manager.record_consent(user_id, ConsentType.DATA_COLLECTION, True)
        privacy_manager.record_consent(user_id, ConsentType.ANALYTICS, False)
        
        # 4. Encrypt sensitive data
        privacy_manager.encrypt_user_data(
            user_id, "messages", 1, "content", "Sensitive message"
        )
        
        # 5. Test data access
        assert privacy_manager.can_access_data(
            user_id, "conversations", "1", AuditAction.READ
        )
        
        # 6. Get complete privacy summary
        summary = privacy_manager.get_privacy_summary(user_id)
        assert summary["role"] == "user"
        assert summary["consent_status"]["data_collection"] is True
        assert summary["consent_status"]["analytics"] is False
        
        # 7. Verify audit trail exists
        logs = privacy_manager.get_audit_log(user_id=user_id)
        assert len(logs) >= 5  # Should have multiple audit entries
    
    def test_privacy_violation_prevention(self, privacy_manager):
        """Test that privacy violations are prevented"""
        user_id = 1
        
        # Try to access data without permissions
        assert not privacy_manager.can_access_data(
            user_id, "conversations", "1", AuditAction.READ
        )
        
        # Grant limited permissions
        privacy_manager.grant_permission(user_id, UserRole.USER, ["read_messages"])
        
        # Should not be able to access conversations
        assert not privacy_manager.can_access_data(
            user_id, "conversations", "1", AuditAction.READ
        )
        
        # Should not be able to delete messages
        assert not privacy_manager.can_access_data(
            user_id, "messages", "1", AuditAction.DELETE
        )
    
    @pytest.mark.asyncio
    async def test_concurrent_access(self, privacy_manager):
        """Test concurrent access to privacy manager"""
        import asyncio
        
        user_id = 1
        
        # Grant permissions for testing
        privacy_manager.grant_permission(
            user_id, UserRole.USER, ["read_data", "write_data"]
        )
        
        async def access_data(resource_id):
            return privacy_manager.access_data_with_audit(
                user_id, "test_resource", str(resource_id), AuditAction.READ
            )
        
        # Test concurrent access
        tasks = [access_data(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(results)
        
        # Should have 10 audit entries
        logs = privacy_manager.get_audit_log(user_id=user_id, action=AuditAction.READ)
        assert len(logs) >= 10


if __name__ == "__main__":
    pytest.main([__file__]) 