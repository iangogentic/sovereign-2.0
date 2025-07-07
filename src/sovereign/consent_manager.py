"""
Consent Manager - Privacy and User Control System
Handles user consent, permissions, and privacy controls for the Sovereign AI Agent
Also includes external model routing consent for backwards compatibility
"""

import json
import time
import logging
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import asyncio
import psutil
import re

from .config import Config


class ConsentLevel(Enum):
    """User consent levels for different data collection types"""
    DENIED = "denied"
    BASIC = "basic"           # Basic screen monitoring with heavy restrictions
    STANDARD = "standard"     # Normal monitoring with privacy protections
    ENHANCED = "enhanced"     # Full monitoring with user awareness
    FULL = "full"            # Complete access for advanced features


class PrivacyZoneType(Enum):
    """Types of privacy zones for screen redaction"""
    FULL_REDACT = "full_redact"         # Complete blackout
    BLUR = "blur"                       # Blur sensitive content
    SKIP_CAPTURE = "skip_capture"       # Don't capture this area
    TEXT_ONLY = "text_only"             # Capture text but redact visual


class MonitoringState(Enum):
    """Current monitoring state"""
    ACTIVE = "active"                   # Normal monitoring
    PAUSED_USER = "paused_user"         # User manually paused
    PAUSED_SENSITIVE = "paused_sensitive"  # Auto-paused for sensitive app
    PAUSED_PRIVACY = "paused_privacy"   # Auto-paused for privacy zone
    DISABLED = "disabled"               # Completely disabled


@dataclass
class PrivacyZone:
    """Defines a privacy-sensitive screen region"""
    zone_id: str
    name: str
    x: int
    y: int
    width: int
    height: int
    zone_type: PrivacyZoneType
    active: bool = True
    created_timestamp: float = field(default_factory=time.time)
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if a point is within this privacy zone"""
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)
    
    def overlaps_region(self, x: int, y: int, width: int, height: int) -> bool:
        """Check if this privacy zone overlaps with a screen region"""
        return not (x >= self.x + self.width or 
                   x + width <= self.x or
                   y >= self.y + self.height or
                   y + height <= self.y)


@dataclass
class ConsentRecord:
    """Records user consent decisions"""
    consent_id: str
    feature: str
    consent_level: ConsentLevel
    granted_timestamp: float
    expires_timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if consent has expired"""
        if self.expires_timestamp is None:
            return False
        return time.time() > self.expires_timestamp
    
    def is_valid(self) -> bool:
        """Check if consent is currently valid"""
        return self.consent_level != ConsentLevel.DENIED and not self.is_expired()


@dataclass
class SensitiveApp:
    """Configuration for sensitive application detection"""
    app_name: str
    process_patterns: List[str]  # Regex patterns to match process names
    window_patterns: List[str]   # Regex patterns to match window titles
    privacy_action: str = "pause_monitoring"  # Action to take when detected
    priority: int = 1  # Higher priority = more important to protect
    
    def matches_process(self, process_name: str) -> bool:
        """Check if process name matches this sensitive app"""
        for pattern in self.process_patterns:
            if re.search(pattern, process_name, re.IGNORECASE):
                return True
        return False
    
    def matches_window(self, window_title: str) -> bool:
        """Check if window title matches this sensitive app"""
        for pattern in self.window_patterns:
            if re.search(pattern, window_title, re.IGNORECASE):
                return True
        return False


# External Model Routing Consent System (backwards compatibility)

class ConsentMethod(Enum):
    """Methods for requesting consent from user"""
    CLI = "cli"
    GUI = "gui"
    AUTO_APPROVE = "auto_approve"
    AUTO_DENY = "auto_deny"


@dataclass
class ConsentRequest:
    """Request for user consent to route to external model"""
    query: str
    routing_decision: Any  # RoutingDecision type
    provider: str
    timestamp: float = field(default_factory=time.time)


# Backwards compatibility consent callbacks
async def cli_consent_callback(consent_request: ConsentRequest) -> bool:
    """CLI-based consent callback for external routing"""
    try:
        print("\n" + "="*60)
        print("🔗 EXTERNAL MODEL ROUTING REQUEST")
        print("="*60)
        print(f"Query: {consent_request.query}")
        print(f"Provider: {consent_request.provider}")
        print(f"Reasoning: {consent_request.routing_decision.reasoning}")
        print(f"Confidence: {consent_request.routing_decision.confidence:.2f}")
        print(f"Criteria: {[c.value for c in consent_request.routing_decision.criteria]}")
        print("="*60)
        
        while True:
            try:
                response = input("Allow external routing? (y/n/d for details): ").strip().lower()
                
                if response in ['y', 'yes']:
                    print("✅ External routing approved")
                    return True
                elif response in ['n', 'no']:
                    print("❌ External routing denied")
                    return False
                elif response in ['d', 'details']:
                    print("\n📋 Detailed Information:")
                    print(f"  - This query requires specialized knowledge or capabilities")
                    print(f"  - It will be sent to: {consent_request.provider}")
                    print(f"  - Your query will be processed externally")
                    print(f"  - Response will be returned through the local system\n")
                else:
                    print("❓ Please enter 'y' for yes, 'n' for no, or 'd' for details")
                    
            except KeyboardInterrupt:
                print("\n❌ External routing cancelled")
                return False
                
    except Exception as e:
        print(f"❌ Error in consent callback: {e}")
        return False


async def gui_consent_callback(consent_request: ConsentRequest) -> bool:
    """GUI-based consent callback - placeholder for future implementation"""
    # This would integrate with a GUI system
    # For now, fallback to CLI
    return await cli_consent_callback(consent_request)


class ConsentManager:
    """Manages user consent, privacy controls, and sensitive app detection"""
    
    def __init__(self, config: Optional[Config] = None, method: ConsentMethod = ConsentMethod.CLI, 
                 enable_sensitive_app_detection: bool = True):
        self.config = config or Config()
        self.logger = logging.getLogger(__name__)
        
        # External routing consent (backwards compatibility)
        self.method = method
        self.consent_history: List[ConsentRequest] = []
        self.gui_consent_callback: Optional[Callable] = None
        
        # Configuration
        self.enable_sensitive_app_detection = enable_sensitive_app_detection
        
        # Storage paths
        self.data_dir = Path.home() / ".sovereign" / "privacy"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.consent_file = self.data_dir / "consent.json"
        self.privacy_zones_file = self.data_dir / "privacy_zones.json"
        
        # State management
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.privacy_zones: Dict[str, PrivacyZone] = {}
        self.sensitive_apps: Dict[str, SensitiveApp] = {}
        self.monitoring_state = MonitoringState.DISABLED
        
        # Event callbacks
        self.state_change_callbacks: List[Callable[[MonitoringState], None]] = []
        self.consent_change_callbacks: List[Callable[[str, ConsentLevel], None]] = []
        
        # Initialize default sensitive apps only if enabled
        if self.enable_sensitive_app_detection:
            self._load_default_sensitive_apps()
        
        # Load existing data
        self._load_consent_data()
        self._load_privacy_zones()
        
        self.logger.info("ConsentManager initialized")
    
    def _load_default_sensitive_apps(self):
        """Load default sensitive application configurations"""
        default_apps = [
            SensitiveApp(
                app_name="Password Managers",
                process_patterns=[
                    r"1password.*",
                    r"keepass.*",
                    r"bitwarden.*",
                    r"lastpass.*",
                    r"dashlane.*",
                    r"roboform.*"
                ],
                window_patterns=[
                    r"1Password.*",
                    r"KeePass.*",
                    r"Bitwarden.*",
                    r"LastPass.*",
                    r"Dashlane.*"
                ],
                priority=10
            ),
            SensitiveApp(
                app_name="Banking & Finance",
                process_patterns=[
                    r".*bank.*",
                    r".*finance.*",
                    r".*trading.*",
                    r".*investment.*",
                    r".*crypto.*wallet.*"
                ],
                window_patterns=[
                    r".*Bank.*",
                    r".*Online Banking.*",
                    r".*Finance.*",
                    r".*Trading.*",
                    r".*Crypto.*",
                    r".*Wallet.*"
                ],
                priority=9
            ),
            SensitiveApp(
                app_name="Privacy Tools",
                process_patterns=[
                    r"tor.*",
                    r".*vpn.*",
                    r".*proxy.*",
                    r"privoxy.*",
                    r"i2p.*"
                ],
                window_patterns=[
                    r".*VPN.*",
                    r".*Tor.*",
                    r".*Proxy.*",
                    r".*Private.*"
                ],
                priority=8
            ),
            SensitiveApp(
                app_name="Medical & Health",
                process_patterns=[
                    r".*health.*",
                    r".*medical.*",
                    r".*patient.*"
                ],
                window_patterns=[
                    r".*Health.*",
                    r".*Medical.*",
                    r".*Patient.*",
                    r".*HIPAA.*"
                ],
                priority=8
            ),
            SensitiveApp(
                app_name="Secure Communication",
                process_patterns=[
                    r"signal.*",
                    r"telegram.*",
                    r"wickr.*",
                    r"protonmail.*",
                    r".*encrypted.*"
                ],
                window_patterns=[
                    r"Signal.*",
                    r"Telegram.*",
                    r"Wickr.*",
                    r"ProtonMail.*",
                    r".*Encrypted.*"
                ],
                priority=7
            )
        ]
        
        for app in default_apps:
            self.sensitive_apps[app.app_name] = app
    
    def _load_consent_data(self):
        """Load consent records from storage"""
        try:
            if self.consent_file.exists():
                with open(self.consent_file, 'r') as f:
                    data = json.load(f)
                
                for record_data in data.get('consent_records', []):
                    record = ConsentRecord(
                        consent_id=record_data['consent_id'],
                        feature=record_data['feature'],
                        consent_level=ConsentLevel(record_data['consent_level']),
                        granted_timestamp=record_data['granted_timestamp'],
                        expires_timestamp=record_data.get('expires_timestamp'),
                        metadata=record_data.get('metadata', {})
                    )
                    self.consent_records[record.consent_id] = record
                
                # Load monitoring state
                state_value = data.get('monitoring_state', 'disabled')
                self.monitoring_state = MonitoringState(state_value)
                
                self.logger.info(f"Loaded {len(self.consent_records)} consent records")
        
        except Exception as e:
            self.logger.error(f"Failed to load consent data: {e}")
    
    def _save_consent_data(self):
        """Save consent records to storage"""
        try:
            data = {
                'consent_records': [],
                'monitoring_state': self.monitoring_state.value,
                'last_updated': time.time()
            }
            
            for record in self.consent_records.values():
                data['consent_records'].append({
                    'consent_id': record.consent_id,
                    'feature': record.feature,
                    'consent_level': record.consent_level.value,
                    'granted_timestamp': record.granted_timestamp,
                    'expires_timestamp': record.expires_timestamp,
                    'metadata': record.metadata
                })
            
            with open(self.consent_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.debug("Saved consent data")
        
        except Exception as e:
            self.logger.error(f"Failed to save consent data: {e}")
    
    def _load_privacy_zones(self):
        """Load privacy zones from storage"""
        try:
            if self.privacy_zones_file.exists():
                with open(self.privacy_zones_file, 'r') as f:
                    data = json.load(f)
                
                for zone_data in data.get('privacy_zones', []):
                    zone = PrivacyZone(
                        zone_id=zone_data['zone_id'],
                        name=zone_data['name'],
                        x=zone_data['x'],
                        y=zone_data['y'],
                        width=zone_data['width'],
                        height=zone_data['height'],
                        zone_type=PrivacyZoneType(zone_data['zone_type']),
                        active=zone_data.get('active', True),
                        created_timestamp=zone_data.get('created_timestamp', time.time())
                    )
                    self.privacy_zones[zone.zone_id] = zone
                
                self.logger.info(f"Loaded {len(self.privacy_zones)} privacy zones")
        
        except Exception as e:
            self.logger.error(f"Failed to load privacy zones: {e}")
    
    def _save_privacy_zones(self):
        """Save privacy zones to storage"""
        try:
            data = {
                'privacy_zones': [],
                'last_updated': time.time()
            }
            
            for zone in self.privacy_zones.values():
                data['privacy_zones'].append({
                    'zone_id': zone.zone_id,
                    'name': zone.name,
                    'x': zone.x,
                    'y': zone.y,
                    'width': zone.width,
                    'height': zone.height,
                    'zone_type': zone.zone_type.value,
                    'active': zone.active,
                    'created_timestamp': zone.created_timestamp
                })
            
            with open(self.privacy_zones_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.debug("Saved privacy zones")
        
        except Exception as e:
            self.logger.error(f"Failed to save privacy zones: {e}")
    
    # Consent Management Methods
    
    def request_privacy_consent(self, feature: str, required_level: ConsentLevel = ConsentLevel.STANDARD,
                               expires_hours: Optional[int] = None) -> ConsentLevel:
        """Request user consent for a feature"""
        consent_id = f"{feature}_{int(time.time())}"
        
        # Check for existing valid consent
        existing = self.get_current_consent(feature)
        if existing and existing.consent_level.value >= required_level.value:
            return existing.consent_level
        
        # For now, auto-grant basic consent (in real implementation, this would show UI)
        granted_level = ConsentLevel.STANDARD
        expires_timestamp = None
        
        if expires_hours:
            expires_timestamp = time.time() + (expires_hours * 3600)
        
        record = ConsentRecord(
            consent_id=consent_id,
            feature=feature,
            consent_level=granted_level,
            granted_timestamp=time.time(),
            expires_timestamp=expires_timestamp,
            metadata={'auto_granted': True}
        )
        
        self.consent_records[consent_id] = record
        self._save_consent_data()
        
        # Notify callbacks
        for callback in self.consent_change_callbacks:
            try:
                callback(feature, granted_level)
            except Exception as e:
                self.logger.error(f"Consent callback failed: {e}")
        
        self.logger.info(f"Granted {granted_level.value} consent for {feature}")
        return granted_level
    
    def revoke_consent(self, feature: str):
        """Revoke consent for a feature"""
        # Find and revoke all consent records for this feature
        revoked_count = 0
        for consent_id, record in list(self.consent_records.items()):
            if record.feature == feature:
                record.consent_level = ConsentLevel.DENIED
                revoked_count += 1
        
        if revoked_count > 0:
            self._save_consent_data()
            
            # Notify callbacks
            for callback in self.consent_change_callbacks:
                try:
                    callback(feature, ConsentLevel.DENIED)
                except Exception as e:
                    self.logger.error(f"Consent callback failed: {e}")
            
            self.logger.info(f"Revoked consent for {feature}")
    
    def get_current_consent(self, feature: str) -> Optional[ConsentRecord]:
        """Get the current valid consent for a feature"""
        valid_consents = [
            record for record in self.consent_records.values()
            if record.feature == feature and record.is_valid()
        ]
        
        if not valid_consents:
            return None
        
        # Return the most recent valid consent
        return max(valid_consents, key=lambda r: r.granted_timestamp)
    
    def has_consent(self, feature: str, required_level: ConsentLevel = ConsentLevel.BASIC) -> bool:
        """Check if we have sufficient consent for a feature"""
        current = self.get_current_consent(feature)
        if not current:
            return False
        
        # Compare consent levels (higher numeric value = more permissive)
        level_values = {
            ConsentLevel.DENIED: 0,
            ConsentLevel.BASIC: 1,
            ConsentLevel.STANDARD: 2,
            ConsentLevel.ENHANCED: 3,
            ConsentLevel.FULL: 4
        }
        
        return level_values[current.consent_level] >= level_values[required_level]
    
    # Privacy Zone Management
    
    def add_privacy_zone(self, name: str, x: int, y: int, width: int, height: int,
                        zone_type: PrivacyZoneType = PrivacyZoneType.BLUR) -> str:
        """Add a new privacy zone"""
        zone_id = f"zone_{int(time.time())}_{len(self.privacy_zones)}"
        
        zone = PrivacyZone(
            zone_id=zone_id,
            name=name,
            x=x,
            y=y,
            width=width,
            height=height,
            zone_type=zone_type
        )
        
        self.privacy_zones[zone_id] = zone
        self._save_privacy_zones()
        
        self.logger.info(f"Added privacy zone: {name} ({zone_type.value})")
        return zone_id
    
    def remove_privacy_zone(self, zone_id: str) -> bool:
        """Remove a privacy zone"""
        if zone_id in self.privacy_zones:
            del self.privacy_zones[zone_id]
            self._save_privacy_zones()
            self.logger.info(f"Removed privacy zone: {zone_id}")
            return True
        return False
    
    def get_privacy_zones_for_region(self, x: int, y: int, width: int, height: int) -> List[PrivacyZone]:
        """Get all privacy zones that overlap with a screen region"""
        overlapping_zones = []
        for zone in self.privacy_zones.values():
            if zone.active and zone.overlaps_region(x, y, width, height):
                overlapping_zones.append(zone)
        return overlapping_zones
    
    # Sensitive App Detection
    
    async def scan_for_sensitive_apps(self) -> List[SensitiveApp]:
        """Scan for currently running sensitive applications"""
        if not self.enable_sensitive_app_detection:
            return []
            
        detected_apps = []
        
        try:
            # Scan running processes
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    process_name = proc.info['name']
                    
                    for app in self.sensitive_apps.values():
                        if app.matches_process(process_name):
                            detected_apps.append(app)
                            break
                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # TODO: Add window title scanning using platform-specific APIs
            # This would require additional platform-specific implementations
            
        except Exception as e:
            self.logger.error(f"Failed to scan for sensitive apps: {e}")
        
        return detected_apps
    
    def add_sensitive_app(self, app: SensitiveApp):
        """Add a custom sensitive application configuration"""
        self.sensitive_apps[app.app_name] = app
        self.logger.info(f"Added sensitive app configuration: {app.app_name}")
    
    # Monitoring State Management
    
    def set_monitoring_state(self, state: MonitoringState, reason: str = ""):
        """Set the monitoring state"""
        old_state = self.monitoring_state
        self.monitoring_state = state
        
        self._save_consent_data()
        
        # Notify callbacks
        for callback in self.state_change_callbacks:
            try:
                callback(state)
            except Exception as e:
                self.logger.error(f"State change callback failed: {e}")
        
        self.logger.info(f"Monitoring state changed: {old_state.value} -> {state.value} ({reason})")
    
    def pause_monitoring(self, reason: str = "user_request"):
        """Pause monitoring"""
        if reason == "sensitive_app":
            self.set_monitoring_state(MonitoringState.PAUSED_SENSITIVE, reason)
        elif reason == "privacy_zone":
            self.set_monitoring_state(MonitoringState.PAUSED_PRIVACY, reason)
        else:
            self.set_monitoring_state(MonitoringState.PAUSED_USER, reason)
    
    def resume_monitoring(self):
        """Resume monitoring if consent allows"""
        if self.has_consent("screen_monitoring"):
            self.set_monitoring_state(MonitoringState.ACTIVE, "user_resume")
        else:
            self.logger.warning("Cannot resume monitoring without consent")
    
    def get_monitoring_state(self) -> MonitoringState:
        """Get the current monitoring state"""
        return self.monitoring_state
    
    def is_monitoring_allowed(self) -> bool:
        """Check if monitoring is currently allowed"""
        return (self.monitoring_state == MonitoringState.ACTIVE and 
                self.has_consent("screen_monitoring"))
    
    # Event Management
    
    def add_state_change_callback(self, callback: Callable[[MonitoringState], None]):
        """Add callback for monitoring state changes"""
        self.state_change_callbacks.append(callback)
    
    def add_consent_change_callback(self, callback: Callable[[str, ConsentLevel], None]):
        """Add callback for consent changes"""
        self.consent_change_callbacks.append(callback)
    
    # Data Management
    
    def get_consent_summary(self) -> Dict[str, Any]:
        """Get summary of current consent status"""
        active_consents = {}
        expired_consents = []
        
        for record in self.consent_records.values():
            if record.is_valid():
                active_consents[record.feature] = {
                    'level': record.consent_level.value,
                    'granted': record.granted_timestamp,
                    'expires': record.expires_timestamp
                }
            elif record.is_expired():
                expired_consents.append(record.feature)
        
        return {
            'monitoring_state': self.monitoring_state.value,
            'active_consents': active_consents,
            'expired_consents': expired_consents,
            'privacy_zones': len(self.privacy_zones),
            'sensitive_apps': len(self.sensitive_apps)
        }
    
    def clear_expired_consents(self):
        """Remove expired consent records"""
        expired_ids = [
            consent_id for consent_id, record in self.consent_records.items()
            if record.is_expired()
        ]
        
        for consent_id in expired_ids:
            del self.consent_records[consent_id]
        
        if expired_ids:
            self._save_consent_data()
            self.logger.info(f"Cleared {len(expired_ids)} expired consent records")
    
    async def cleanup(self):
        """Cleanup resources"""
        self.clear_expired_consents()
        self._save_consent_data()
        self._save_privacy_zones()
        self.logger.info("ConsentManager cleanup complete")
    
    # External Model Routing Consent Methods (backwards compatibility)
    
    def set_gui_callback(self, callback: Callable):
        """Set the GUI consent callback for external routing"""
        self.gui_consent_callback = callback
    
    async def request_consent(self, query: str, routing_decision: Any) -> bool:
        """Request user consent for external model routing"""
        consent_request = ConsentRequest(
            query=query,
            routing_decision=routing_decision,
            provider=getattr(routing_decision, 'provider', 'external')
        )
        
        # Add to history
        self.consent_history.append(consent_request)
        
        # Handle consent based on method
        try:
            if self.method == ConsentMethod.AUTO_APPROVE:
                return True
            elif self.method == ConsentMethod.AUTO_DENY:
                return False
            elif self.method == ConsentMethod.GUI:
                if self.gui_consent_callback:
                    return await self.gui_consent_callback(consent_request)
                else:
                    # Fallback to CLI if no GUI callback
                    return await cli_consent_callback(consent_request)
            else:  # CLI method
                return await cli_consent_callback(consent_request)
                
        except Exception as e:
            self.logger.error(f"Error in consent request: {e}")
            return False
    
    def _format_criteria(self, criteria) -> str:
        """Format external routing criteria for display"""
        criteria_map = {
            "SPECIALIZED_KNOWLEDGE": "Specialized Knowledge",
            "COMPLEX_TOOL_USE": "Complex Tool Use", 
            "USER_EXPLICIT_REQUEST": "Explicit Request",
            "LOCAL_MODEL_FAILURE": "Local Model Limitation",
            "RECENT_INFORMATION": "Recent Information"
        }
        
        if hasattr(criteria, 'value'):
            return criteria_map.get(criteria.value, criteria.value)
        else:
            return criteria_map.get(str(criteria), str(criteria))
    
    def _show_detailed_info(self, consent_request: ConsentRequest):
        """Show detailed information about consent request"""
        print(f"\n📋 Detailed Information:")
        print(f"  Query: {consent_request.query}")
        print(f"  Provider: {consent_request.provider}")
        print(f"  Reasoning: {consent_request.routing_decision.reasoning}")
        print(f"  Confidence: {consent_request.routing_decision.confidence:.2f}")
        print(f"  Criteria: {[self._format_criteria(c) for c in consent_request.routing_decision.criteria]}")
        print(f"  Timestamp: {consent_request.timestamp}")
    
    def get_consent_history(self) -> List[ConsentRequest]:
        """Get the consent request history"""
        return self.consent_history.copy()
    
    def clear_consent_history(self):
        """Clear the consent request history"""
        self.consent_history.clear() 