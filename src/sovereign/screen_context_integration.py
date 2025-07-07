"""
Screen Context Integration - Bridge between screen context system and AI architecture
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import hashlib

from .screen_context_manager import ScreenContextManager, ScreenCapture, ScreenReference, ContextRelevance
from .config import Config
from .consent_manager import ConsentLevel, MonitoringState


class ContextAccessLevel(Enum):
    """Access levels for screen context data"""
    NONE = "none"           # No access to screen context
    BASIC = "basic"         # Basic text only, no images
    STANDARD = "standard"   # Text + basic screen references
    ENHANCED = "enhanced"   # Full access including images and detailed references
    FULL = "full"          # Complete access with historical data


class ContextUpdateType(Enum):
    """Types of context updates"""
    NEW_CAPTURE = "new_capture"
    TEXT_CHANGE = "text_change"
    RELEVANCE_CHANGE = "relevance_change"
    PRIVACY_CHANGE = "privacy_change"
    APP_CHANGE = "app_change"


@dataclass
class ContextAccessRequest:
    """Request for accessing screen context data"""
    requester: str  # AI model name or component
    access_level: ContextAccessLevel
    max_age_seconds: Optional[int] = None
    text_filter: Optional[str] = None
    relevance_threshold: Optional[float] = None
    include_images: bool = False
    include_references: bool = True
    privacy_aware: bool = True


@dataclass
class ContextResponse:
    """Response containing screen context data"""
    success: bool
    data: Dict[str, Any]
    access_level: ContextAccessLevel
    timestamp: float
    captures_count: int
    references_count: int
    privacy_filtered: bool
    error_message: Optional[str] = None


@dataclass
class ScreenElementReference:
    """AI-friendly reference to screen elements"""
    ref_id: str
    element_type: str
    text_content: str
    position: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    capture_timestamp: float
    semantic_description: str
    interaction_hints: List[str] = field(default_factory=list)


@dataclass
class ContextUpdateNotification:
    """Notification about context changes"""
    update_type: ContextUpdateType
    timestamp: float
    details: Dict[str, Any]
    relevant_captures: List[str]
    significance_score: float


class ScreenContextIntegration:
    """
    Integration layer between screen context system and AI architecture.
    
    This class provides AI-friendly APIs for accessing screen context data,
    manages privacy-aware access control, and handles real-time updates.
    """
    
    def __init__(self, screen_context_manager: ScreenContextManager, config: Config):
        self.screen_manager = screen_context_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Access control and privacy
        self.access_policies: Dict[str, ContextAccessLevel] = {
            "orchestrator": ContextAccessLevel.ENHANCED,
            "thinker_model": ContextAccessLevel.STANDARD,
            "talker_model": ContextAccessLevel.BASIC,
            "external_model": ContextAccessLevel.NONE,
        }
        
        # Update notification system
        self.update_callbacks: List[callable] = []
        self.last_significant_update = 0.0
        self.update_debounce_time = 2.0  # Seconds
        
        # Performance tracking
        self.access_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "privacy_blocked": 0,
            "cache_hits": 0,
            "avg_response_time": 0.0,
            "access_by_level": {level.value: 0 for level in ContextAccessLevel}
        }
        
        # Context cache for performance
        self.context_cache: Dict[str, Tuple[ContextResponse, float]] = {}
        self.cache_ttl = 5.0  # Cache for 5 seconds
        
        # AI-friendly context formatting
        self.context_formatters = {
            "text_summary": self._format_text_summary,
            "element_list": self._format_element_list,
            "screen_description": self._format_screen_description,
            "interaction_context": self._format_interaction_context,
        }
        
        self.logger.info("ScreenContextIntegration initialized")
    
    async def initialize(self) -> bool:
        """Initialize the integration layer"""
        try:
            # Ensure screen context manager is initialized
            if not await self.screen_manager.initialize():
                self.logger.error("Failed to initialize screen context manager")
                return False
            
            # Setup update monitoring
            await self._setup_update_monitoring()
            
            # Register privacy change callbacks
            self._setup_privacy_callbacks()
            
            self.logger.info("ScreenContextIntegration initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ScreenContextIntegration: {e}")
            return False
    
    async def get_current_context(self, request: ContextAccessRequest) -> ContextResponse:
        """
        Get current screen context for AI use.
        
        Args:
            request: Context access request with permissions and filters
            
        Returns:
            ContextResponse with screen context data
        """
        start_time = time.time()
        self.access_stats["total_requests"] += 1
        
        try:
            # Check access permissions
            if not self._check_access_permission(request):
                self.access_stats["privacy_blocked"] += 1
                return ContextResponse(
                    success=False,
                    data={},
                    access_level=ContextAccessLevel.NONE,
                    timestamp=time.time(),
                    captures_count=0,
                    references_count=0,
                    privacy_filtered=True,
                    error_message="Access denied due to privacy settings"
                )
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                self.access_stats["cache_hits"] += 1
                return cached_response
            
            # Get recent captures based on access level
            captures = await self._get_filtered_captures(request)
            
            # Get screen references if requested
            references = []
            if request.include_references:
                references = await self._get_filtered_references(request)
            
            # Format context data
            context_data = await self._format_context_data(
                captures, references, request.access_level
            )
            
            # Apply privacy filtering
            context_data, privacy_filtered = self._apply_privacy_filtering(
                context_data, request.privacy_aware
            )
            
            # Create response
            response = ContextResponse(
                success=True,
                data=context_data,
                access_level=request.access_level,
                timestamp=time.time(),
                captures_count=len(captures),
                references_count=len(references),
                privacy_filtered=privacy_filtered
            )
            
            # Cache the response
            self._cache_response(cache_key, response)
            
            # Update stats
            self.access_stats["successful_requests"] += 1
            self.access_stats["access_by_level"][request.access_level.value] += 1
            
            elapsed_time = time.time() - start_time
            self.access_stats["avg_response_time"] = (
                (self.access_stats["avg_response_time"] * (self.access_stats["successful_requests"] - 1) + elapsed_time)
                / self.access_stats["successful_requests"]
            )
            
            self.logger.debug(f"Context request processed in {elapsed_time:.3f}s")
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing context request: {e}")
            return ContextResponse(
                success=False,
                data={},
                access_level=ContextAccessLevel.NONE,
                timestamp=time.time(),
                captures_count=0,
                references_count=0,
                privacy_filtered=False,
                error_message=f"Internal error: {str(e)}"
            )
    
    async def get_screen_element_references(self, 
                                          element_type: Optional[str] = None,
                                          text_filter: Optional[str] = None,
                                          max_age_seconds: int = 300) -> List[ScreenElementReference]:
        """
        Get AI-friendly references to screen elements.
        
        Args:
            element_type: Type of elements to retrieve (text, image, ui_element)
            text_filter: Filter elements by text content
            max_age_seconds: Maximum age of elements to include
            
        Returns:
            List of screen element references
        """
        try:
            # Get screen references from manager
            references = await self.screen_manager.get_screen_references(
                element_type=element_type,
                max_age_hours=max_age_seconds / 3600
            )
            
            # Convert to AI-friendly format
            ai_references = []
            for ref in references:
                # Apply text filter if specified
                if text_filter and text_filter.lower() not in ref.text_content.lower():
                    continue
                
                # Create AI-friendly reference
                ai_ref = ScreenElementReference(
                    ref_id=ref.reference_id,
                    element_type=ref.element_type,
                    text_content=ref.text_content,
                    position=ref.bounding_box,
                    confidence=ref.confidence,
                    capture_timestamp=0.0,  # Will be filled from capture data
                    semantic_description=self._generate_semantic_description(ref),
                    interaction_hints=self._generate_interaction_hints(ref)
                )
                
                ai_references.append(ai_ref)
            
            return ai_references
            
        except Exception as e:
            self.logger.error(f"Error getting screen element references: {e}")
            return []
    
    async def search_context_by_text(self, 
                                   search_term: str,
                                   limit: int = 10,
                                   requester: str = "unknown") -> List[Dict[str, Any]]:
        """
        Search screen context by text content.
        
        Args:
            search_term: Text to search for
            limit: Maximum number of results
            requester: Name of the requesting component
            
        Returns:
            List of matching context items
        """
        try:
            # Check access permissions
            access_level = self.access_policies.get(requester, ContextAccessLevel.NONE)
            if access_level == ContextAccessLevel.NONE:
                return []
            
            # Search captures
            captures = await self.screen_manager.search_captures_by_text(search_term, limit)
            
            # Format results
            results = []
            for capture in captures:
                result = {
                    "capture_id": capture.id,
                    "timestamp": capture.timestamp,
                    "text_content": capture.text_content,
                    "relevance_score": capture.relevance_score,
                    "match_type": "text_search"
                }
                
                # Add additional data based on access level
                if access_level in [ContextAccessLevel.ENHANCED, ContextAccessLevel.FULL]:
                    result["bounding_boxes"] = capture.bounding_boxes
                    result["ocr_confidence"] = capture.ocr_confidence
                
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching context by text: {e}")
            return []
    
    async def get_context_summary(self, 
                                format_type: str = "text_summary",
                                max_age_seconds: int = 300) -> Optional[str]:
        """
        Get a formatted summary of current screen context.
        
        Args:
            format_type: Type of formatting (text_summary, element_list, screen_description)
            max_age_seconds: Maximum age of context to include
            
        Returns:
            Formatted context summary
        """
        try:
            # Get recent captures
            captures = self.screen_manager.get_recent_captures(limit=5)
            
            # Filter by age
            current_time = time.time()
            recent_captures = [
                c for c in captures 
                if current_time - c.timestamp <= max_age_seconds
            ]
            
            if not recent_captures:
                return "No recent screen context available."
            
            # Apply formatter
            formatter = self.context_formatters.get(format_type, self._format_text_summary)
            return await formatter(recent_captures)
            
        except Exception as e:
            self.logger.error(f"Error getting context summary: {e}")
            return None
    
    def register_update_callback(self, callback: callable):
        """Register a callback for context updates"""
        self.update_callbacks.append(callback)
    
    def set_access_policy(self, requester: str, access_level: ContextAccessLevel):
        """Set access policy for a requester"""
        self.access_policies[requester] = access_level
        self.logger.info(f"Set access policy for {requester}: {access_level.value}")
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration performance statistics"""
        return {
            "access_stats": self.access_stats.copy(),
            "cache_stats": {
                "total_entries": len(self.context_cache),
                "hit_rate": self.access_stats["cache_hits"] / max(self.access_stats["total_requests"], 1)
            },
            "update_stats": {
                "last_significant_update": self.last_significant_update,
                "registered_callbacks": len(self.update_callbacks)
            },
            "privacy_stats": {
                "privacy_blocked_rate": self.access_stats["privacy_blocked"] / max(self.access_stats["total_requests"], 1),
                "monitoring_state": self.screen_manager.consent_manager.get_monitoring_state().value
            }
        }
    
    # Private methods
    
    def _check_access_permission(self, request: ContextAccessRequest) -> bool:
        """Check if the request has permission to access context"""
        # Check global access policy
        allowed_level = self.access_policies.get(request.requester, ContextAccessLevel.NONE)
        
        # Check if requested level is allowed
        if request.access_level.value not in [level.value for level in ContextAccessLevel 
                                            if level.value <= allowed_level.value]:
            return False
        
        # Check privacy settings
        if request.privacy_aware:
            monitoring_state = self.screen_manager.consent_manager.get_monitoring_state()
            if monitoring_state in [MonitoringState.PAUSED_PRIVACY, MonitoringState.PAUSED_USER]:
                return False
        
        return True
    
    def _generate_cache_key(self, request: ContextAccessRequest) -> str:
        """Generate cache key for request"""
        key_data = f"{request.requester}_{request.access_level.value}_{request.max_age_seconds}_{request.text_filter}_{request.relevance_threshold}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[ContextResponse]:
        """Get cached response if valid"""
        if cache_key in self.context_cache:
            response, timestamp = self.context_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return response
            else:
                # Remove expired cache entry
                del self.context_cache[cache_key]
        return None
    
    def _cache_response(self, cache_key: str, response: ContextResponse):
        """Cache response with timestamp"""
        self.context_cache[cache_key] = (response, time.time())
        
        # Clean up old cache entries
        if len(self.context_cache) > 100:
            self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.context_cache.items()
            if current_time - timestamp > self.cache_ttl
        ]
        for key in expired_keys:
            del self.context_cache[key]
    
    async def _get_filtered_captures(self, request: ContextAccessRequest) -> List[ScreenCapture]:
        """Get captures filtered by request parameters"""
        # Get recent captures
        captures = self.screen_manager.get_recent_captures(limit=20)
        
        # Apply filters
        filtered_captures = []
        for capture in captures:
            # Age filter
            if request.max_age_seconds:
                if time.time() - capture.timestamp > request.max_age_seconds:
                    continue
            
            # Text filter
            if request.text_filter:
                if request.text_filter.lower() not in capture.text_content.lower():
                    continue
            
            # Relevance filter
            if request.relevance_threshold:
                if capture.relevance_score < request.relevance_threshold:
                    continue
            
            filtered_captures.append(capture)
        
        return filtered_captures
    
    async def _get_filtered_references(self, request: ContextAccessRequest) -> List[ScreenReference]:
        """Get screen references filtered by request parameters"""
        max_age_hours = None
        if request.max_age_seconds:
            max_age_hours = request.max_age_seconds / 3600
        
        return await self.screen_manager.get_screen_references(
            max_age_hours=max_age_hours
        )
    
    async def _format_context_data(self, 
                                 captures: List[ScreenCapture],
                                 references: List[ScreenReference],
                                 access_level: ContextAccessLevel) -> Dict[str, Any]:
        """Format context data based on access level"""
        context_data = {
            "timestamp": time.time(),
            "captures_count": len(captures),
            "references_count": len(references)
        }
        
        if access_level == ContextAccessLevel.BASIC:
            # Basic: text only
            context_data["text_content"] = "\n".join([c.text_content for c in captures if c.text_content])
            
        elif access_level == ContextAccessLevel.STANDARD:
            # Standard: text + basic references
            context_data["text_content"] = "\n".join([c.text_content for c in captures if c.text_content])
            context_data["references"] = [
                {
                    "ref_id": ref.reference_id,
                    "text": ref.text_content,
                    "type": ref.element_type,
                    "confidence": ref.confidence
                }
                for ref in references
            ]
            
        elif access_level in [ContextAccessLevel.ENHANCED, ContextAccessLevel.FULL]:
            # Enhanced/Full: complete data
            context_data["captures"] = [
                {
                    "id": c.id,
                    "timestamp": c.timestamp,
                    "text_content": c.text_content,
                    "bounding_boxes": c.bounding_boxes,
                    "relevance_score": c.relevance_score,
                    "ocr_confidence": c.ocr_confidence
                }
                for c in captures
            ]
            
            context_data["references"] = [
                {
                    "ref_id": ref.reference_id,
                    "capture_id": ref.capture_id,
                    "text": ref.text_content,
                    "type": ref.element_type,
                    "position": ref.bounding_box,
                    "confidence": ref.confidence,
                    "semantic_tags": ref.semantic_tags
                }
                for ref in references
            ]
            
            # Add summary
            context_data["summary"] = await self._format_text_summary(captures)
        
        return context_data
    
    def _apply_privacy_filtering(self, 
                               context_data: Dict[str, Any],
                               privacy_aware: bool) -> Tuple[Dict[str, Any], bool]:
        """Apply privacy filtering to context data"""
        if not privacy_aware:
            return context_data, False
        
        # Check if sensitive content should be filtered
        # This is a simplified implementation - real filtering would be more sophisticated
        privacy_filtered = False
        
        # Filter sensitive text patterns
        sensitive_patterns = [
            "password", "credit card", "social security", "ssn",
            "bank account", "routing number", "api key", "token"
        ]
        
        if "text_content" in context_data:
            original_text = context_data["text_content"]
            filtered_text = original_text
            
            for pattern in sensitive_patterns:
                if pattern in original_text.lower():
                    filtered_text = filtered_text.replace(pattern, "[REDACTED]")
                    privacy_filtered = True
            
            context_data["text_content"] = filtered_text
        
        return context_data, privacy_filtered
    
    def _generate_semantic_description(self, reference: ScreenReference) -> str:
        """Generate semantic description for screen reference"""
        desc_parts = []
        
        if reference.element_type == "text":
            desc_parts.append(f"Text element containing: '{reference.text_content[:50]}...'")
        elif reference.element_type == "ui_element":
            desc_parts.append(f"UI element with text: '{reference.text_content[:30]}...'")
        else:
            desc_parts.append(f"{reference.element_type} element")
        
        if reference.confidence > 0.8:
            desc_parts.append("(high confidence)")
        elif reference.confidence > 0.5:
            desc_parts.append("(medium confidence)")
        else:
            desc_parts.append("(low confidence)")
        
        return " ".join(desc_parts)
    
    def _generate_interaction_hints(self, reference: ScreenReference) -> List[str]:
        """Generate interaction hints for screen reference"""
        hints = []
        
        if reference.element_type == "text":
            hints.append("Can be referenced in responses")
            hints.append("May contain selectable text")
        elif reference.element_type == "ui_element":
            hints.append("Interactive element")
            hints.append("May be clickable")
        
        if "button" in reference.text_content.lower():
            hints.append("Appears to be a button")
        elif "link" in reference.text_content.lower():
            hints.append("Appears to be a link")
        
        return hints
    
    async def _format_text_summary(self, captures: List[ScreenCapture]) -> str:
        """Format captures as text summary"""
        if not captures:
            return "No screen context available."
        
        latest_capture = captures[0]
        text_content = latest_capture.text_content
        
        if not text_content.strip():
            return "Current screen contains no readable text."
        
        # Create summary
        summary_lines = [
            f"Current screen context (captured {datetime.fromtimestamp(latest_capture.timestamp).strftime('%H:%M:%S')}):",
            "",
            text_content[:500] + ("..." if len(text_content) > 500 else "")
        ]
        
        return "\n".join(summary_lines)
    
    async def _format_element_list(self, captures: List[ScreenCapture]) -> str:
        """Format captures as element list"""
        if not captures:
            return "No screen elements available."
        
        elements = []
        for capture in captures:
            if capture.text_content.strip():
                elements.append(f"- {capture.text_content[:100]}...")
        
        if not elements:
            return "No readable elements found on screen."
        
        return "Screen elements:\n" + "\n".join(elements[:10])
    
    async def _format_screen_description(self, captures: List[ScreenCapture]) -> str:
        """Format captures as screen description"""
        if not captures:
            return "No screen information available."
        
        latest_capture = captures[0]
        
        description = f"Screen captured at {datetime.fromtimestamp(latest_capture.timestamp).strftime('%H:%M:%S')} "
        description += f"with {len(latest_capture.bounding_boxes)} text elements detected. "
        
        if latest_capture.text_content.strip():
            description += f"Main content: {latest_capture.text_content[:200]}..."
        else:
            description += "No readable text detected."
        
        return description
    
    async def _format_interaction_context(self, captures: List[ScreenCapture]) -> str:
        """Format captures for interaction context"""
        if not captures:
            return "No interaction context available."
        
        context_parts = []
        for capture in captures[:3]:  # Last 3 captures
            if capture.text_content.strip():
                context_parts.append(f"[{datetime.fromtimestamp(capture.timestamp).strftime('%H:%M:%S')}] {capture.text_content[:150]}...")
        
        if not context_parts:
            return "No interaction context available."
        
        return "Recent screen context:\n" + "\n".join(context_parts)
    
    async def _setup_update_monitoring(self):
        """Setup monitoring for context updates"""
        # This would typically involve setting up callbacks with the screen manager
        # For now, we'll implement a simple polling mechanism
        pass
    
    def _setup_privacy_callbacks(self):
        """Setup privacy-related callbacks"""
        def on_privacy_change(state: MonitoringState):
            """Handle privacy state changes"""
            if state in [MonitoringState.PAUSED_PRIVACY, MonitoringState.PAUSED_USER]:
                # Clear cache when privacy is enabled
                self.context_cache.clear()
                self.logger.info("Cleared context cache due to privacy state change")
        
        self.screen_manager.consent_manager.add_state_change_callback(on_privacy_change)
    
    async def cleanup(self):
        """Cleanup resources"""
        self.context_cache.clear()
        self.update_callbacks.clear()
        self.logger.info("ScreenContextIntegration cleanup complete") 