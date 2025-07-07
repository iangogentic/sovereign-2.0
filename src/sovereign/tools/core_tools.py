"""
Core Tool Implementations - Essential tools for Sovereign AI Agent

This module contains the core set of tools that provide essential functionality
for the Sovereign AI system, including internet search, system information,
file access, and advanced calculations.
"""

import asyncio
import json
import os
import subprocess
import time
import math
import statistics
import re
import requests
import platform
import psutil
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from pathlib import Path

from ..tool_integration_framework import (
    BaseTool, ToolMetadata, ToolParameter, ToolType, 
    ToolSecurityLevel, ToolExecutionMode, ToolExecutionError
)


class InternetSearchTool(BaseTool):
    """Privacy-focused internet search tool using DuckDuckGo API"""
    
    def __init__(self):
        metadata = ToolMetadata(
            name="internet_search",
            description="Search the internet using privacy-focused DuckDuckGo search engine",
            version="1.0.0",
            tool_type=ToolType.API,
            security_level=ToolSecurityLevel.MODERATE,
            execution_mode=ToolExecutionMode.ASYNCHRONOUS,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query string",
                    required=True
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum number of results to return",
                    required=False,
                    default=5,
                    allowed_values=list(range(1, 21))  # 1-20 results
                ),
                ToolParameter(
                    name="region",
                    type="string", 
                    description="Search region code",
                    required=False,
                    default="us-en",
                    allowed_values=["us-en", "uk-en", "ca-en", "au-en", "de-de", "fr-fr"]
                ),
                ToolParameter(
                    name="safe_search",
                    type="string",
                    description="Safe search setting",
                    required=False,
                    default="moderate",
                    allowed_values=["strict", "moderate", "off"]
                )
            ],
            return_type="object",
            examples=[
                {
                    "input": {"query": "Python programming tutorial", "max_results": 3},
                    "output": {
                        "query": "Python programming tutorial",
                        "results": [
                            {
                                "title": "Python Tutorial - Official Documentation",
                                "url": "https://docs.python.org/tutorial/",
                                "snippet": "The Python Tutorial — Python 3.x documentation"
                            }
                        ],
                        "total_results": 3,
                        "search_time": 1.2
                    }
                }
            ],
            tags=["search", "internet", "research", "privacy"],
            max_execution_time=30,
            rate_limit=30,  # 30 searches per minute
            requires_consent=True,
            data_handling="Sends search queries to DuckDuckGo API. No personal data stored."
        )
        super().__init__(metadata)
        
    async def execute(self, parameters: Dict[str, Any]) -> Any:
        """Execute internet search using DuckDuckGo"""
        query = parameters["query"]
        max_results = parameters.get("max_results", 5)
        region = parameters.get("region", "us-en")
        safe_search = parameters.get("safe_search", "moderate")
        
        start_time = time.time()
        
        try:
            # DuckDuckGo Instant Answer API
            api_url = "https://api.duckduckgo.com/"
            
            # Safe search mapping
            safe_search_map = {"strict": "1", "moderate": "1", "off": "-1"}
            
            params = {
                "q": query,
                "format": "json",
                "no_redirect": "1",
                "no_html": "1",
                "skip_disambig": "1",
                "safe_search": safe_search_map.get(safe_search, "1")
            }
            
            # Make API request with timeout
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: requests.get(api_url, params=params, timeout=15)
            )
            
            if response.status_code != 200:
                raise ToolExecutionError(f"Search API returned status code: {response.status_code}")
            
            data = response.json()
            
            # Process results
            results = []
            
            # Add instant answer if available
            if data.get("Abstract"):
                results.append({
                    "title": data.get("AbstractText", "")[:100] + "...",
                    "url": data.get("AbstractURL", ""),
                    "snippet": data.get("Abstract", ""),
                    "type": "instant_answer"
                })
            
            # Add related topics
            for topic in data.get("RelatedTopics", [])[:max_results-len(results)]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("Text", "")[:100] + "...",
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", ""),
                        "type": "related_topic"
                    })
            
            # If no results from DuckDuckGo API, use web search fallback
            if not results:
                results = await self._fallback_search(query, max_results)
            
            search_time = time.time() - start_time
            
            result = {
                "query": query,
                "results": results[:max_results],
                "total_results": len(results),
                "search_time": round(search_time, 2),
                "region": region,
                "safe_search": safe_search,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Internet search completed: '{query}' -> {len(results)} results")
            return result
            
        except Exception as e:
            raise ToolExecutionError(f"Search failed: {e}")
    
    async def _fallback_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Fallback search implementation"""
        # Simple fallback - could be enhanced with additional search APIs
        return [
            {
                "title": f"Search results for: {query}",
                "url": f"https://duckduckgo.com/?q={query.replace(' ', '+')}",
                "snippet": f"No direct results available. Click to search on DuckDuckGo.",
                "type": "fallback"
            }
        ]
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate search parameters"""
        if "query" not in parameters:
            return False, "Missing required parameter: query"
        
        query = parameters["query"]
        if not isinstance(query, str) or not query.strip():
            return False, "Parameter 'query' must be a non-empty string"
        
        if len(query) > 500:
            return False, "Query too long (max 500 characters)"
        
        if "max_results" in parameters:
            max_results = parameters["max_results"]
            if not isinstance(max_results, int) or max_results < 1 or max_results > 20:
                return False, "Parameter 'max_results' must be an integer between 1 and 20"
        
        return True, None


class EnhancedSystemInfoTool(BaseTool):
    """Enhanced system information tool with comprehensive system details"""
    
    def __init__(self):
        metadata = ToolMetadata(
            name="enhanced_system_info",
            description="Get comprehensive system information including hardware, software, and performance metrics",
            version="1.0.0",
            tool_type=ToolType.SYSTEM,
            security_level=ToolSecurityLevel.MODERATE,
            execution_mode=ToolExecutionMode.SYNCHRONOUS,
            parameters=[
                ToolParameter(
                    name="info_category",
                    type="string",
                    description="Category of system information to retrieve",
                    required=False,
                    default="overview",
                    allowed_values=["overview", "hardware", "software", "performance", "network", "storage", "all"]
                ),
                ToolParameter(
                    name="include_sensitive",
                    type="boolean",
                    description="Include potentially sensitive information like IP addresses",
                    required=False,
                    default=False
                )
            ],
            return_type="object",
            examples=[
                {
                    "input": {"info_category": "overview"},
                    "output": {
                        "platform": "Windows",
                        "architecture": "x86_64",
                        "python_version": "3.11.0",
                        "cpu_count": 8,
                        "memory_total_gb": 16.0
                    }
                }
            ],
            tags=["system", "hardware", "diagnostics", "performance"],
            max_execution_time=20,
            requires_consent=True,
            data_handling="Collects system information for diagnostics and optimization purposes"
        )
        super().__init__(metadata)
    
    async def execute(self, parameters: Dict[str, Any]) -> Any:
        """Execute enhanced system info collection"""
        info_category = parameters.get("info_category", "overview")
        include_sensitive = parameters.get("include_sensitive", False)
        
        try:
            if info_category == "overview":
                result = await self._get_overview_info()
            elif info_category == "hardware":
                result = await self._get_hardware_info()
            elif info_category == "software":
                result = await self._get_software_info()
            elif info_category == "performance":
                result = await self._get_performance_info()
            elif info_category == "network":
                result = await self._get_network_info(include_sensitive)
            elif info_category == "storage":
                result = await self._get_storage_info()
            elif info_category == "all":
                result = {
                    "overview": await self._get_overview_info(),
                    "hardware": await self._get_hardware_info(),
                    "software": await self._get_software_info(),
                    "performance": await self._get_performance_info(),
                    "network": await self._get_network_info(include_sensitive),
                    "storage": await self._get_storage_info()
                }
            else:
                raise ToolExecutionError(f"Unknown info_category: {info_category}")
            
            result["timestamp"] = datetime.now().isoformat()
            result["category"] = info_category
            
            self.logger.info(f"System info collected: {info_category}")
            return result
            
        except Exception as e:
            raise ToolExecutionError(f"Failed to get system info: {e}")
    
    async def _get_overview_info(self) -> Dict[str, Any]:
        """Get basic system overview"""
        return {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "architecture": platform.machine(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
        }
    
    async def _get_hardware_info(self) -> Dict[str, Any]:
        """Get detailed hardware information"""
        cpu_freq = psutil.cpu_freq()
        return {
            "processor": platform.processor(),
            "cpu_frequency_mhz": {
                "current": round(cpu_freq.current, 2) if cpu_freq else None,
                "min": round(cpu_freq.min, 2) if cpu_freq else None,
                "max": round(cpu_freq.max, 2) if cpu_freq else None
            },
            "memory": {
                "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "used_percent": psutil.virtual_memory().percent
            },
            "swap": {
                "total_gb": round(psutil.swap_memory().total / (1024**3), 2),
                "used_gb": round(psutil.swap_memory().used / (1024**3), 2),
                "used_percent": psutil.swap_memory().percent
            }
        }
    
    async def _get_software_info(self) -> Dict[str, Any]:
        """Get software and environment information"""
        return {
            "python_implementation": platform.python_implementation(),
            "python_compiler": platform.python_compiler(),
            "environment_variables_count": len(os.environ),
            "path_entries": len(os.environ.get("PATH", "").split(os.pathsep)),
            "current_working_directory": os.getcwd(),
            "process_count": len(psutil.pids())
        }
    
    async def _get_performance_info(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        # Get CPU usage over a short period
        cpu_percent = psutil.cpu_percent(interval=1.0)
        
        return {
            "cpu_usage_percent": cpu_percent,
            "cpu_per_core": psutil.cpu_percent(percpu=True),
            "memory_usage_percent": psutil.virtual_memory().percent,
            "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None,
            "process_count": len(psutil.pids()),
            "thread_count": sum(p.num_threads() for p in psutil.process_iter(['num_threads']) if p.info['num_threads'])
        }
    
    async def _get_network_info(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Get network information"""
        network_stats = psutil.net_io_counters()
        
        result = {
            "bytes_sent": network_stats.bytes_sent,
            "bytes_recv": network_stats.bytes_recv,
            "packets_sent": network_stats.packets_sent,
            "packets_recv": network_stats.packets_recv,
            "connections_count": len(psutil.net_connections())
        }
        
        if include_sensitive:
            # Only include IP addresses if explicitly requested
            try:
                import socket
                result["hostname"] = socket.gethostname()
                result["local_ip"] = socket.gethostbyname(socket.gethostname())
            except:
                result["hostname"] = "unknown"
                result["local_ip"] = "unknown"
        
        return result
    
    async def _get_storage_info(self) -> Dict[str, Any]:
        """Get storage/disk information"""
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        return {
            "disk_usage": {
                "total_gb": round(disk_usage.total / (1024**3), 2),
                "used_gb": round(disk_usage.used / (1024**3), 2),
                "free_gb": round(disk_usage.free / (1024**3), 2),
                "used_percent": round((disk_usage.used / disk_usage.total) * 100, 2)
            },
            "disk_io": {
                "read_bytes": disk_io.read_bytes if disk_io else 0,
                "write_bytes": disk_io.write_bytes if disk_io else 0,
                "read_count": disk_io.read_count if disk_io else 0,
                "write_count": disk_io.write_count if disk_io else 0
            } if disk_io else None
        }
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate system info parameters"""
        if "info_category" in parameters:
            category = parameters["info_category"]
            valid_categories = ["overview", "hardware", "software", "performance", "network", "storage", "all"]
            if category not in valid_categories:
                return False, f"Invalid info_category. Must be one of: {valid_categories}"
        
        if "include_sensitive" in parameters:
            if not isinstance(parameters["include_sensitive"], bool):
                return False, "Parameter 'include_sensitive' must be a boolean"
        
        return True, None 


class FileAccessTool(BaseTool):
    """Secure file access tool with permission controls"""
    
    def __init__(self):
        metadata = ToolMetadata(
            name="file_access",
            description="Secure file operations including read, write, list, and metadata access with permission controls",
            version="1.0.0",
            tool_type=ToolType.FILE,
            security_level=ToolSecurityLevel.ELEVATED,
            execution_mode=ToolExecutionMode.SYNCHRONOUS,
            parameters=[
                ToolParameter(
                    name="operation",
                    type="string",
                    description="File operation to perform",
                    required=True,
                    allowed_values=["read", "write", "append", "list", "info", "exists", "create_dir", "delete"]
                ),
                ToolParameter(
                    name="path",
                    type="string",
                    description="File or directory path",
                    required=True
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="Content to write (for write/append operations)",
                    required=False
                ),
                ToolParameter(
                    name="encoding",
                    type="string",
                    description="Text encoding for file operations",
                    required=False,
                    default="utf-8",
                    allowed_values=["utf-8", "ascii", "latin-1"]
                ),
                ToolParameter(
                    name="max_size_mb",
                    type="integer",
                    description="Maximum file size to read in MB",
                    required=False,
                    default=10,
                    allowed_values=list(range(1, 101))  # 1-100 MB
                ),
                ToolParameter(
                    name="safe_mode",
                    type="boolean",
                    description="Enable additional safety checks",
                    required=False,
                    default=True
                )
            ],
            return_type="object",
            examples=[
                {
                    "input": {"operation": "read", "path": "/tmp/test.txt"},
                    "output": {
                        "success": True,
                        "content": "Hello World",
                        "size": 11,
                        "encoding": "utf-8"
                    }
                },
                {
                    "input": {"operation": "list", "path": "/tmp"},
                    "output": {
                        "success": True,
                        "files": ["test.txt", "data.json"],
                        "directories": ["subfolder"],
                        "total_items": 3
                    }
                }
            ],
            tags=["file", "io", "storage", "security"],
            max_execution_time=30,
            requires_consent=True,
            data_handling="Accesses local file system. Respects permission boundaries and safety limits."
        )
        super().__init__(metadata)
        
        # Define safe directory boundaries
        self.safe_directories = [
            str(Path.home() / "Documents"),
            str(Path.home() / "Downloads"),
            str(Path.home() / "Desktop"),
            "/tmp",
            "./data",
            "./workspace"
        ]
        
        # Define forbidden paths
        self.forbidden_paths = [
            "/etc/passwd", "/etc/shadow", "/etc/hosts",
            "C:\\Windows\\System32", "C:\\Program Files",
            "/System", "/usr/bin", "/sbin"
        ]
    
    async def execute(self, parameters: Dict[str, Any]) -> Any:
        """Execute file operation"""
        operation = parameters["operation"]
        path = Path(parameters["path"]).resolve()
        safe_mode = parameters.get("safe_mode", True)
        
        try:
            # Security checks
            if safe_mode:
                if not self._is_path_safe(str(path)):
                    raise ToolExecutionError(f"Access denied: Path is outside safe boundaries")
                
                if self._is_path_forbidden(str(path)):
                    raise ToolExecutionError(f"Access denied: Path is forbidden")
            
            if operation == "read":
                result = await self._read_file(path, parameters)
            elif operation == "write":
                result = await self._write_file(path, parameters)
            elif operation == "append":
                result = await self._append_file(path, parameters)
            elif operation == "list":
                result = await self._list_directory(path, parameters)
            elif operation == "info":
                result = await self._get_file_info(path)
            elif operation == "exists":
                result = await self._check_exists(path)
            elif operation == "create_dir":
                result = await self._create_directory(path, parameters)
            elif operation == "delete":
                result = await self._delete_file(path, parameters)
            else:
                raise ToolExecutionError(f"Unknown operation: {operation}")
            
            result["operation"] = operation
            result["path"] = str(path)
            result["timestamp"] = datetime.now().isoformat()
            
            self.logger.info(f"File operation completed: {operation} on {path}")
            return result
            
        except Exception as e:
            raise ToolExecutionError(f"File operation failed: {e}")
    
    async def _read_file(self, path: Path, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Read file content"""
        encoding = parameters.get("encoding", "utf-8")
        max_size_mb = parameters.get("max_size_mb", 10)
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if not path.exists():
            return {"success": False, "error": "File not found"}
        
        if path.is_dir():
            return {"success": False, "error": "Path is a directory, not a file"}
        
        file_size = path.stat().st_size
        if file_size > max_size_bytes:
            return {
                "success": False, 
                "error": f"File too large ({file_size} bytes > {max_size_bytes} bytes limit)"
            }
        
        try:
            content = path.read_text(encoding=encoding)
            return {
                "success": True,
                "content": content,
                "size": file_size,
                "encoding": encoding,
                "lines": len(content.splitlines())
            }
        except UnicodeDecodeError:
            return {"success": False, "error": f"Unable to decode file with {encoding} encoding"}
    
    async def _write_file(self, path: Path, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Write content to file"""
        content = parameters.get("content", "")
        encoding = parameters.get("encoding", "utf-8")
        
        try:
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            path.write_text(content, encoding=encoding)
            
            return {
                "success": True,
                "bytes_written": len(content.encode(encoding)),
                "encoding": encoding
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _append_file(self, path: Path, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Append content to file"""
        content = parameters.get("content", "")
        encoding = parameters.get("encoding", "utf-8")
        
        try:
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, "a", encoding=encoding) as f:
                f.write(content)
            
            return {
                "success": True,
                "bytes_appended": len(content.encode(encoding)),
                "encoding": encoding
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _list_directory(self, path: Path, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """List directory contents"""
        if not path.exists():
            return {"success": False, "error": "Directory not found"}
        
        if not path.is_dir():
            return {"success": False, "error": "Path is not a directory"}
        
        try:
            files = []
            directories = []
            
            for item in path.iterdir():
                if item.is_file():
                    files.append({
                        "name": item.name,
                        "size": item.stat().st_size,
                        "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    })
                elif item.is_dir():
                    directories.append({
                        "name": item.name,
                        "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    })
            
            return {
                "success": True,
                "files": files,
                "directories": directories,
                "total_files": len(files),
                "total_directories": len(directories),
                "total_items": len(files) + len(directories)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _get_file_info(self, path: Path) -> Dict[str, Any]:
        """Get detailed file information"""
        if not path.exists():
            return {"success": False, "error": "File/directory not found"}
        
        try:
            stat = path.stat()
            
            return {
                "success": True,
                "name": path.name,
                "size": stat.st_size,
                "is_file": path.is_file(),
                "is_directory": path.is_dir(),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
                "permissions": oct(stat.st_mode)[-3:],
                "absolute_path": str(path.absolute())
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _check_exists(self, path: Path) -> Dict[str, Any]:
        """Check if file/directory exists"""
        return {
            "success": True,
            "exists": path.exists(),
            "is_file": path.is_file() if path.exists() else None,
            "is_directory": path.is_dir() if path.exists() else None
        }
    
    async def _create_directory(self, path: Path, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create directory"""
        try:
            path.mkdir(parents=True, exist_ok=True)
            return {"success": True, "created": str(path)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _delete_file(self, path: Path, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Delete file or directory (with extreme caution)"""
        if not path.exists():
            return {"success": False, "error": "File/directory not found"}
        
        try:
            if path.is_file():
                path.unlink()
                return {"success": True, "deleted_file": str(path)}
            elif path.is_dir():
                # Only delete empty directories for safety
                if any(path.iterdir()):
                    return {"success": False, "error": "Directory not empty (safety restriction)"}
                path.rmdir()
                return {"success": True, "deleted_directory": str(path)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _is_path_safe(self, path_str: str) -> bool:
        """Check if path is within safe boundaries"""
        path = Path(path_str).resolve()
        
        # Check if path is within any safe directory
        for safe_dir in self.safe_directories:
            try:
                if str(path).startswith(str(Path(safe_dir).resolve())):
                    return True
            except:
                continue
        
        # Allow current working directory and subdirectories
        try:
            if str(path).startswith(str(Path.cwd().resolve())):
                return True
        except:
            pass
        
        return False
    
    def _is_path_forbidden(self, path_str: str) -> bool:
        """Check if path is explicitly forbidden"""
        path_lower = path_str.lower()
        
        for forbidden in self.forbidden_paths:
            if forbidden.lower() in path_lower:
                return True
        
        return False
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate file access parameters"""
        if "operation" not in parameters:
            return False, "Missing required parameter: operation"
        
        if "path" not in parameters:
            return False, "Missing required parameter: path"
        
        operation = parameters["operation"]
        valid_operations = ["read", "write", "append", "list", "info", "exists", "create_dir", "delete"]
        if operation not in valid_operations:
            return False, f"Invalid operation. Must be one of: {valid_operations}"
        
        path = parameters["path"]
        if not isinstance(path, str) or not path.strip():
            return False, "Parameter 'path' must be a non-empty string"
        
        # Content is required for write/append operations
        if operation in ["write", "append"] and "content" not in parameters:
            return False, f"Parameter 'content' is required for {operation} operation"
        
        return True, None 


class AdvancedCalculationTool(BaseTool):
    """Advanced mathematical calculation tool with support for complex operations"""
    
    def __init__(self):
        metadata = ToolMetadata(
            name="advanced_calculator",
            description="Perform advanced mathematical calculations including algebra, statistics, trigonometry, and data analysis",
            version="1.0.0",
            tool_type=ToolType.PYTHON,
            security_level=ToolSecurityLevel.SAFE,
            execution_mode=ToolExecutionMode.SYNCHRONOUS,
            parameters=[
                ToolParameter(
                    name="operation",
                    type="string",
                    description="Mathematical operation to perform",
                    required=True,
                    allowed_values=[
                        "basic", "trigonometry", "logarithm", "statistics", 
                        "algebra", "expression", "matrix", "sequence"
                    ]
                ),
                ToolParameter(
                    name="expression",
                    type="string",
                    description="Mathematical expression to evaluate (for expression operation)",
                    required=False
                ),
                ToolParameter(
                    name="numbers",
                    type="array",
                    description="Array of numbers for calculations",
                    required=False
                ),
                ToolParameter(
                    name="function",
                    type="string",
                    description="Specific function to apply",
                    required=False,
                    allowed_values=[
                        "sin", "cos", "tan", "asin", "acos", "atan",
                        "log", "log10", "ln", "exp", "sqrt", "pow",
                        "mean", "median", "mode", "std", "var", "sum", "product",
                        "factorial", "gcd", "lcm", "prime_check"
                    ]
                ),
                ToolParameter(
                    name="precision",
                    type="integer",
                    description="Number of decimal places for results",
                    required=False,
                    default=6,
                    allowed_values=list(range(1, 16))  # 1-15 decimal places
                ),
                ToolParameter(
                    name="angle_unit",
                    type="string",
                    description="Unit for angle calculations",
                    required=False,
                    default="radians",
                    allowed_values=["radians", "degrees"]
                )
            ],
            return_type="object",
            examples=[
                {
                    "input": {"operation": "basic", "function": "sqrt", "numbers": [16]},
                    "output": {"result": 4.0, "operation": "sqrt", "input": [16]}
                },
                {
                    "input": {"operation": "statistics", "function": "mean", "numbers": [1, 2, 3, 4, 5]},
                    "output": {"result": 3.0, "operation": "mean", "input": [1, 2, 3, 4, 5]}
                },
                {
                    "input": {"operation": "expression", "expression": "2 * pi * 5"},
                    "output": {"result": 31.415927, "expression": "2 * pi * 5"}
                }
            ],
            tags=["math", "calculation", "statistics", "algebra", "science"],
            max_execution_time=10
        )
        super().__init__(metadata)
    
    async def execute(self, parameters: Dict[str, Any]) -> Any:
        """Execute advanced mathematical calculation"""
        operation = parameters["operation"]
        precision = parameters.get("precision", 6)
        
        try:
            if operation == "basic":
                result = await self._basic_operations(parameters)
            elif operation == "trigonometry":
                result = await self._trigonometry_operations(parameters)
            elif operation == "logarithm":
                result = await self._logarithm_operations(parameters)
            elif operation == "statistics":
                result = await self._statistics_operations(parameters)
            elif operation == "algebra":
                result = await self._algebra_operations(parameters)
            elif operation == "expression":
                result = await self._expression_evaluation(parameters)
            elif operation == "matrix":
                result = await self._matrix_operations(parameters)
            elif operation == "sequence":
                result = await self._sequence_operations(parameters)
            else:
                raise ToolExecutionError(f"Unknown operation: {operation}")
            
            # Apply precision formatting
            if isinstance(result.get("result"), (int, float)):
                result["result"] = round(result["result"], precision)
            elif isinstance(result.get("result"), list):
                result["result"] = [round(x, precision) if isinstance(x, (int, float)) else x 
                                 for x in result["result"]]
            
            result["operation_type"] = operation
            result["precision"] = precision
            result["timestamp"] = datetime.now().isoformat()
            
            self.logger.info(f"Advanced calculation completed: {operation}")
            return result
            
        except Exception as e:
            raise ToolExecutionError(f"Calculation failed: {e}")
    
    async def _basic_operations(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform basic mathematical operations"""
        function = parameters.get("function")
        numbers = parameters.get("numbers", [])
        
        if not numbers:
            raise ToolExecutionError("Numbers array is required for basic operations")
        
        if function == "sqrt":
            if len(numbers) != 1:
                raise ToolExecutionError("Square root requires exactly one number")
            result = math.sqrt(numbers[0])
        elif function == "pow":
            if len(numbers) != 2:
                raise ToolExecutionError("Power function requires exactly two numbers (base, exponent)")
            result = math.pow(numbers[0], numbers[1])
        elif function == "factorial":
            if len(numbers) != 1 or not isinstance(numbers[0], int) or numbers[0] < 0:
                raise ToolExecutionError("Factorial requires exactly one non-negative integer")
            result = math.factorial(numbers[0])
        elif function == "gcd":
            if len(numbers) < 2:
                raise ToolExecutionError("GCD requires at least two numbers")
            result = numbers[0]
            for num in numbers[1:]:
                result = math.gcd(int(result), int(num))
        elif function == "sum":
            result = sum(numbers)
        elif function == "product":
            result = 1
            for num in numbers:
                result *= num
        else:
            raise ToolExecutionError(f"Unknown basic function: {function}")
        
        return {"result": result, "function": function, "input": numbers}
    
    async def _trigonometry_operations(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform trigonometric operations"""
        function = parameters.get("function")
        numbers = parameters.get("numbers", [])
        angle_unit = parameters.get("angle_unit", "radians")
        
        if not numbers or len(numbers) != 1:
            raise ToolExecutionError("Trigonometric functions require exactly one number")
        
        angle = numbers[0]
        
        # Convert degrees to radians if necessary
        if angle_unit == "degrees":
            angle = math.radians(angle)
        
        if function == "sin":
            result = math.sin(angle)
        elif function == "cos":
            result = math.cos(angle)
        elif function == "tan":
            result = math.tan(angle)
        elif function == "asin":
            result = math.asin(angle)
            if angle_unit == "degrees":
                result = math.degrees(result)
        elif function == "acos":
            result = math.acos(angle)
            if angle_unit == "degrees":
                result = math.degrees(result)
        elif function == "atan":
            result = math.atan(angle)
            if angle_unit == "degrees":
                result = math.degrees(result)
        else:
            raise ToolExecutionError(f"Unknown trigonometric function: {function}")
        
        return {
            "result": result, 
            "function": function, 
            "input": numbers[0], 
            "angle_unit": angle_unit
        }
    
    async def _logarithm_operations(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform logarithmic operations"""
        function = parameters.get("function")
        numbers = parameters.get("numbers", [])
        
        if not numbers or len(numbers) < 1:
            raise ToolExecutionError("Logarithmic functions require at least one number")
        
        if function == "log":
            if len(numbers) == 1:
                result = math.log(numbers[0])  # Natural log
            elif len(numbers) == 2:
                result = math.log(numbers[0], numbers[1])  # Log with custom base
            else:
                raise ToolExecutionError("Log function requires 1 or 2 numbers")
        elif function == "log10":
            if len(numbers) != 1:
                raise ToolExecutionError("Log10 requires exactly one number")
            result = math.log10(numbers[0])
        elif function == "ln":
            if len(numbers) != 1:
                raise ToolExecutionError("Natural log requires exactly one number")
            result = math.log(numbers[0])
        elif function == "exp":
            if len(numbers) != 1:
                raise ToolExecutionError("Exponential requires exactly one number")
            result = math.exp(numbers[0])
        else:
            raise ToolExecutionError(f"Unknown logarithmic function: {function}")
        
        return {"result": result, "function": function, "input": numbers}
    
    async def _statistics_operations(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical operations"""
        function = parameters.get("function")
        numbers = parameters.get("numbers", [])
        
        if not numbers:
            raise ToolExecutionError("Numbers array is required for statistical operations")
        
        if function == "mean":
            result = statistics.mean(numbers)
        elif function == "median":
            result = statistics.median(numbers)
        elif function == "mode":
            try:
                result = statistics.mode(numbers)
            except statistics.StatisticsError:
                # If no unique mode, return all modes
                result = list(statistics.multimode(numbers))
        elif function == "std":
            if len(numbers) < 2:
                raise ToolExecutionError("Standard deviation requires at least 2 numbers")
            result = statistics.stdev(numbers)
        elif function == "var":
            if len(numbers) < 2:
                raise ToolExecutionError("Variance requires at least 2 numbers")
            result = statistics.variance(numbers)
        else:
            raise ToolExecutionError(f"Unknown statistical function: {function}")
        
        return {
            "result": result, 
            "function": function, 
            "input": numbers,
            "sample_size": len(numbers)
        }
    
    async def _algebra_operations(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform algebraic operations"""
        # Simplified algebra operations
        function = parameters.get("function")
        numbers = parameters.get("numbers", [])
        
        if function == "quadratic":
            # Solve quadratic equation ax^2 + bx + c = 0
            if len(numbers) != 3:
                raise ToolExecutionError("Quadratic equation requires 3 coefficients [a, b, c]")
            
            a, b, c = numbers
            if a == 0:
                raise ToolExecutionError("Coefficient 'a' cannot be zero for quadratic equation")
            
            discriminant = b*b - 4*a*c
            
            if discriminant >= 0:
                sqrt_discriminant = math.sqrt(discriminant)
                x1 = (-b + sqrt_discriminant) / (2*a)
                x2 = (-b - sqrt_discriminant) / (2*a)
                result = [x1, x2]
            else:
                # Complex roots
                real_part = -b / (2*a)
                imaginary_part = math.sqrt(-discriminant) / (2*a)
                result = [
                    f"{real_part} + {imaginary_part}i",
                    f"{real_part} - {imaginary_part}i"
                ]
        else:
            raise ToolExecutionError(f"Unknown algebraic function: {function}")
        
        return {"result": result, "function": function, "coefficients": numbers}
    
    async def _expression_evaluation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate mathematical expressions safely"""
        expression = parameters.get("expression", "")
        
        if not expression:
            raise ToolExecutionError("Expression is required for expression evaluation")
        
        # Define safe mathematical functions and constants
        safe_dict = {
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "asin": math.asin, "acos": math.acos, "atan": math.atan,
            "log": math.log, "log10": math.log10, "exp": math.exp,
            "sqrt": math.sqrt, "pow": math.pow, "abs": abs,
            "pi": math.pi, "e": math.e, "tau": math.tau,
            "ceil": math.ceil, "floor": math.floor, "round": round,
            "max": max, "min": min, "sum": sum
        }
        
        # Remove potentially dangerous operations but keep safe characters
        expression = re.sub(r'[^0-9+\-*/().,\s\w]', '', expression)
        
        try:
            # Evaluate expression in safe environment
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            
            return {
                "result": result,
                "expression": expression,
                "type": type(result).__name__
            }
        except Exception as e:
            raise ToolExecutionError(f"Expression evaluation failed: {e}")
    
    async def _matrix_operations(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Basic matrix operations"""
        # Simplified matrix operations without external libraries
        function = parameters.get("function", "determinant")
        numbers = parameters.get("numbers", [])
        
        if function == "determinant":
            # Only support 2x2 matrices for simplicity
            if len(numbers) != 4:
                raise ToolExecutionError("Determinant calculation requires 4 numbers for 2x2 matrix [a,b,c,d]")
            
            a, b, c, d = numbers
            result = a*d - b*c
            
            return {
                "result": result,
                "function": function,
                "matrix": [[a, b], [c, d]]
            }
        else:
            raise ToolExecutionError(f"Unknown matrix function: {function}")
    
    async def _sequence_operations(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Sequence and series operations"""
        function = parameters.get("function", "fibonacci")
        numbers = parameters.get("numbers", [])
        
        if function == "fibonacci":
            if not numbers or len(numbers) != 1:
                raise ToolExecutionError("Fibonacci requires exactly one number (n)")
            
            n = int(numbers[0])
            if n < 0:
                raise ToolExecutionError("Fibonacci sequence requires non-negative integer")
            
            if n == 0:
                result = []
            elif n == 1:
                result = [0]
            else:
                result = [0, 1]
                for i in range(2, n):
                    result.append(result[i-1] + result[i-2])
            
            return {
                "result": result,
                "function": function,
                "length": n
            }
        else:
            raise ToolExecutionError(f"Unknown sequence function: {function}")
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate calculation parameters"""
        if "operation" not in parameters:
            return False, "Missing required parameter: operation"
        
        operation = parameters["operation"]
        valid_operations = ["basic", "trigonometry", "logarithm", "statistics", "algebra", "expression", "matrix", "sequence"]
        if operation not in valid_operations:
            return False, f"Invalid operation. Must be one of: {valid_operations}"
        
        # Expression operation requires expression parameter
        if operation == "expression" and "expression" not in parameters:
            return False, "Expression operation requires 'expression' parameter"
        
        # Most other operations require numbers or function
        if operation != "expression":
            if "function" not in parameters and "numbers" not in parameters:
                return False, f"Operation '{operation}' requires either 'function' or 'numbers' parameter"
        
        return True, None


# Factory function for core tools
def create_core_tools() -> List[BaseTool]:
    """Create and return the core set of tools"""
    return [
        InternetSearchTool(),
        EnhancedSystemInfoTool(),
        FileAccessTool(),
        AdvancedCalculationTool()
    ]


# Tool registration helper
async def register_core_tools(framework) -> int:
    """Register core tools with the framework"""
    tools = create_core_tools()
    registered_count = 0
    
    for tool in tools:
        if framework.registry.register_tool(tool):
            registered_count += 1
    
    return registered_count 