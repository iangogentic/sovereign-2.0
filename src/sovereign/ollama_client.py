"""
Ollama client for communicating with local Ollama server
"""

import json
import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional, AsyncGenerator
import time

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Exception raised for Ollama API errors"""
    pass


class OllamaClient:
    """Client for communicating with Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 10.0):
        """
        Initialize Ollama client
        
        Args:
            base_url: Base URL for Ollama server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _ensure_session(self):
        """Ensure we have an active session"""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
    
    async def check_health(self) -> bool:
        """
        Check if Ollama server is running and accessible
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            await self._ensure_session()
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def list_models(self) -> Dict[str, Any]:
        """
        List available models in Ollama
        
        Returns:
            Dictionary containing model information
        """
        try:
            await self._ensure_session()
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise OllamaError(f"Failed to list models: HTTP {response.status}")
        except aiohttp.ClientError as e:
            raise OllamaError(f"Network error listing models: {e}")
    
    async def generate_stream(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate text using streaming response
        
        Args:
            model: Model name (e.g., 'gemma2:9b')
            prompt: User prompt
            system: System prompt (optional)
            temperature: Response randomness (0.0-1.0)
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            
        Yields:
            Generated text chunks
        """
        try:
            await self._ensure_session()
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                }
            }
            
            if system:
                payload["system"] = system
            
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
            
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise OllamaError(f"Generation failed: HTTP {response.status} - {error_text}")
                
                async for line in response.content:
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            
                            if chunk.get('error'):
                                raise OllamaError(f"Ollama error: {chunk['error']}")
                            
                            if 'response' in chunk:
                                yield chunk['response']
                            
                            # Check if generation is complete
                            if chunk.get('done', False):
                                break
                                
                        except json.JSONDecodeError:
                            continue  # Skip malformed JSON lines
                            
        except aiohttp.ClientError as e:
            raise OllamaError(f"Network error during generation: {e}")
    
    async def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text using non-streaming response
        
        Args:
            model: Model name (e.g., 'gemma2:9b')
            prompt: User prompt
            system: System prompt (optional)
            temperature: Response randomness (0.0-1.0)
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Complete generated text
        """
        try:
            await self._ensure_session()
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                }
            }
            
            if system:
                payload["system"] = system
            
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
            
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    if result.get('error'):
                        raise OllamaError(f"Ollama error: {result['error']}")
                    return result.get('response', '')
                else:
                    error_text = await response.text()
                    raise OllamaError(f"Generation failed: HTTP {response.status} - {error_text}")
                    
        except aiohttp.ClientError as e:
            raise OllamaError(f"Network error during generation: {e}")
    
    async def pull_model(self, model: str) -> bool:
        """
        Pull a model from Ollama registry
        
        Args:
            model: Model name to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self._ensure_session()
            
            payload = {"name": model, "stream": False}
            
            async with self.session.post(
                f"{self.base_url}/api/pull",
                json=payload
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    return result.get('status') == 'success'
                else:
                    logger.error(f"Failed to pull model {model}: HTTP {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error pulling model {model}: {e}")
            return False
    
    async def close(self):
        """Close the client session"""
        if self.session and not self.session.closed:
            await self.session.close() 