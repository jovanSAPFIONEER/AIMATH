"""
LLM Providers - Unified interface for multiple LLM backends.

Supported providers:
- OpenAI (GPT-4, GPT-4-turbo, etc.)
- Anthropic (Claude 3, etc.)
- Google (Gemini)
- Ollama (local models)

Users provide their own API keys via environment variables or config.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    model: str
    tokens_used: int
    finish_reason: str
    raw_response: Optional[Dict] = None


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All providers implement this interface for consistent usage.
    """
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Optional[str]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            max_tokens: Maximum tokens in response
            temperature: Creativity parameter
            
        Returns:
            Generated text or None on error
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is properly configured."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass


class OpenAIProvider(LLMProvider):
    """
    OpenAI API provider (GPT-4, etc.).
    
    Requires OPENAI_API_KEY environment variable or explicit key.
    
    Example:
        >>> provider = OpenAIProvider()  # Uses env var
        >>> provider = OpenAIProvider(api_key="sk-...")  # Explicit key
        >>> 
        >>> response = provider.generate("What is 2+2?")
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-4o-mini",
        base_url: str = None,
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (or OPENAI_API_KEY env var)
            model: Model to use
            base_url: Custom API base URL (for Azure, etc.)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.base_url = base_url
        self._client = None
    
    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                
                kwargs = {'api_key': self.api_key}
                if self.base_url:
                    kwargs['base_url'] = self.base_url
                
                self._client = OpenAI(**kwargs)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        
        return self._client
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Optional[str]:
        """Generate using OpenAI."""
        if not self.api_key:
            return None
        
        try:
            client = self._get_client()
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"[OpenAIProvider] Error: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if OpenAI is configured."""
        return bool(self.api_key)
    
    @property
    def name(self) -> str:
        return f"OpenAI ({self.model})"


class AnthropicProvider(LLMProvider):
    """
    Anthropic API provider (Claude 3, etc.).
    
    Requires ANTHROPIC_API_KEY environment variable or explicit key.
    
    Example:
        >>> provider = AnthropicProvider()
        >>> response = provider.generate("Explain calculus")
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key (or ANTHROPIC_API_KEY env var)
            model: Model to use
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.model = model
        self._client = None
    
    def _get_client(self):
        """Get or create Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        return self._client
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Optional[str]:
        """Generate using Anthropic."""
        if not self.api_key:
            return None
        
        try:
            client = self._get_client()
            
            kwargs = {
                'model': self.model,
                'max_tokens': max_tokens,
                'messages': [{"role": "user", "content": prompt}],
            }
            
            if system_prompt:
                kwargs['system'] = system_prompt
            
            response = client.messages.create(**kwargs)
            
            return response.content[0].text
            
        except Exception as e:
            print(f"[AnthropicProvider] Error: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if Anthropic is configured."""
        return bool(self.api_key)
    
    @property
    def name(self) -> str:
        return f"Anthropic ({self.model})"


class GeminiProvider(LLMProvider):
    """
    Google Gemini API provider.
    
    Requires GOOGLE_API_KEY environment variable or explicit key.
    
    Example:
        >>> provider = GeminiProvider()
        >>> response = provider.generate("What is pi?")
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "gemini-1.5-flash",
    ):
        """
        Initialize Gemini provider.
        
        Args:
            api_key: Google API key (or GOOGLE_API_KEY env var)
            model: Model to use
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.model = model
        self._model_instance = None
    
    def _get_model(self):
        """Get or create Gemini model."""
        if self._model_instance is None:
            try:
                import google.generativeai as genai
                
                genai.configure(api_key=self.api_key)
                self._model_instance = genai.GenerativeModel(self.model)
            except ImportError:
                raise ImportError(
                    "google-generativeai package not installed. "
                    "Run: pip install google-generativeai"
                )
        
        return self._model_instance
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Optional[str]:
        """Generate using Gemini."""
        if not self.api_key:
            return None
        
        try:
            model = self._get_model()
            
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            response = model.generate_content(
                full_prompt,
                generation_config={
                    'max_output_tokens': max_tokens,
                    'temperature': temperature,
                }
            )
            
            return response.text
            
        except Exception as e:
            print(f"[GeminiProvider] Error: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if Gemini is configured."""
        return bool(self.api_key)
    
    @property
    def name(self) -> str:
        return f"Gemini ({self.model})"


class OllamaProvider(LLMProvider):
    """
    Ollama provider for local models.
    
    No API key required - runs locally.
    
    Example:
        >>> provider = OllamaProvider(model="llama3.1")
        >>> response = provider.generate("Solve x^2 = 4")
    """
    
    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize Ollama provider.
        
        Args:
            model: Model to use (must be pulled locally)
            base_url: Ollama API URL
        """
        self.model = model
        self.base_url = base_url
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Optional[str]:
        """Generate using Ollama."""
        try:
            import requests
            
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    }
                },
                timeout=120,
            )
            
            if response.status_code == 200:
                return response.json().get('response')
            else:
                print(f"[OllamaProvider] Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"[OllamaProvider] Error: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    @property
    def name(self) -> str:
        return f"Ollama ({self.model})"


def get_provider(
    provider_type: str = None,
    api_key: str = None,
    model: str = None,
) -> Optional[LLMProvider]:
    """
    Factory function to get an LLM provider.
    
    Automatically selects based on available API keys if provider_type not specified.
    
    Args:
        provider_type: 'openai', 'anthropic', 'gemini', 'ollama', or None for auto
        api_key: API key (optional, uses env vars)
        model: Model override (optional)
        
    Returns:
        LLMProvider instance or None
    """
    providers = {
        'openai': lambda: OpenAIProvider(api_key=api_key, model=model or "gpt-4o-mini"),
        'anthropic': lambda: AnthropicProvider(api_key=api_key, model=model or "claude-sonnet-4-20250514"),
        'gemini': lambda: GeminiProvider(api_key=api_key, model=model or "gemini-1.5-flash"),
        'ollama': lambda: OllamaProvider(model=model or "llama3.1"),
    }
    
    if provider_type:
        if provider_type in providers:
            return providers[provider_type]()
        else:
            raise ValueError(f"Unknown provider: {provider_type}")
    
    # Auto-select based on available keys
    for name, factory in providers.items():
        provider = factory()
        if provider.is_available():
            return provider
    
    return None
