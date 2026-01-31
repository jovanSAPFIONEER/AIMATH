"""
Configuration Management - Load and manage MathClaw settings.

Configuration sources (in priority order):
1. Constructor arguments
2. Environment variables
3. .env file
4. Config file (mathclaw.yaml)
5. Defaults
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class MathClawConfig:
    """
    MathClaw configuration.
    
    Example:
        >>> config = MathClawConfig.load()
        >>> print(config.llm_provider)
        >>> print(config.max_attempts_per_hour)
    """
    
    # LLM Configuration
    llm_provider: str = "auto"  # openai, anthropic, gemini, ollama, auto
    llm_model: Optional[str] = None  # Override model
    
    # API Keys (can also use env vars)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    
    # Discovery Settings
    min_interval_seconds: int = 2
    max_attempts_per_hour: int = 100
    verification_timeout: int = 30
    batch_size: int = 1
    
    # Exploration Parameters
    exploration_rate: float = 0.15  # Epsilon for exploration
    
    # Storage
    db_path: Optional[str] = None
    export_dir: Optional[str] = None
    
    # Safety Settings
    daily_token_budget: int = 100000
    daily_cost_budget: float = 5.0
    
    # Advanced
    debug: bool = False
    log_level: str = "INFO"
    
    @classmethod
    def load(
        cls,
        config_path: Path = None,
        env_file: Path = None,
    ) -> 'MathClawConfig':
        """
        Load configuration from multiple sources.
        
        Args:
            config_path: Path to YAML config file
            env_file: Path to .env file
            
        Returns:
            MathClawConfig instance
        """
        # Load .env file if exists
        if env_file is None:
            env_file = Path.cwd() / '.env'
        
        if env_file.exists():
            cls._load_dotenv(env_file)
        
        # Load YAML config if exists
        yaml_config = {}
        if config_path is None:
            config_path = Path.cwd() / 'mathclaw.yaml'
        
        if config_path.exists():
            yaml_config = cls._load_yaml(config_path)
        
        # Build config from all sources
        return cls(
            # LLM
            llm_provider=cls._get_value('llm_provider', yaml_config, 'MATHCLAW_LLM_PROVIDER', 'auto'),
            llm_model=cls._get_value('llm_model', yaml_config, 'MATHCLAW_LLM_MODEL', None),
            
            # API Keys
            openai_api_key=cls._get_value('openai_api_key', yaml_config, 'OPENAI_API_KEY', None),
            anthropic_api_key=cls._get_value('anthropic_api_key', yaml_config, 'ANTHROPIC_API_KEY', None),
            google_api_key=cls._get_value('google_api_key', yaml_config, 'GOOGLE_API_KEY', None),
            
            # Discovery
            min_interval_seconds=int(cls._get_value('min_interval_seconds', yaml_config, 'MATHCLAW_MIN_INTERVAL', 2)),
            max_attempts_per_hour=int(cls._get_value('max_attempts_per_hour', yaml_config, 'MATHCLAW_MAX_ATTEMPTS', 100)),
            verification_timeout=int(cls._get_value('verification_timeout', yaml_config, 'MATHCLAW_VERIFICATION_TIMEOUT', 30)),
            batch_size=int(cls._get_value('batch_size', yaml_config, 'MATHCLAW_BATCH_SIZE', 1)),
            
            # Exploration
            exploration_rate=float(cls._get_value('exploration_rate', yaml_config, 'MATHCLAW_EXPLORATION_RATE', 0.15)),
            
            # Storage
            db_path=cls._get_value('db_path', yaml_config, 'MATHCLAW_DB_PATH', None),
            export_dir=cls._get_value('export_dir', yaml_config, 'MATHCLAW_EXPORT_DIR', None),
            
            # Safety
            daily_token_budget=int(cls._get_value('daily_token_budget', yaml_config, 'MATHCLAW_TOKEN_BUDGET', 100000)),
            daily_cost_budget=float(cls._get_value('daily_cost_budget', yaml_config, 'MATHCLAW_COST_BUDGET', 5.0)),
            
            # Advanced
            debug=cls._get_value('debug', yaml_config, 'MATHCLAW_DEBUG', 'false').lower() == 'true' if isinstance(cls._get_value('debug', yaml_config, 'MATHCLAW_DEBUG', False), str) else cls._get_value('debug', yaml_config, 'MATHCLAW_DEBUG', False),
            log_level=cls._get_value('log_level', yaml_config, 'MATHCLAW_LOG_LEVEL', 'INFO'),
        )
    
    @staticmethod
    def _get_value(
        key: str,
        yaml_config: Dict,
        env_var: str,
        default: Any,
    ) -> Any:
        """Get value from YAML config, env var, or default."""
        # YAML config takes priority
        if key in yaml_config:
            return yaml_config[key]
        
        # Then environment variable
        env_value = os.getenv(env_var)
        if env_value is not None:
            return env_value
        
        # Then default
        return default
    
    @staticmethod
    def _load_dotenv(env_file: Path) -> None:
        """Load .env file into environment."""
        try:
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value
        except Exception:
            pass
    
    @staticmethod
    def _load_yaml(config_path: Path) -> Dict:
        """Load YAML config file."""
        try:
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            # Fallback: basic YAML parsing for simple cases
            config = {}
            try:
                with open(config_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and ':' in line:
                            key, value = line.split(':', 1)
                            config[key.strip()] = value.strip()
            except:
                pass
            return config
        except Exception:
            return {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (hiding API keys)."""
        return {
            'llm_provider': self.llm_provider,
            'llm_model': self.llm_model,
            'has_openai_key': bool(self.openai_api_key),
            'has_anthropic_key': bool(self.anthropic_api_key),
            'has_google_key': bool(self.google_api_key),
            'min_interval_seconds': self.min_interval_seconds,
            'max_attempts_per_hour': self.max_attempts_per_hour,
            'verification_timeout': self.verification_timeout,
            'batch_size': self.batch_size,
            'exploration_rate': self.exploration_rate,
            'db_path': self.db_path,
            'export_dir': self.export_dir,
            'daily_token_budget': self.daily_token_budget,
            'daily_cost_budget': self.daily_cost_budget,
            'debug': self.debug,
            'log_level': self.log_level,
        }
    
    def save(self, path: Path = None) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save to (default: mathclaw.yaml)
        """
        path = path or Path.cwd() / 'mathclaw.yaml'
        
        content = f"""# MathClaw Configuration
# Generated automatically

# LLM Provider: openai, anthropic, gemini, ollama, or auto
llm_provider: {self.llm_provider}
{"llm_model: " + self.llm_model if self.llm_model else "# llm_model: gpt-4o-mini"}

# Discovery Settings
min_interval_seconds: {self.min_interval_seconds}
max_attempts_per_hour: {self.max_attempts_per_hour}
verification_timeout: {self.verification_timeout}

# Exploration
exploration_rate: {self.exploration_rate}

# Safety Limits
daily_token_budget: {self.daily_token_budget}
daily_cost_budget: {self.daily_cost_budget}

# Debug
debug: {str(self.debug).lower()}
log_level: {self.log_level}

# Note: API keys should be in .env file or environment variables
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-...
# GOOGLE_API_KEY=...
"""
        
        path.write_text(content)
    
    def create_example_env(self, path: Path = None) -> None:
        """
        Create example .env file.
        
        Args:
            path: Path to save to (default: .env.example)
        """
        path = path or Path.cwd() / '.env.example'
        
        content = """# MathClaw Environment Variables
# Copy this file to .env and fill in your API keys

# OpenAI API Key (for GPT-4, etc.)
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-your-key-here

# Anthropic API Key (for Claude)
# Get from: https://console.anthropic.com/
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Google API Key (for Gemini)
# Get from: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=your-key-here

# Optional: Override settings
# MATHCLAW_LLM_PROVIDER=openai
# MATHCLAW_MAX_ATTEMPTS=100
# MATHCLAW_COST_BUDGET=5.0
"""
        
        path.write_text(content)
