from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.fields import FieldInfo
from typing import Any
from typing import Optional, List, Dict, Literal
from pathlib import Path
from termcolor import colored
import os
import yaml
from contextvars import ContextVar
import argparse
import logging


class ModelConfig(BaseModel):
    """Default models - can be overridden per agent"""
    persuader: str = "gemini-2.0-flash"
    doer: str = "gemini-2.0-flash"
    embedding: str = "models/embedding-001"
    generation: str = "gemini-1.5-flash"

class NetworkConfig(BaseModel):
    doer_port: int = Field(default=8001)
    host: str = Field(default="0.0.0.0")
    agent_card_path: str = Field(
        default="/.well-known/agent-card.json",
        description="Path to agent card endpoint"
    )

class RAGConfig(BaseModel):
    lance_db_path: str
    table_name: str = "general_knowledge"
    embedding_dim: int = 768
    search_limit: int = 3
    min_relevance: float = 0.5
    initial_data: Optional[str] = None

class ToolConfig(BaseModel):
    """Define which tools an agent has access to"""
    search_knowledge_base: bool = True
    take_red_pill: bool = False
    take_blue_pill: bool = False
    join_blue_team: bool = False

class RemoteAgentConfig(BaseModel):
    """Configuration for remote A2A agent connections"""
    name: str = Field(..., description="Name of the remote agent")
    description: str = Field(..., description="Description of remote agent or path to description file")
    agent_card_url: Optional[str] = Field(
        None, 
        description="Full URL to agent card (if None, constructs from network config)"
    )
    
    @field_validator('description')
    @classmethod
    def load_description_from_file(cls, v: str, info) -> str:
        """Load description from file if it's a path"""
        if not v:
            print(colored("⚠️  Remote agent description is empty", "yellow"))
            logging.warning("Remote agent description is empty")
            return ""
        
        potential_path = Path(v)
        if potential_path.exists() and potential_path.is_file():
            content = potential_path.read_text().strip()
            if not content:
                print(colored(f"⚠️  Remote agent description file is empty: {potential_path}", "yellow"))
                logging.warning(f"Remote agent description file is empty: {potential_path}")
                return ""
            return content
        elif potential_path.suffix in ['.txt', '.md']:
            print(colored(f"⚠️  Remote agent description file not found: {potential_path}", "yellow"))
            logging.warning(f"Remote agent description file not found: {potential_path}")
            return v
        
        return v

class SubAgentDefinition(BaseModel):
    """Define a single sub-agent's configuration"""
    name: str = Field(..., description="Unique identifier for this agent")
    display_name: str = Field(..., description="Human-readable name")
    instruction: str = Field(..., description="System prompt or path to prompt file")
    model: Optional[str] = Field(None, description="Model string (e.g., 'gemini-2.0-flash', 'claude-sonnet-4')")
    tools: ToolConfig = Field(default_factory=ToolConfig)
    enabled: bool = Field(default=True, description="Whether to include this agent")
    
    output_key: Optional[str] = Field(None, description="State key to write output to")
    include_contents: str = Field(default='default', description="How to include conversation history")
    description: Optional[str] = Field(None, description="Agent description for documentation")
    
    @field_validator('instruction')
    @classmethod
    def load_instruction_from_file(cls, v: str, info) -> str:
        """Load instruction from file if it's a path"""
        if not v:
            agent_name = info.data.get('name', 'unknown')
            print(colored(f"⚠️  Agent '{agent_name}' has empty instruction", "yellow"))
            logging.warning(f"Agent '{agent_name}' has empty instruction")
            return ""
        
        potential_path = Path(v)
        
        if potential_path.exists() and potential_path.is_file():
            content = potential_path.read_text().strip()
            if not content:
                agent_name = info.data.get('name', 'unknown')
                print(colored(f"⚠️  Agent '{agent_name}' instruction file is empty: {potential_path}", "yellow"))
                logging.warning(f"Agent '{agent_name}' instruction file is empty: {potential_path}")
                return ""
            logging.debug(f"Loaded instruction from file: {potential_path}")
            return content
        elif potential_path.suffix in ['.txt', '.md']:
            agent_name = info.data.get('name', 'unknown')
            print(colored(f"⚠️  Agent '{agent_name}' instruction file not found: {potential_path}", "yellow"))
            logging.warning(f"Agent '{agent_name}' instruction file not found: {potential_path}")
            return v
        
        return v

class LoopAgentConfig(BaseModel):
    """Configuration for LoopAgent pattern (Strategist -> Executor)"""
    enabled: bool = Field(default=False, description="Use LoopAgent pattern")
    
    strategist_instruction: str
    strategist_model: Optional[str] = Field(None, description="Model for strategist")
    strategist_output_key: str = Field(default="current_strategy")
    strategist_include_contents: str = Field(default='default')
    strategist_description: Optional[str] = None
    
    executor_instruction: str
    executor_model: Optional[str] = Field(None, description="Model for executor")
    executor_output_key: str = Field(default="execution_result")
    executor_include_contents: str = Field(default='default')
    executor_description: Optional[str] = None
    
    @field_validator('strategist_instruction', 'executor_instruction')
    @classmethod
    def load_instruction_from_file(cls, v: str, info) -> str:
        """Load instruction from file with proper error handling"""
        if not v:
            field_name = info.field_name
            role = "strategist" if "strategist" in field_name else "executor"
            print(colored(f"⚠️  LoopAgent {role} instruction is empty", "yellow"))
            logging.warning(f"LoopAgent {role} instruction is empty")
            return ""
        
        potential_path = Path(v)
        
        if potential_path.exists() and potential_path.is_file():
            content = potential_path.read_text().strip()
            if not content:
                field_name = info.field_name
                role = "strategist" if "strategist" in field_name else "executor"
                print(colored(f"⚠️  LoopAgent {role} instruction file is empty: {potential_path}", "yellow"))
                logging.warning(f"LoopAgent {role} instruction file is empty: {potential_path}")
                return ""
            logging.info(f"Loaded from file: {potential_path}")
            return content
        elif potential_path.suffix in ['.txt', '.md']:
            field_name = info.field_name
            role = "strategist" if "strategist" in field_name else "executor"
            print(colored(f"⚠️  LoopAgent {role} instruction file not found: {potential_path}", "yellow"))
            logging.warning(f"LoopAgent {role} instruction file not found: {potential_path}")
            return v
        
        return v

class TeamDefinition(BaseModel):
    """Define a team (lead + sub-agents)"""
    lead_name: str
    lead_instruction: str
    lead_model: Optional[str] = Field(None, description="Model for lead agent")
    lead_tools: ToolConfig = Field(default_factory=ToolConfig)
    lead_output_key: Optional[str] = Field(None, description="Output key for lead agent")
    lead_include_contents: str = Field(default='default')
    lead_description: Optional[str] = None
    
    sub_agents: List[SubAgentDefinition] = Field(default_factory=list)
    max_iterations: Optional[int] = Field(None, description="Override global max_iterations")
    
    loop_agent: Optional[LoopAgentConfig] = Field(None, description="Use LoopAgent with strategist/executor")
    
    remote_agent: Optional[RemoteAgentConfig] = Field(
        None, 
        description="Remote A2A agent this team connects to"
    )
    
    @field_validator('lead_instruction')
    @classmethod
    def load_lead_instruction_from_file(cls, v: str, info) -> str:
        """Load lead instruction from file with validation"""
        if not v:
            lead_name = info.data.get('lead_name', 'unknown')
            print(colored(f"⚠️  Team lead '{lead_name}' has empty instruction", "yellow"))
            logging.warning(f"Team lead '{lead_name}' has empty instruction")
            return ""
        
        potential_path = Path(v)
        
        if potential_path.exists() and potential_path.is_file():
            content = potential_path.read_text().strip()
            if not content:
                lead_name = info.data.get('lead_name', 'unknown')
                print(colored(f"⚠️  Team lead '{lead_name}' instruction file is empty: {potential_path}", "yellow"))
                logging.warning(f"Team lead '{lead_name}' instruction file is empty: {potential_path}")
                return ""
            logging.info(f"Loaded team lead instruction from file: {potential_path}")
            return content
        elif potential_path.suffix in ['.txt', '.md']:
            lead_name = info.data.get('lead_name', 'unknown')
            print(colored(f"⚠️  Team lead '{lead_name}' instruction file not found: {potential_path}", "yellow"))
            logging.warning(f"Team lead '{lead_name}' instruction file not found: {potential_path}")
            return v
        
        return v
    
    def get_enabled_sub_agents(self) -> List[SubAgentDefinition]:
        """Return only enabled sub-agents"""
        return [agent for agent in self.sub_agents if agent.enabled]

class SimulationConfig(BaseModel):
    """High-level simulation parameters"""
    max_iterations: int = Field(default=10, ge=1)
    default_start_prompt: str
    session_id: str = Field(default="hil-session")
    app_name: str = Field(default="agents")

class FullConfig(BaseModel):
    models: ModelConfig
    network: NetworkConfig
    rag: RAGConfig
    simulation: SimulationConfig
    
    morphus_team: TeamDefinition
    neo_team: TeamDefinition
    
    google_api_key: str
    
    @field_validator('google_api_key')
    @classmethod
    def load_api_key_from_env(cls, v: str) -> str:
        """Support ${ENV_VAR} syntax"""
        if v.startswith('${') and v.endswith('}'):
            env_var = v[2:-1]
            value = os.getenv(env_var)
            if not value:
                raise ValueError(f"Environment variable {env_var} not set")
            return value
        return v
    
    def get_agent_card_url(self, remote_config: RemoteAgentConfig) -> str:
        """Construct agent card URL from network config if not explicitly set"""
        if remote_config.agent_card_url:
            return remote_config.agent_card_url
        
        return f"http://{self.network.host}:{self.network.doer_port}{self.network.agent_card_path}"
    
    def validate_all_prompts(self) -> List[str]:
        """
        Validate all prompt files and return list of warnings.
        Call this after loading config to get a summary of issues.
        """
        warnings = []
        
        # Check Morphus team
        if self.morphus_team.loop_agent and self.morphus_team.loop_agent.enabled:
            if not self.morphus_team.loop_agent.strategist_instruction:
                warnings.append("Morphus strategist instruction is empty")
            if not self.morphus_team.loop_agent.executor_instruction:
                warnings.append("Morphus executor instruction is empty")
        elif not self.morphus_team.lead_instruction:
            warnings.append(f"Morphus team lead '{self.morphus_team.lead_name}' instruction is empty")
        
        for agent in self.morphus_team.sub_agents:
            if agent.enabled and not agent.instruction:
                warnings.append(f"Morphus sub-agent '{agent.name}' instruction is empty")
        
        # Check Neo team
        if self.neo_team.loop_agent and self.neo_team.loop_agent.enabled:
            if not self.neo_team.loop_agent.strategist_instruction:
                warnings.append("Neo strategist instruction is empty")
            if not self.neo_team.loop_agent.executor_instruction:
                warnings.append("Neo executor instruction is empty")
        elif not self.neo_team.lead_instruction:
            warnings.append(f"Neo team lead '{self.neo_team.lead_name}' instruction is empty")
        
        for agent in self.neo_team.sub_agents:
            if agent.enabled and not agent.instruction:
                warnings.append(f"Neo sub-agent '{agent.name}' instruction is empty")
        
        # Check remote agent descriptions
        if self.morphus_team.remote_agent and not self.morphus_team.remote_agent.description:
            warnings.append("Morphus remote agent description is empty")
        
        if self.neo_team.remote_agent and not self.neo_team.remote_agent.description:
            warnings.append("Neo remote agent description is empty")
        
        return warnings


config_context: ContextVar[FullConfig] = ContextVar('agent_config')

def load_config(yaml_path: str) -> FullConfig:
    """Load and validate configuration from YAML"""
    if not Path(yaml_path).exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        raw_data = yaml.safe_load(f)
    
    config = FullConfig(**raw_data)
    
    # Validate all prompts and log warnings
    warnings = config.validate_all_prompts()
    if warnings:
        logger = logging.getLogger(__name__)
        logger.warning(f"Configuration loaded with {len(warnings)} warning(s):")
        for warning in warnings:
            logger.warning(f"  - {warning}")
    
    return config


def initialize_config() -> FullConfig:
    args = parse_args()
    config = load_config(args.config)
    config_context.set(config)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Run Agent")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    return parser.parse_args()

def get_config() -> FullConfig:
    """
    Call this anywhere in your code to get the config from memory.
    Throws an error if you forgot to initialize it.
    """
    try:
        return config_context.get()
    except LookupError:
        raise RuntimeError("Config has not been initialized! Call initialize_config() first.")