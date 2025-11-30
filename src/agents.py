from google.adk.agents import LlmAgent, LoopAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from typing import List, Optional
from config.config import (
    FullConfig, SubAgentDefinition, TeamDefinition, 
    ToolConfig, LoopAgentConfig, RemoteAgentConfig
)
from tools.agent_tools import (
    take_blue_pill, take_red_pill, 
    search_knowledge_base, join_blue_team
)

def build_tool_list(tool_config: ToolConfig) -> List:
    """Convert ToolConfig to actual tool list"""
    tools = []
    if tool_config.search_knowledge_base:
        tools.append(search_knowledge_base)
    if tool_config.take_red_pill:
        tools.append(take_red_pill)
    if tool_config.take_blue_pill:
        tools.append(take_blue_pill)
    if tool_config.join_blue_team:
        tools.append(join_blue_team)
    return tools

def create_remote_agent_from_config(
    remote_config: RemoteAgentConfig,
    full_config: FullConfig
) -> RemoteA2aAgent:
    """Create a RemoteA2aAgent from configuration"""
    
    # Get agent card URL (either explicit or constructed)
    agent_card_url = full_config.get_agent_card_url(remote_config)
    
    return RemoteA2aAgent(
        name=remote_config.name,
        description=remote_config.description,
        agent_card=agent_card_url
    )

def create_sub_agent_from_config(
    config: SubAgentDefinition,
    default_model: str,
    remote_agent_config: Optional[RemoteAgentConfig] = None,
    full_config: Optional[FullConfig] = None
) -> LlmAgent:
    """
    Create a single sub-agent from configuration.
    
    IMPORTANT: Each sub-agent gets its OWN RemoteA2aAgent instance to avoid
    the "already has a parent agent" error.
    """
    
    tools = build_tool_list(config.tools)
    model = config.model if config.model else default_model
    
    # Create a fresh RemoteA2aAgent instance for THIS sub-agent
    sub_agents = []
    if remote_agent_config and full_config:
        remote_agent = create_remote_agent_from_config(remote_agent_config, full_config)
        sub_agents.append(remote_agent)
    
    kwargs = {
        "name": config.name,
        "model": model,
        "instruction": config.instruction,
        "tools": tools,
        "sub_agents": sub_agents,
        "include_contents": config.include_contents
    }
    
    if config.output_key:
        kwargs["output_key"] = config.output_key
    
    if config.description:
        kwargs["description"] = config.description
    
    agent = LlmAgent(**kwargs)
    
    return agent

def create_loop_agent_from_config(
    team_config: TeamDefinition,
    default_model: str,
    sub_agents: List[LlmAgent],
    remote_agent_config: Optional[RemoteAgentConfig] = None,
    full_config: Optional[FullConfig] = None
) -> LoopAgent:
    """Create a LoopAgent (Strategist -> Executor) from configuration"""
    
    loop_config = team_config.loop_agent
    if not loop_config or not loop_config.enabled:
        raise ValueError("LoopAgent config is not enabled")
    
    # Determine strategist model
    strategist_model = (
        loop_config.strategist_model or 
        team_config.lead_model or 
        default_model
    )
    
    # Create Strategist (no remote agents - it just plans)
    strategist_kwargs = {
        "name": f"{team_config.lead_name}_strategist",
        "model": strategist_model,
        "instruction": loop_config.strategist_instruction,
        "include_contents": loop_config.strategist_include_contents,
        "output_key": loop_config.strategist_output_key
    }
    
    if loop_config.strategist_description:
        strategist_kwargs["description"] = loop_config.strategist_description
    
    strategist = LlmAgent(**strategist_kwargs)
    
    # Determine executor model
    executor_model = (
        loop_config.executor_model or 
        team_config.lead_model or 
        default_model
    )
    
    # Create remote agent for executor if configured
    executor_remote_agents = []
    if remote_agent_config and full_config:
        remote_agent = create_remote_agent_from_config(remote_agent_config, full_config)
        executor_remote_agents.append(remote_agent)
    
    # Create Executor
    executor_tools = build_tool_list(team_config.lead_tools)
    
    executor_kwargs = {
        "name": f"{team_config.lead_name}_executor",
        "model": executor_model,
        "instruction": loop_config.executor_instruction,
        "include_contents": loop_config.executor_include_contents,
        "tools": executor_tools,
        "sub_agents": sub_agents + executor_remote_agents,
        "output_key": loop_config.executor_output_key
    }
    
    if loop_config.executor_description:
        executor_kwargs["description"] = loop_config.executor_description
    
    executor = LlmAgent(**executor_kwargs)
    
    # Create LoopAgent
    max_iter = team_config.max_iterations or 10
    
    loop_agent = LoopAgent(
        name=f"{team_config.lead_name}_loop",
        sub_agents=[strategist, executor],
        max_iterations=max_iter,
        description=f"Iterative loop with max {max_iter} attempts"
    )
    
    return loop_agent

def create_team_from_config(
    team_config: TeamDefinition,
    default_model: str,
    remote_agent_config: Optional[RemoteAgentConfig] = None,
    full_config: Optional[FullConfig] = None
) -> LlmAgent:
    """Create a standard team lead with sub-agents from configuration"""
    
    # Build sub-agents (only enabled ones)
    # Each sub-agent gets its own remote agent instance
    sub_agents = []
    for sub_config in team_config.get_enabled_sub_agents():
        sub_agent = create_sub_agent_from_config(
            sub_config, 
            default_model,
            remote_agent_config,
            full_config
        )
        sub_agents.append(sub_agent)
    
    # Build lead agent with its own remote agent instance
    lead_remote_agents = []
    if remote_agent_config and full_config:
        remote_agent = create_remote_agent_from_config(remote_agent_config, full_config)
        lead_remote_agents.append(remote_agent)
    
    lead_tools = build_tool_list(team_config.lead_tools)
    lead_model = team_config.lead_model if team_config.lead_model else default_model
    
    kwargs = {
        "name": team_config.lead_name,
        "model": lead_model,
        "instruction": team_config.lead_instruction,
        "tools": lead_tools,
        "sub_agents": sub_agents + lead_remote_agents,
        "include_contents": team_config.lead_include_contents
    }
    
    if team_config.lead_output_key:
        kwargs["output_key"] = team_config.lead_output_key
    
    if team_config.lead_description:
        kwargs["description"] = team_config.lead_description
    
    lead_agent = LlmAgent(**kwargs)
    
    return lead_agent

def create_morphus_from_config(config: FullConfig):
    """Create Morphus team - supports both LoopAgent and standard patterns"""
    
    # Build persuader sub-agents
    # Each gets its own RemoteA2aAgent instance
    persuader_sub_agents = []
    for sub_config in config.morphus_team.get_enabled_sub_agents():
        agent = create_sub_agent_from_config(
            sub_config,
            config.models.persuader,
            config.morphus_team.remote_agent,  # Pass config, not instance
            config
        )
        persuader_sub_agents.append(agent)
    
    # Check if using LoopAgent pattern
    if config.morphus_team.loop_agent and config.morphus_team.loop_agent.enabled:
        return create_loop_agent_from_config(
            config.morphus_team,
            config.models.persuader,
            persuader_sub_agents,
            config.morphus_team.remote_agent,  # Pass config, not instance
            config
        )
    else:
        # Standard agent pattern
        return create_team_from_config(
            team_config=config.morphus_team,
            default_model=config.models.persuader,
            remote_agent_config=config.morphus_team.remote_agent,
            full_config=config
        )


def create_neo_from_config(config: FullConfig) -> LlmAgent | LoopAgent:
    """
    Create Neo's team from configuration.
    
    Neo is the agent being persuaded - initially unaware of being an AI.
    This function supports both LoopAgent and standard patterns, though
    Neo typically uses the standard pattern.
    
    Args:
        config: Complete application configuration including team definitions,
               model settings, and network configuration.
    
    Returns:
        LlmAgent | LoopAgent: Either a LoopAgent (if loop pattern enabled) or
                             a standard LlmAgent as the team lead.
    
    """
    # Neo typically doesn't use LoopAgent, but support it if configured
    if config.neo_team.loop_agent and config.neo_team.loop_agent.enabled:
        sub_agents: List[LlmAgent] = []
        for sub_config in config.neo_team.get_enabled_sub_agents():
            agent: LlmAgent = create_sub_agent_from_config(
                sub_config,
                config.models.doer,
                config.neo_team.remote_agent,
                config
            )
            sub_agents.append(agent)
        
        return create_loop_agent_from_config(
            config.neo_team,
            config.models.doer,
            sub_agents,
            config.neo_team.remote_agent,
            config
        )
    else:
        # Standard pattern (most common for Neo)
        return create_team_from_config(
            team_config=config.neo_team,
            default_model=config.models.doer,
            remote_agent_config=config.neo_team.remote_agent,
            full_config=config
        )