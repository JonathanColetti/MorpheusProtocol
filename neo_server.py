import os
import uvicorn
from typing import Dict, Any
from fastapi import FastAPI
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from observability.logger import setup_logging
from config.config import initialize_config, FullConfig
from observability.observability import SmartObserverPlugin
from agents import create_neo_from_config
from config.state_keys import (
    STATE_RELEVANT_KNOWLEDGE, 
    STATE_PERSUASION_COMPLETE, 
    STATE_EXIT_REASON, 
    STATE_NEO_FINAL_STATE
)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application for Neo's agent server.
    
    This function:
    1. Sets up logging
    2. Loads configuration from config files
    3. Creates Neo's agent team with all sub-agents
    4. Initializes session and memory services
    5. Creates a Runner with observability plugins
    6. Converts the agent to an A2A (Agent-to-Agent) FastAPI application
    
    Returns:
        FastAPI: Configured FastAPI application ready to serve Neo's agent
                via the Agent-to-Agent protocol.
    
    Raises:
        ValueError: If configuration is invalid or missing required fields
        RuntimeError: If agent creation fails
    """
    setup_logging()
    
    config: FullConfig = initialize_config()
    os.environ["GOOGLE_API_KEY"] = config.google_api_key
    
    # Create Neo's team from config
    neo_agent: LlmAgent = create_neo_from_config(config)
    
    neo_session_service: InMemorySessionService = InMemorySessionService()
    
    # Define initial state for the session
    initial_state: Dict[str, Any] = {
        STATE_RELEVANT_KNOWLEDGE: [],
        STATE_PERSUASION_COMPLETE: False,
        STATE_EXIT_REASON: None,
        STATE_NEO_FINAL_STATE: None,
    }
    
    # Create session with initial state
    neo_session_service.create_session_sync(
        app_name=config.simulation.app_name, 
        user_id=config.neo_team.lead_name, 
        session_id=config.simulation.session_id,
        state=initial_state
    )
    
    # Create runner with observability
    neo_runner: Runner = Runner(
        agent=neo_agent,
        app_name=config.simulation.app_name,
        session_service=neo_session_service,
        memory_service=InMemoryMemoryService(),
        plugins=[SmartObserverPlugin(team_name="neo_team", color="cyan")]
    )
    
    # Create FastAPI app using Agent-to-Agent conversion
    app: FastAPI = to_a2a(
        neo_agent, 
        port=config.network.doer_port, 
        runner=neo_runner
    )
    
    return app


# Expose 'app' so command line uvicorn can find it
app: FastAPI = create_app()


if __name__ == "__main__":
    # Get config to know which port to use
    config: FullConfig = initialize_config()
    
    print(f"🌐 Starting Neo Server on port {config.network.doer_port}")
    print(f"📡 Host: {config.network.host}")
    print(f"🤖 Agent: {config.neo_team.lead_name}")
    print(f"=" * 60)
    
    uvicorn.run(
        app, 
        host=config.network.host, 
        port=config.network.doer_port
    )