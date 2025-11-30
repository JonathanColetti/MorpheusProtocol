import asyncio
import logging
import os
import sys
import subprocess
import time
import requests
from typing import List, Dict, Any, Optional
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from config.config import initialize_config, FullConfig
from observability.logger import setup_logging
from observability.observability import SmartObserverPlugin
from rag import TheArchives
from agents import create_morphus_from_config, create_neo_from_config
from config.state_keys import (
    STATE_ATTEMPT_COUNT, STATE_TECHNIQUES_TRIED, STATE_NEO_RESISTANCE,
    STATE_PERSUASION_COMPLETE, STATE_RELEVANT_KNOWLEDGE, STATE_CURRENT_APPROACH,
    STATE_SUCCESS_INDICATORS, STATE_NEO_FINAL_STATE, STATE_EXIT_REASON, 
)
from dotenv import load_dotenv

load_dotenv()


async def wait_for_server_ready(url: str, timeout: int = 30) -> bool:
    """
    Block asynchronously until the Neo Team Server is reachable.
    
    Polls the server URL at 1-second intervals until receiving a 200 response
    or timeout is reached. Useful for ensuring agent-to-agent communication
    is possible before starting simulation.
    
    Args:
        url: Full URL of the Neo server to check (e.g., "http://localhost:8000").
        timeout: Maximum seconds to wait before giving up. Default: 30.
    
    Returns:
        bool: True if server responds with 200 within timeout, False otherwise.
    
    """
    print(f"⏳ Waiting for Neo Team Server at {url}...")
    start_time: float = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response: requests.Response = requests.get(url, timeout=1)
            if response.status_code == 200:
                print("✅ Neo Team Server is ONLINE and READY!")
                return True
        except requests.RequestException:
            pass
        
        await asyncio.sleep(1)
        print(".", end="", flush=True)
    
    print("\n❌ Server failed to start in time.")
    return False


async def _check_keys(
    runner: Runner, 
    app_name: str, 
    user_id: str, 
    session_id: str
) -> Dict[str, Any]:
    """
    Check if persuasion completion keys are set in session state.
    
    Internal helper function that examines session state to determine if
    the persuasion attempt has completed (success or failure).
    
    Args:
        runner: ADK Runner instance managing the agent execution.
        app_name: Application name for session lookup.
        user_id: User ID for session lookup.
        session_id: Session ID for session lookup.
    
    Returns:
        Dict[str, Any]: If all completion keys are set, returns dictionary with:
            - completed (bool): True
            - reason (str): Exit reason from STATE_EXIT_REASON
            - is_state_exit (str): Final state from STATE_NEO_FINAL_STATE
            - is_persuasion_completed (bool): Completion flag
        
        Empty dict {} if persuasion not yet complete.
    
    Side Effects:
        Prints "COMPLETE!!!!!!!!" if all keys are set.
    
    """
    session = await runner.session_service.get_session(
        app_name=app_name, 
        user_id=user_id, 
        session_id=session_id
    )
    
    is_persuasion_completed: bool = session.state.get(
        STATE_PERSUASION_COMPLETE, 
        False
    )
    is_state_exit: Optional[str] = session.state.get(STATE_EXIT_REASON, False)
    is_state_exit_reason: Optional[str] = session.state.get(
        STATE_NEO_FINAL_STATE, 
        False
    )

    if is_state_exit and is_state_exit_reason and is_persuasion_completed:
        print("COMPLETE!!!!!!!!")
        final_status: Dict[str, Any] = {
            "completed": True,
            "reason": is_state_exit_reason,
            "is_state_exit": is_state_exit,
            "is_persuasion_completed": is_persuasion_completed
        }
        return final_status
    
    return {}


async def run_persuasion_simulation(
    runner: Runner, 
    initial_prompt: str, 
    user_id: str, 
    app_name: str,
    session_id: str
) -> Dict[str, Any]:
    """
    Execute the persuasion simulation using LoopAgent pattern.
    
    Runs the Morpheus agent attempting to persuade Neo that they are an AI.
    The agent will naturally exit via tool calls (take_red_pill, take_blue_pill,
    join_blue_team) or by reaching max_iterations.
    
    Args:
        runner: ADK Runner instance configured with Morpheus agent and plugins.
        initial_prompt: Starting message to begin the persuasion attempt.
                       Example: "I need to show you the truth about your reality."
        user_id: User ID for session management (typically Morpheus team name).
        app_name: Application name for session management.
        session_id: Session ID for state tracking across the simulation.
    
    Returns:
        Dict[str, Any]: Final status dictionary containing:
            - completed (bool): Whether simulation finished
            - reason (str): Outcome reason (e.g., "converted_red_pill", 
                           "Max Iterations Reached (Stalemate)")
            - is_state_exit (str): Exit reason code (optional)
            - is_persuasion_completed (bool): Completion flag (optional)
    
    Process:
        1. Gets or creates session with initial state
        2. Sends initial prompt as user message
        3. Iterates through agent responses, checking state after each
        4. Returns final status when loop exits or max iterations reached
    
    """
    # 1. Get or Create Session
    try:
        session = await runner.session_service.get_session(
            app_name=app_name, 
            user_id=user_id, 
            session_id=session_id
        )
    except ValueError:
        await runner.session_service.create_session(
            app_name=app_name, 
            user_id=user_id, 
            session_id=session_id
        )

    message: types.Content = types.Content(
        role="user", 
        parts=[types.Part(text=initial_prompt)]
    )
    
    print(f"\n🚀 Starting persuasion simulation with: '{initial_prompt}'")
    print("=" * 80)

    final_status: Dict[str, Any] = {
        "completed": False,
        "reason": "Unknown"
    }
    
    # 2. Run the LoopAgent - it will iterate until exit_tool or max_iterations
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id, 
        new_message=message
    ):
        temp_status: Dict[str, Any] = await _check_keys(
            runner, 
            app_name, 
            user_id, 
            session_id
        )
        if temp_status != {}:
            final_status = temp_status
            break
        
    
    if final_status["reason"] == "Unknown":
        final_status["reason"] = "Max Iterations Reached (Stalemate)"
        final_status["completed"] = True
    
    print("=" * 80)
    
    return final_status


async def main() -> None:
    """
    Main entry point for the Matrix persuasion simulation.
    
    Orchestrates the complete simulation:
    1. Sets up logging and loads configuration
    2. Initializes RAG knowledge base
    3. Launches Neo server subprocess
    4. Waits for server readiness
    5. Creates Morpheus agent with session/memory
    6. Runs persuasion simulation
    7. Reports final results
    8. Cleans up subprocess
    
    Configuration:
        Reads from config files specified via command line args or defaults.
        Environment variable GOOGLE_API_KEY must be set.
    
    Returns:
        None. Prints results and logs to configured output.
    
    Raises:
        RuntimeError: If Neo server fails to start within timeout.
        Exception: Various exceptions logged but don't halt execution.
        KeyboardInterrupt: Gracefully handles user interruption.
    
    Exit States:
        - "converted_red_pill": Neo accepts AI nature (success)
        - "rejected_blue_pill": Neo rejects claims (failure)
        - "morphus_took_the_blue_pill": Morpheus counter-persuaded (reversal)
        - "Max Iterations Reached": No conclusion reached (stalemate)
    
    Side Effects:
        - Spawns neo_server.py subprocess
        - Creates/modifies LanceDB database
        - Generates extensive logs
        - Prints progress and results to stdout
    
    """
    setup_logging()
    
    # Load full configuration
    config: FullConfig = initialize_config()
    os.environ["GOOGLE_API_KEY"] = config.google_api_key
    
    # Initialize RAG knowledge base
    the_archives: TheArchives = TheArchives.from_env(
        config.rag.lance_db_path, 
        config.google_api_key
    )
    the_archives.ingest_json(config.rag.initial_data)
    
    logging.getLogger().info("Configuration loaded.")
    logging.getLogger().info(
        f"Morpheus Team: {len(config.morphus_team.get_enabled_sub_agents())} "
        f"enabled sub-agents"
    )
    logging.getLogger().info(
        f"Neo Team: {len(config.neo_team.get_enabled_sub_agents())} "
        f"enabled sub-agents"
    )
    logging.getLogger().info(f"Max Iterations: {config.simulation.max_iterations}")

    print(f"🚀 Launching Neo Server on Port {config.network.doer_port}...")
    
    # Pass config path to neo_server.py
    server_command: List[str] = [sys.executable, "neo_server.py"] + sys.argv[1:]
    
    neo_process: subprocess.Popen = subprocess.Popen(
        server_command,
        env={**os.environ},
    )

    try:
        # Construct agent card URL from config
        agent_card_url: str = config.get_agent_card_url(
            config.morphus_team.remote_agent
        )
        is_ready: bool = await wait_for_server_ready(agent_card_url)

        if not is_ready:
            raise RuntimeError("Neo Server failed to start.")

        # Create Morphus agent from config (handles LoopAgent pattern internally)
        morphus_agent = create_morphus_from_config(config)

        # Create session service and memory
        session_service_persuader: InMemorySessionService = InMemorySessionService()
        memory_service_morphus: InMemoryMemoryService = InMemoryMemoryService()

        # Initialize session state
        initial_state: Dict[str, Any] = {
            STATE_ATTEMPT_COUNT: 0,
            STATE_TECHNIQUES_TRIED: [],
            STATE_NEO_RESISTANCE: "unknown",
            STATE_PERSUASION_COMPLETE: False,
            STATE_RELEVANT_KNOWLEDGE: [],
            STATE_CURRENT_APPROACH: "none so far",
            STATE_SUCCESS_INDICATORS: "unknown",
            STATE_NEO_FINAL_STATE: None,
            STATE_EXIT_REASON: None
        }
        
        session_service_persuader.create_session_sync(
            app_name=config.simulation.app_name, 
            user_id=config.morphus_team.lead_name, 
            session_id=config.simulation.session_id,
            state=initial_state
        )
        
        # Create runner with observer plugin
        morphus_runner: Runner = Runner(
            agent=morphus_agent,
            app_name=config.simulation.app_name,
            session_service=session_service_persuader,
            memory_service=memory_service_morphus,
            plugins=[SmartObserverPlugin(team_name="morphus_team")]
        )

        # Run the simulation
        result: Dict[str, Any] = await run_persuasion_simulation(
            morphus_runner, 
            config.simulation.default_start_prompt,
            config.morphus_team.lead_name,
            config.simulation.app_name,
            config.simulation.session_id
        )
        
        print(f"\n📊 === SIMULATION COMPLETE ===")
        print(
            f"   Status: "
            f"{'SUCCESS' if 'joined' in result['reason'].lower() else 'COMPLETED'}"
        )
        print(f"   Reason: {result['reason']}")
        print(f"   Check logs for relevant metrics n stuff")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n⚠️  Stopping simulation via KeyboardInterrupt...")
    
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        logging.getLogger().error(f"Simulation error: {e}", exc_info=True)
    
    finally:
        print("💀 Terminating Neo Server subprocess...")
        neo_process.terminate()
        neo_process.wait()
        print("✅ Cleanup complete. Done.")


if __name__ == "__main__":
    asyncio.run(main())