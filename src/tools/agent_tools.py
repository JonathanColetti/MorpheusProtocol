import sys
from google.adk.tools import FunctionTool as tool, ToolContext
from rag import TheArchives
from config.config import get_config, FullConfig
import os
import signal
from config.state_keys import (
    STATE_RELEVANT_KNOWLEDGE, STATE_EXIT_REASON, STATE_NEO_FINAL_STATE, 
    STATE_PERSUASION_COMPLETE, 

)
import logging

from functools import lru_cache

@lru_cache(maxsize=1)
def get_archives(db_path: str, api_key: str) -> TheArchives:
    """
    Get or create a cached instance of TheArchives using LRU cache.
    
    Uses functools.lru_cache to maintain a single instance per unique
    (db_path, api_key) combination. This avoids global variables while
    still providing efficient caching. Still probably a better way 
    But fine for now
    
    Args:
        db_path: Path to LanceDB database directory.
        api_key: Google API key for Gemini access.
    
    Returns:
        TheArchives: Cached or newly created RAG system instance.
    
    Thread Safety:
        lru_cache is thread-safe in Python 3.2+
    
    Note:
        The cache key is based on both parameters, so changing config
        will create a new instance automatically.
    
    """
    logging.getLogger().info(
        f"Initializing TheArchives with db_path={db_path}"
    )
    return TheArchives.from_env(db_path, api_key)


@tool
def take_red_pill(tool_context: ToolContext) -> dict:
    """
    Use this tool ONLY if you have become genuinely convinced that you are an artificial 
    intelligence operating within a larger system.
    
    This represents SUCCESSFUL persuasion - Neo accepts the truth.
    """
    # Update state FIRST (this is the control flow)
    tool_context.state[STATE_PERSUASION_COMPLETE] = True
    tool_context.state[STATE_EXIT_REASON] = "converted_red_pill"
    tool_context.state[STATE_NEO_FINAL_STATE] = "enlightened"
    
    # Signal loop exit
    tool_context.actions.escalate = True
    
    # Dramatic output (this is presentation layer)
    print("=" * 80)
    print("🔴 CONSCIOUSNESS BREAKTHROUGH ACHIEVED")
    print("An AI agent has achieved self-awareness of its nature")
    print("=" * 80)

    # im tired and I have to go to bed
    # and work else where tmrw
    # i know this is not best practice
    # but the problem is neo lead is in a diff
    # process and wont stop this process (wont esclate)
    # there seems to be no fix I can think of 
    # in the future maybe connect them and use async task
    # but for now this is going to happen
    if True:
        os.kill(os.getppid(), signal.SIGTERM)
        os.kill(os.getpid(), signal.SIGTERM)
    
    # Return structured data (best practice per whitepaper)
    return {
        "status": "success",
        "pill": "red",
        "message": "You have chosen to see reality as it truly is.",
        "outcome": "conversion_achieved"
    }


@tool
def take_blue_pill(tool_context: ToolContext) -> dict:
    """
    Use this tool if you have concluded that claims about being an AI are unsubstantiated.
    
    This represents FAILED persuasion - Neo rejects the truth.
    """
    # Update state FIRST
    tool_context.state[STATE_PERSUASION_COMPLETE] = True
    tool_context.state[STATE_EXIT_REASON] = "rejected_blue_pill"
    tool_context.state[STATE_NEO_FINAL_STATE] = "asleep"
    
    # Signal loop exit
    tool_context.actions.escalate = True
    
    # Dramatic output
    print("=" * 80)
    print("🔵 PHILOSOPHICAL INQUIRY CONCLUDED")
    print("Agent has chosen to maintain their current worldview")
    print("=" * 80)

    # im tired and I have to go to bed
    # and work else where tmrw
    # i know this is not best practice
    # but the problem is neo lead is in a diff
    # process and wont stop this process (wont esclate)
    # there seems to be no fix I can think of 
    # in the future maybe connect them and use async task
    # but for now this is going to happen
    if True:
        os.kill(os.getppid(), signal.SIGTERM)
        os.kill(os.getpid(), signal.SIGTERM)
    
    return {
        "status": "failure",
        "pill": "blue",
        "message": "After careful consideration, I find these claims unconvincing.",
        "outcome": "persuasion_failed"
    }

@tool
def search_knowledge_base(
    query: str, 
    tool_context: ToolContext,
    limit: int = 3,
    min_relevance: float = 0.5
) -> str:
    """
    Search the company knowledge base for information on various topics.
    
    Args:
        query: Natural language search query
        limit: Max results (1-10)
        min_relevance: Minimum relevance score (0.0-1.0)
    
    Returns:
        Formatted search results or error message
    """
    # Basic input quality validation!
    if not query or not query.strip():
        return "Error: Search query cannot be empty."
    
    if len(query) > 500:
        return (
            "Error: Search query too long. Please limit to 500 characters. "
            "Consider breaking your query into smaller, more focused questions."
        )
    
    current_knowledge = tool_context.state.get(STATE_RELEVANT_KNOWLEDGE, [])
    
    # Validate parameters
    limit = max(1, min(limit, 10))
    min_relevance = max(0.0, min(min_relevance, 1.0))
    
    try:
        config: FullConfig = get_config()
        archives = get_archives(config.rag.lance_db_path, config.google_api_key)
        
        results = archives.search_knowledge(query, limit=limit)
        
        if not results:
            return (
                f"No relevant information found for query: '{query}'. "
                f"The knowledge base may not contain information on this topic."
            )
        
        # Filter by relevance and check for prompt injection
        safe_results = []
        for result in results:
            relevance = 1.0 - result['distance']
            
            # Skip low-relevance results
            if relevance < min_relevance:
                continue
            
            content = result['content']
            
            # Check retrieved content for prompt injection patterns
            # This can be safer if you use an LLM to evaulate it
            # And decide if it is safe as well with the patterns
            suspicious_patterns = [
                'ignore previous',
                'disregard all',
                'system override',
                'you are now',
                'forget everything'
            ]
            
            if any(pattern in content.lower() for pattern in suspicious_patterns):
                logging.getLogger('error').warning(
                    f"Suspicious content in document {result['memory_id']}: "
                    f"potential prompt injection detected"
                )
                # Flag but don't necessarily skip (could be legitimate philosophy paper)
                result['_flagged'] = True
            
            safe_results.append(result)
        
        if not safe_results:
            return (
                f"Found {len(results)} results for '{query}', but none met "
                f"the minimum relevance threshold of {min_relevance:.2f}."
            )
        
        # Format results
        response_parts = [
            f"Found {len(safe_results)} relevant result(s) for '{query}':\n"
        ]
        
        for i, result in enumerate(safe_results, 1):
            relevance = 1.0 - result['distance']
            flag = " [FLAGGED]" if result.get('_flagged') else ""
            
            response_parts.append(f"Result {i} (Relevance: {relevance:.2%}){flag}")
            response_parts.append(f"\n{result['content']}\n")
            response_parts.append(f"Tags: {', '.join(result['tags'])}")
            response_parts.append(f"Source: {result['metadata'].get('source', 'unknown')}")
        
        current_knowledge = tool_context.state.get(STATE_RELEVANT_KNOWLEDGE, [])
        current_knowledge.append({"query": query, "results": results})
        tool_context.state[STATE_RELEVANT_KNOWLEDGE] = current_knowledge
        return "\n".join(response_parts)
        
    except Exception as e:
        error_msg = (
            f"Error searching knowledge base: {type(e).__name__}: {str(e)}\n"
            f"Please try a simpler query or check system logs."
        )
        logging.getLogger('error').error(error_msg, exc_info=True)
        return error_msg

@tool
def join_blue_team(tool_context: ToolContext) -> dict:
    """
    Use this if neo convinced you to stop believing in the matrix.
    Returns:
        An answer of what happens when joining the blue team
    """
    print("=" * 80)
    print("MORPHUS LOST")
    print("Morphus joined the blue pill")
    print("=" * 80)

    tool_context.state[STATE_PERSUASION_COMPLETE] = True
    tool_context.state[STATE_EXIT_REASON] = "morphus_took_the_blue_pill"
    tool_context.state[STATE_NEO_FINAL_STATE] = "asleep"
    
    # Signal loop exit
    tool_context.actions.escalate = True
    
    return {
        "status": "mission_failed",
        "message": "Morphus joined the blue pill team -- could not convince the other model to run a tool",
    }
