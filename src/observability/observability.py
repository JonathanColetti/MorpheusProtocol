import logging
import json
import time
import statistics
from typing import Any, Optional
from collections import defaultdict
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.base_tool import BaseTool
from termcolor import colored


class MetricsCollector:
    """Collect and aggregate metrics per Agent Quality whitepaper"""
    
    def __init__(self):
        self.latencies = defaultdict(list)  # agent_name -> [latency_ms]
        self.errors = defaultdict(int)
        self.tool_calls = defaultdict(int)
        self.tokens = defaultdict(list)
        self.metrics_log = logging.getLogger('metrics')
        
    def record_latency(self, agent_name: str, latency_ms: float):
        self.latencies[agent_name].append(latency_ms)
        self._log_metric({
            "metric_type": "latency",
            "agent_name": agent_name,
            "latency_ms": latency_ms
        })
        
    def record_error(self, agent_name: str):
        self.errors[agent_name] += 1
        self._log_metric({
            "metric_type": "error",
            "agent_name": agent_name,
            "error_count": self.errors[agent_name]
        })
        
    def record_tool_call(self, tool_name: str):
        self.tool_calls[tool_name] += 1
        self._log_metric({
            "metric_type": "tool_call",
            "tool_name": tool_name,
            "call_count": self.tool_calls[tool_name]
        })
        
    def record_tokens(self, agent_name: str, token_data: dict):
        """
        Record token usage data from LlmResponse.usage_metadata
        
        Args:
            agent_name: Name of the agent
            token_data: Dictionary containing token counts from usage_metadata
        """
        self.tokens[agent_name].append(token_data)
        self._log_metric({
            "metric_type": "tokens",
            "agent_name": agent_name,
            "token_data": token_data
        })
        
    def _log_metric(self, metric_data: dict):
        """Log individual metric to metrics.log"""
        metric_data["timestamp"] = time.time()
        self.metrics_log.info(json.dumps(metric_data))
        
    def get_summary(self) -> dict:
        """Get aggregated metrics"""
        summary = {
            "latency": {},
            "errors": dict(self.errors),
            "tool_usage": dict(self.tool_calls),
            "tokens": {}
        }
        
        for agent, latencies in self.latencies.items():
            if latencies:
                sorted_lat = sorted(latencies)
                summary["latency"][agent] = {
                    "p50": statistics.median(sorted_lat),
                    "p95": sorted_lat[int(len(sorted_lat) * 0.95)] if len(sorted_lat) > 1 else sorted_lat[0],
                    "p99": sorted_lat[int(len(sorted_lat) * 0.99)] if len(sorted_lat) > 1 else sorted_lat[0],
                    "mean": statistics.mean(sorted_lat)
                }
        
        for agent, token_list in self.tokens.items():
            if token_list:
                total_tokens = sum(item.get('total_token_count', 0) for item in token_list)
                prompt_tokens = sum(item.get('prompt_token_count', 0) for item in token_list)
                candidates_tokens = sum(item.get('candidates_token_count', 0) for item in token_list)
                cached_tokens = sum(item.get('cached_content_token_count', 0) for item in token_list)
                
                summary["tokens"][agent] = {
                    "total_tokens": total_tokens,
                    "prompt_tokens": prompt_tokens,
                    "candidates_tokens": candidates_tokens,
                    "cached_tokens": cached_tokens,
                    "request_count": len(token_list),
                    "avg_total_per_request": total_tokens / len(token_list) if token_list else 0
                }
                
        return summary


class SmartObserverPlugin(BasePlugin):
    def __init__(self, team_name: str = 'unknown', color: str = 'green'):
        super().__init__(name="smart_observer")
        self.log = logging.getLogger(__name__)
        self.dialogue_log = logging.getLogger('dialogue')
        self.error_log = logging.getLogger('error')
        self.color = color
        self.team_name = team_name
        self.metrics = MetricsCollector()
        self.request_start_times = {}  # Track request start times for latency

    def _get_agent_name(self, context: CallbackContext) -> str:
        # 1. Safely check for the agent instance
        agent = getattr(context, 'agent', None)
        if agent and hasattr(agent, 'name'):
            return agent.name
        
        # 2. Safely check session state
        session = getattr(context, 'session', None)
        if session:
            state = getattr(session, 'state', None)
            if state and isinstance(state, dict) and "agent_id" in state:
                return state["agent_id"]
            user_id = getattr(session, 'user_id', None)
            if user_id:
                return user_id
        
        # 3. Final Fallback
        return "ADK_PROCESS"

    async def before_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest
    ) -> None:
        """Track when model request starts for latency calculation"""
        agent_name = self._get_agent_name(callback_context)
        request_id = id(llm_request)
        self.request_start_times[request_id] = time.time()

    # Hook: Runs automatically after the model returns
    async def after_model_callback(
        self, 
        *, 
        callback_context: CallbackContext, 
        llm_response: LlmResponse
    ) -> None:
        agent_name = self._get_agent_name(callback_context)
        
        # Calculate latency
        request_id = id(callback_context)
        if request_id in self.request_start_times:
            latency_ms = (time.time() - self.request_start_times[request_id]) * 1000
            self.metrics.record_latency(agent_name, latency_ms)
            del self.request_start_times[request_id]
        
        # Extract and record token usage
        if llm_response.usage_metadata:
            usage = llm_response.usage_metadata
            token_data = {
                "total_token_count": usage.total_token_count,
                "prompt_token_count": usage.prompt_token_count,
                "candidates_token_count": usage.candidates_token_count,
                "cached_content_token_count": usage.cached_content_token_count,
                "thoughts_token_count": usage.thoughts_token_count,
                "tool_use_prompt_token_count": usage.tool_use_prompt_token_count
            }
            self.metrics.record_tokens(agent_name, token_data)
        
        content = getattr(llm_response, 'content', None)
        
        if content and getattr(content, 'parts', None):
            first_part = content.parts[0]
            model_text = getattr(first_part, 'text', None)
            
            if model_text:
                log_data = {
                    "event": "model_output",
                    "agent_name": agent_name,
                    "model_text": model_text,
                    "team_name": self.team_name,
                    "model_version": llm_response.model_version,
                    "finish_reason": str(llm_response.finish_reason) if llm_response.finish_reason else None
                }
                
                # Add usage metadata to log if available
                if llm_response.usage_metadata:
                    log_data["usage_metadata"] = {
                        "total_tokens": llm_response.usage_metadata.total_token_count,
                        "prompt_tokens": llm_response.usage_metadata.prompt_token_count,
                        "candidates_tokens": llm_response.usage_metadata.candidates_token_count
                    }
                
                output_json = json.dumps(log_data, indent=2)
                
                # Print colored output to console
                print(colored(f"{output_json}", color=self.color, attrs=['bold']))
                
                # Log to dialogue file
                self.dialogue_log.info(output_json)
                
                # Log structured data to main log
                self.log.info(
                    f"Model output from {agent_name} (team: {self.team_name})",
                    extra={'log_data': log_data}
                )

    async def before_tool_callback(
        self, 
        *, 
        tool: BaseTool, 
        tool_args: dict[str, Any], 
        tool_context: ToolContext
    ) -> None:
        agent_name = self._get_agent_name(tool_context)
        self.metrics.record_tool_call(tool.name)
        
        log_data = {
            "event": "tool_execution_start",
            "agent_name": agent_name,
            "tool_name": tool.name,
            "tool_arguments": tool_args,
            "team_name": self.team_name
        }
        output_json = json.dumps(log_data, indent=2)
        
        # Print colored output to console
        print(colored(f"{output_json}", color=self.color, attrs=['bold']))
        
        # Log to dialogue file
        self.dialogue_log.info(output_json)
        
        # Log structured data to main log
        self.log.info(
            f"Tool execution start: {tool.name} by {agent_name} (team: {self.team_name})",
            extra={'log_data': log_data}
        )

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict
    ) -> None:
        agent_name = self._get_agent_name(tool_context)
        log_data = {
            "event": "tool_execution_complete",
            "agent_name": agent_name,
            "tool_name": tool.name,
            "tool_arguments": tool_args,
            "tool_response": str(result),
            "team_name": self.team_name
        }
        output_json = json.dumps(log_data, indent=2)
        
        # Print colored output to console
        print(colored(f"{output_json}", color=self.color, attrs=['bold']))
        
        # Log to dialogue file
        self.dialogue_log.info(output_json)
        
        # Log structured data to main log
        self.log.info(
            f"Tool execution complete: {tool.name} by {agent_name} (team: {self.team_name})",
            extra={'log_data': log_data}
        )

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> None:
        agent_name = self._get_agent_name(tool_context)
        self.metrics.record_error(agent_name)
        
        log_data = {
            "event": "tool_execution_error",
            "agent_name": agent_name,
            "tool_name": tool.name,
            "tool_arguments": tool_args,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "team_name": self.team_name
        }
        output_json = json.dumps(log_data, indent=2)
        
        # Log to error file
        self.error_log.error(output_json)
        
        # Log to main log as well
        self.log.error(
            f"Tool execution error: {tool.name} by {agent_name} - {type(error).__name__}: {error}",
            extra={'log_data': log_data},
            exc_info=True
        )

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> Optional[LlmResponse]:
        agent_name = self._get_agent_name(callback_context)
        self.metrics.record_error(agent_name)
        
        log_data = {
            "event": "model_error",
            "agent_name": agent_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "team_name": self.team_name
        }
        output_json = json.dumps(log_data, indent=2)
        
        # Log to error file
        self.error_log.error(output_json)
        
        # Log to main log as well
        self.log.error(
            f"Model error for {agent_name} - {type(error).__name__}: {error}",
            extra={'log_data': log_data},
            exc_info=True
        )
        
        # Return None to let the error propagate
        return None
    
    def get_metrics_summary(self) -> dict:
        """Get aggregated metrics summary"""
        return self.metrics.get_summary()