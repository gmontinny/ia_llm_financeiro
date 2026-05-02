import sys
import io
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult, AgentStream
from src.utils import format_response
from src.config import AGENT_SYSTEM_PROMPT


async def run_agent(agent, query: str):
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()

    try:
        handler = agent.run(query)

        async for event in handler.stream_events():
            if isinstance(event, ToolCallResult):
                print(f"Call {event.tool_name} with args {event.tool_kwargs}\nReturned: {event.tool_output}")
            elif isinstance(event, AgentStream):
                print(event.delta, end="", flush=True)

        response = await handler
        formatted = format_response(str(response), return_thinking=True)
    finally:
        sys.stdout = old_stdout

    return formatted, captured.getvalue()


def create_agent(tools, llm):
    return FunctionAgent(tools=tools, llm=llm, system_prompt=AGENT_SYSTEM_PROMPT)
