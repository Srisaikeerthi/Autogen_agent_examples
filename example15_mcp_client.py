#math_mcp_server.py
#pip install -U "autogen-ext[mcp]"
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv   
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams,mcp_server_tools
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
 
async def main():
    load_dotenv()

    #get the current python interpreter from the venv
    python_path = sys.executable
    server_script = Path(__file__).parent/"math_mcp_server.py"

    print(f"Using Python: {python_path}", file=sys.stderr)
    print(f"Server script: {server_script}", file=sys.stderr)

    server_params = StdioServerParams(
        command = python_path,  # Use current venv python
        args=[str(server_script)],
        env={
            "PYTHONUNBUFFERED" : "1",
            "PYTHONPATH" : os.getcwd(),
            **os.environ
        },
        read_timeout_seconds=60
    )

    try:
        print("Connecting to MCP Server...",file=sys.stderr)
        tools=await mcp_server_tools(server_params)
        print(f"Got {len(tools)} tools", file=sys.stderr)

        model_client=OpenAIChatCompletionClient(
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY")
        )

        agent = AssistantAgent(
            name="math_assistant",
            model_client=model_client,
            tools=tools,
            system_message="You are a math assistant."
        )

        await Console(
            agent.run_stream(
                task="Calculate 25//5",
                #cancellaion_token=CancellationToken()
            )
        )
    except Exception as e:
        print(f"Error: {e}",file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)


if __name__=="__main__":
    asyncio.run(main())
