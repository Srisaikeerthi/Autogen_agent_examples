import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os 
from autogen_core.models import ModelInfo
from dotenv import load_dotenv 
from autogen_agentchat.ui import Console
load_dotenv()
openai_api_key=os.environ.get("GEMINI_API_KEY")
async def main() -> None:
    model_client=OpenAIChatCompletionClient(
        model="gemini-2.5-pro",
        model_info=ModelInfo(vision=True, function_calling=True, json_output=True,
        family="unknown",structured_output=True),
        api_key=openai_api_key
    )
    writer_agent = AssistantAgent(
        name="creative_writer",
        model_client=model_client,
        system_message="""You are a creative writer specializing in science ficition.
        write a engaging stories with vivid descriptions and compelling characters""",
        model_client_stream=True
    )
    print("starting creative writing session... \n")
    await Console(
        writer_agent.run_stream(task="""Write a short science fiction story about a time
        traveler who discovers something unexpected  about their past.
        Make it engaging and include dialogue."""),
        output_stats=True, #show token usage
    )
    await model_client.close()
asyncio.run(main())