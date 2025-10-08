import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.model_context import BufferedChatCompletionContext
import os 
from dotenv import load_dotenv   

load_dotenv()

#create specialized agent functions that can be used as tools
async def research_agent_tool(query:str) -> str:
    """Research agent that provides market data and insights."""
    model_client=OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    research_agent=AssistantAgent(
        "research_specialist",
        model_client=model_client,
        system_message="""You are a specialized research agent.
        Provide concise, factual research findings."""
    )

    result=await research_agent.run(task=f"Research this topic: {query}")
    await model_client.close()
    return result.messages[-1].content

#sync def calculator_agent_tool(expression: str) -> str:


async def main() -> None:
    model_client=OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    #create coordinator agent that uses other agents as tools
    coordinator = AssistantAgent(
        "coordinator",
        model_client=model_client,
        system_message="""You are a project coordinator.
         Use the research tool to provide comprehensive analysis.""",
         max_tool_iterations=5,
    )

    #create reviewer agent
    reviewer = AssistantAgent(
        "reviewer",
        model_client=model_client,
        system_message="""You are a project reviewer. Evaluate the coordinator's 
        analysis and suggest improvements. """,
    )

    #create team
    team=RoundRobinGroupChat(
        [coordinator,reviewer],
        termination_condition=MaxMessageTermination(6)
    )

    print("=== AGENT_AS_TOOL PATTERN ===")
    result=await team.run(task="""Analyze the ROI of investing $100,000 in a 
    SAAS startup.Research the market and calculate potential returns.""")

    print(f"\nfinal analysis completed with {len(result.messages)} messages")

    print("Conversation Transcript:")
    for msg in result.messages:
        print(f"{msg.source}: {msg.content}\n")

    await model_client.close()

asyncio.run(main())
