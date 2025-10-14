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

async def main() -> None:
    model_client=OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    #create model contextx with different buffer sizes for memory management
    strategist_context=BufferedChatCompletionContext(buffer_size=5)
    #keeps the last 5 messages (including system message)
    analyst_context = BufferedChatCompletionContext(buffer_size=8)
    #keeps the last 8 messages (including system message)

    #create agents with different memory contexts
    strategist = AssistantAgent(
        "strategist",
        model_client=model_client,
        model_context=strategist_context,    #keep only last 5 messages in context
        system_message="""You are a business strategist.
        Focus on high-level strategy and planning."""
    )

    analyst = AssistantAgent(
        "analyst",
        model_client=model_client,
        model_context=analyst_context,
        system_message="""You are a financial analyst.
        Provide detailed financial analyis and projections."""
    )

    #create team
    team = RoundRobinGroupChat(
        [strategist,analyst],
        termination_condition=MaxMessageTermination(8)
    )

    print("=== CUSTOM CONTEXT MANAGEMENT ===")
    result = await Console(team.run_stream(task="""Analyze the business potential of entering the 
    electric vehicle market in India. Consider both strategic and financial aspects."""))

    print(result)
    #inspect the conversation
    print(f"\n================Total messages: {len(result.messages)}")
    print(f"==================Stop reason: {result.stop_reason}")
    
    '''
    #reset and continue with new context
    await team.reset()

    print("\n=== NEW CONVERSATION AFTER RESET ===")
    await Console(team.run_stream(task="Now analyze the renewable energy sector instead."))
'''
    await model_client.close()

asyncio.run(main())