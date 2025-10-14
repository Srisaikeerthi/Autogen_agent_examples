# external termination example
import asyncio
from autogen_agentchat.agents import AssistantAgent 
from autogen_agentchat.conditions import ExternalTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import CancellationToken
import os
from dotenv import load_dotenv

load_dotenv()

async def main() -> None:
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create agents
    brainstormer = AssistantAgent(
        "brainstormer",
        model_client=model_client,
        system_message="You generate creative ideas and solutions.",
    )
    
    evaluator = AssistantAgent(
        "evaluator",
        model_client=model_client,
        system_message="You evaluate and critique ideas, providing constructive feedback.",
    )
    
    # Create external termination condition
    external_termination = ExternalTermination()
    max_messages = MaxMessageTermination(20)
    
    # Combine termination conditions with OR logic
    combined_termination = external_termination | max_messages
    
    team = RoundRobinGroupChat(
        [brainstormer, evaluator],
        termination_condition=combined_termination
    )
    
    print("=== EXTERNAL TERMINATION EXAMPLE ===")
    print("Team will brainstorm for 3 seconds, then be stopped externally...")
    
    # Run team in background
    task = asyncio.create_task(
        Console(team.run_stream(task="Brainstorm innovative features for a productivity app"))
    )
    
    # Wait 3 seconds then stop externally  
    await asyncio.sleep(3)
    external_termination.set()
    
    try:
        result = await task
        print(f"\nTeam stopped: {result.stop_reason}")
    except Exception as e:
        print(f"Task handling completed: {e}")
    
    # Example with cancellation token
    print("\n=== CANCELLATION TOKEN EXAMPLE ===")
    cancellation_token = CancellationToken()
    
    # Start new task
    task2 = asyncio.create_task(
        team.run(
            task="Now brainstorm marketing strategies",
            cancellation_token=cancellation_token
        )
    )
    
    # Cancel immediately
    cancellation_token.cancel()
    
    try:
        result2 = await task2
    except asyncio.CancelledError:
        print("Task was successfully cancelled")
    
    await model_client.close()

asyncio.run(main())