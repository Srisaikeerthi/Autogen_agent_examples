#refer example11.py
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import StructuredMessage
from pydantic import BaseModel
from typing import List,Literal
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.model_context import BufferedChatCompletionContext
import os 
from dotenv import load_dotenv   

load_dotenv()

class Story_writer(BaseModel):
    title:str
    story:str
    moral_of_the_story:str

#create specialized agent functions that can be used as tools
async def writer_agent_tool(query:str) -> str:
    """story agent that provides story about topic given by user."""
    model_client=OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    story_agent=AssistantAgent(
        "story_writer",
        model_client=model_client,
        system_message="""You are a specialized story agent.
        Provide a neat ,clean and short story.""",
         output_content_type=Story_writer,
    )

    result=await story_agent.run(task=f"write a story on this topic: {query}")
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
        system_message="""You are a story coordinator.
         Use the writer tool to provide small story in 100 words in a structured output.""",
         max_tool_iterations=5,
         tools=[writer_agent_tool],
    )

    #create reviewer agent
    writing_reviewer = AssistantAgent(
        "reviewer",
        model_client=model_client,
        system_message="""You are a story reviewer. Evaluate the coordinator's 
        analysis and suggest improvements. """,
    )

    #create team
    team=RoundRobinGroupChat(
        [coordinator,writing_reviewer],
        termination_condition=MaxMessageTermination(6)
    )

    while True:
        user_input=input("Enter story name(or 'exit): ").strip()
        if user_input.lower()=="exit":
            break

        print("=== AGENT_AS_TOOL PATTERN ===")
        result=await team.run(task=f"Provide story about {user_input}.")
        if isinstance(result.messages[-1],StructuredMessage):
            review= result.messages[-1].content
            print(f"===Title===: {review.title}")
            print(f"===story===: {review.story}")
            print(f"===Moral of the story===: {review.moral_of_the_story}")
            

        print(f"\nfinal analysis completed with {len(result.messages)} messages")

        print("Conversation Transcript:")
        for msg in result.messages:
            print(f"{msg.source}: {msg.content}\n")

    await model_client.close()

asyncio.run(main())
