import asyncio #content creators and reviewer
from autogen_agentchat.ui import Console
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
 
from dotenv import load_dotenv
load_dotenv()
 
openai_api_key=os.environ.get("OPENAI_API_KEY")
async def main() -> None:
    model_client=OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=openai_api_key
    )
 
    content_creator=AssistantAgent(
        name="King",   
        model_client=model_client,   
        system_message="You are a content creator assistant based on {topic}.",      
        model_client_stream=True,        
    )
    content_reviewver=AssistantAgent(
        name="Queen",
        model_client=model_client,
        system_message="""you need to review the content.""",
        model_client_stream=True,
    )   
    topic=input("Enter topic:").strip()
    creator_result=await content_creator.run(task=f"give a content based on {topic}")
    print("Creator",creator_result.messages[-1].content)
    reviewver_result=await content_reviewver.run(task=f"""The creator message:
    {creator_result.messages[-1].content}.
    evaluvate the content """)
    print("reviewer:",reviewver_result.messages[-1].content)
    creator_result=await content_creator.run(task="""The reviewver message:
    {reviewver_result.messages[-1].content}.
    Please provide reviced content.""")
    print("++++++++++++++++creator+++++++++",creator_result.messages[-1].content)
    await model_client.close()
asyncio.run(main())