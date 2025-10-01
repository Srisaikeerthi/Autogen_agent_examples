import asyncio
#pip install -U "autogen-agentchat" "autogen-ext[openai]"

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os 
from dotenv import load_dotenv   #pip install dotenv  pip uninstall dotenv / python-dotenv

load_dotenv()

openai_api_key=os.environ.get("OPENAI_API_KEY")   #print env variables
#print(openai_api_key)

async def main() -> None:

    #create a model client
    model_client=OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=openai_api_key
    )

    #print(model_client)  #connection success

    #create an assistant agent
    agent=AssistantAgent("assistant",model_client=model_client)

    #run a simple task
    print(await agent.run(task="Say 'Hello World!'"))
    print("====================Response received from model=====================") #printed after model response is received (await msg)

    #close the client connection
    await model_client.close()


#Run the aync function
asyncio.run(main())