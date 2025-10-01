#yessss
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os 
from dotenv import load_dotenv   #pip install dotenv  pip uninstall dotenv / python-dotenv
from autogen_agentchat.ui import Console
import requests

load_dotenv()

#openai_api_key=os.environ.get("OPENAI_API_KEY")   #print env variables
#print(openai_api_key)



#define a simple weather tool. Hardcoded by us.
async def get_weather(city:str)->str:
    """Get the weather for a given city."""
    #referred example 10 from langchain_example1 folder
    try:
        API_KEY=os.getenv("OPEN_WEATHER_API_KEY")
            
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response=requests.get(url,timeout=60)
        data=response.json()

        temp=data['main']['temp']
        description=data['weather'][0]['description']

        return f"The weather in {city} is {temp}!C with {description}."
    
    except Exception as e:
        return f"Error: {str(e)}"

'''async def is_valid_city(city:str)->bool:
    """Check the city is valid or not. If the given city is valid then only call get_weather tool. Else respond as Give a """
    return True'''

async def main() -> None:
    #create a model client
    model_client=OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    #print(model_client)  #connection success

    #create agent with tool capability
    agent=AssistantAgent(
        name="Joe",    #can give name to the agent . not like the before code.
        model_client=model_client,     #brain of the agent
        tools=[get_weather],         #tools of the agent
        system_message="""You are a helpful weather assistant.Check if the given 
        city is valid.If yes proceed or respond as 'invalid city'.""",
        model_client_stream=True,        #shouldthe repsonse be a stream?
    )   
    
    while True:
        city=input("Enter city name(or 'exit): ").strip()
        if city.lower()=="exit":
            break

        #run the agent with streamiing output
        response=agent.run_stream(task=f"What is the weather in {city}?")
        await Console(response)
        print("-"*40)
        
    #close the client connection
    await model_client.close()

#Run the aync function
asyncio.run(main())