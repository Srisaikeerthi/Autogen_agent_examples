import asyncio   #custom tool
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv
import random
import math
 
load_dotenv()
 
async def calculate_circle_area(radius:float) -> str:
    """Calculate the area of a circle given its radius."""
    print(f"Calculating area for radius:{radius}")
    area = math.pi * radius ** 2
    return f"The  area of a circle with radius {radius} is {area:.2f} square units."
 
async def roll_dice(sides: int = 6, count: int = 1) -> str:
    """Roll dice and return the results."""
    if count < 1 or count > 10:
        return "can only roll between 1 and 10 at atime."
    if sides < 2 or sides > 100:
        return f"Dice must have between 2 and 100 sides."
    results = [random.randint(1,sides)for _ in range(count)]
    total = sum(results)
    return f"Rolled {count}d{sides}: {results}(Total:{total})"
 
async def get_random_fact()-> str:
    """Get a random interesting fact."""
    facts = [
        "Octopuses have three hearts  and blue blood."
        "A group of flamingos is called a 'flamboyance'."
        "Homey never spoils.Archaelogists have found edible honey in ancient Egyptain tombs."
        "A shrimp's heart is in its head."
        "Bananas are berries, but strawberries aren't."
    ]
    return random.choice(facts)
 
async def main() -> None:
    model_client = OpenAIChatCompletionClient(
        model = "gpt-4",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    #create agent with mul tools 
    tool_agent = AssistantAgent(
        name="tool_master",
        model_client=model_client,
        tools=[calculate_circle_area,roll_dice,get_random_fact], #never call just register func name smart detacte by user input raduis
        system_message="""You are a helpful assistant with access to various tools.
        use them to help users with calculation, games, and interesting facts.you must call
        max 2 tools per requests.you do not have access  
        to call more than 2 tools at a time.""",  #autonomous behaviour over choosing tools 
        max_tool_iterations=2, #multiple tool call
    )
 
    #FYI: max_tool_iteration at most 3 iterations of toolsbefore stopping loop
    #The agent can be confused to exexute multiple iterations until the model stops
    #generating tool calls ot the maximum number of iterations is reacged
    #task = "Calculate the area of a circle with radius 5"
    tasks = [
        "Calculate the area of circle with radius 5",
        "Rool 2 six-sided dice",
        "Tool me a random facts",
        "calculate the area of a circle with radius 3.5 and then roll 3 dices with 8 sides each.",
        """calculate the area of a circle with radius 7,roll 4 ten-sided dice, and tell
        me a random fact."""  #tool call in chain
    ]
 
    for task in tasks:
        result = await tool_agent.run(task=task)
        print(f"Response:{result.messages[-1].content}")
    await model_client.close()
 
asyncio.run(main())