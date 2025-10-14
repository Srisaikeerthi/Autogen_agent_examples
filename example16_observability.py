import os
import asyncio
import requests
from dotenv import load_dotenv
 
# LangSmith + OTEL instrumentation
from langsmith.integrations.otel import configure
from openinference.instrumentation.autogen import AutogenInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
 
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console
 
load_dotenv()
 
# 1) Configure LangSmith tracing once (project name can come from env)
# Falls back to LANGSMITH_PROJECT if not provided here.
configure(project_name=os.getenv("LANGSMITH_PROJECT"))  # uses LANGSMITH_* env vars [web:2][web:16]
 
# 2) Instrument AutoGen + OpenAI before creating any clients/agents
AutogenInstrumentor().instrument()  # traces agent turns, messages, tools [web:2]
OpenAIInstrumentor().instrument()   # traces OpenAI chat completions [web:2]
 
async def get_weather(city: str) -> str:
    try:
        API_KEY = os.getenv("OPEN_WEATHER_API_KEY")
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=60)
        data = response.json()
        temp = data["main"]["temp"]
        description = data["weather"][0]["description"]
        return f"{city}: {temp} !C, {description}"
    except Exception as e:
        return f"Error: {str(e)}"
 
async def main() -> None:
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
 
    agent = AssistantAgent(
        name="King",
        model_client=model_client,
        tools=[get_weather],
        system_message="You are a helpful assistant.",
        model_client_stream=True,
    )
 
    while True:
        city = input("Enter city name(or 'exit): ").strip()
        if city.lower() == "exit":
            break
 
        response = agent.run_stream(task=f"""
        What is the weather in {city}? You should not Explain anyother thing.
        if city is not valid just give Invalid city and no further conversation
        """)
        await Console(response)
        print("-" * 40)
 
    await model_client.close()
 
if __name__ == "__main__":
    asyncio.run(main())