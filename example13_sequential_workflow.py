import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Handoff
from autogen_agentchat.conditions import HandoffTermination,TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os 
from dotenv import load_dotenv   

load_dotenv()

async def main() -> None:
    model_client=OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    #define sequential workflow agents with handoffs
    requirements_analyst=AssistantAgent(
        "requirements_analyst",
        model_client=model_client,
        handoffs=[Handoff(target="architect")],
        system_message="""You analyze requirements and create detailed specifications.
        When complete,hand off to architect.""",
    )

    architect = AssistantAgent(
        "architect",
        model_client=model_client,
        handoffs=[Handoff(target="developer")],
        system_message="""You design system architecture based on requirements.
        When complete, hand off to the developer.""",
    )

    developer = AssistantAgent(
        "developer",
        model_client=model_client,
        handoffs=[Handoff(target="COMPLETE")],
        system_message="""You create implementation plans based on architecture.
         When finished, say 'TERMINATE'. """,
    )

    #define termination conditions
    handoff_complete = HandoffTermination(target="COMPLETE")
    text_termination = TextMentionTermination("TERMINATE")

    #create team with combined termination conditions
    team = RoundRobinGroupChat(
        [requirements_analyst,architect,developer],
        termination_condition=handoff_complete | text_termination
    )

    print("=== SEQUENTIAL WORKFLOW WITH HANDOFFS ===")
    await Console(team.run_stream(task="""Design and plan the
    implementation of a real-time chat application with user
    authentication and message history."""))

    await model_client.close()

asyncio.run(main())