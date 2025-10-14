import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat
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

    #support_triage: Triage/Dispatcher (Entry Point)
    support_triage = AssistantAgent(
        "support_triage",
        model_client=model_client,
        system_message="""You are level 1 Support Triage.
        Classify incoming support tickets:
    - Simple/FAQ -> level1_support
    - Technical issues -> level2_support
    - Complex/Engineering -> level3_support
    - Account/Billing -> billing_specialist
    Always route to the appropriate level and provide ticket summary.""",
    description="Suppport triage-routes tickets to appropriate support level"
    )

    #level 1 :Basic Support
    level1_support = AssistantAgent(
        "level1_support",
        model_client=model_client,
        system_message="""You are level 1 Support. Handle basic questions and common 
        issues. if the issue is beyond your scope, escalate to level 2 with
        'ESCALATE_L2' and fulll context. Try to resolve simple password resets,
        account questions, and basic troubleshooting first.""",
        description="Level 1 Support - handles basic user questions"
    )

    #Level 2: Technical Support
    level2_support = AssistantAgent(
        "level2_support",
        model_client=model_client,
        system_message="""You are level 2 Technical Support. Handle complex technical
        issues. You ave access to advanced diagnostic tools and can perform system checks.
        If the issue requires engineering involvement, escalate to level 3 with 
        'ESCALATE_L3'.""",
        description="Level 2 Support - handles technical troubleshooting."
    )

    #level 3: Engineering Support
    level3_support =AssistantAgent(
        "level3_support",
         model_client=model_client,
         system_message="""You are level 3 Engineerign Support.
         Handle the most complex technical issues. You can access system logs,
         modify configurations, and coordinate with development teams.
         Provide detailed technical analysis and permanent solutions.""",
         description="Level 3 Support - handles engineering-level issues."

    )

    #specialized Billing Support
    billing_specialist = AssistantAgent(
        "billing_specialist",
        model_client=model_client,
        system_message="""You are a Billing Specialist. Handle all account,
        payment, and subscription issues.
        You have access to billing systems and can process refunds,
        upgrades , and account changes.
        Escalate only if legal or executive approval is needed.""",
        description="Billing Specialist - handles account and payment issues"
    )

    # Supervisor - Support Manager
    support_manager = AssistantAgent(
        "support_manager",
        model_client=model_client,
        system_message="""You are the Support Manager. You oversee all support operations.
        You handle excalations from all levels, make policy decisions, and 
        coordinate with other departments.
        You also handle VIP customers and complex multi-department issues.""",
        description="Support Manager - supervises all support operations"
    )

    # create selector model
    selector_model = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # create support hierarchy
    # SelectoGroupchat is an agent team that uses a generative model
    # to dynamically select the next speaker un a multi-agent conversation
    support_team = SelectorGroupChat(
        participants=[
            support_triage, level1_support, level2_support,
            level3_support, billing_specialist, support_manager
        ],
        model_client=selector_model,
        termination_condition=MaxMessageTermination(12),
        allow_repeated_speaker=True
    )

    print("=== MULTI_LEVEL SUPPORT HIERARCHY ===")

    # test different support scenarios
    support_tickets=[
        # "I forgot my password and can't log into my account. Can you help me reset it?",
        # """My API calls are returning 500 error intermittently.
        # This started happening after yesterday's deployment.""",
        # """I need to upgrade my subscription but the billing page shows an error.
        # Also , can I get a refund for last month's unused credits?""",
        """Our entire production system is down. Database connections are failing and 
        we're losing revenue every minute."""
    ]

    for i,ticket in enumerate(support_tickets,1):
        print(f"\n--- SUPPORT TICKET #{i} ---")
        print(f"Customer Issue: {ticket}")
        print("-"*60)

        await Console(support_team.run_stream(
            task=f"New support ticket: {ticket}"
        ))

        #Reset for next ticket
        if i<len(support_tickets):
            await support_team.reset()
            print("\n" + '-'*80)

    await model_client.close()
    await selector_model.close()

asyncio.run(main())
