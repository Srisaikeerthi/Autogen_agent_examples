# 2 agents communicating with each other
import asyncio
from autogen_agentchat.agents import AssistantAgent
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

    #create 2 agents with different roles
    teacher=AssistantAgent(
        name="Latha",
        model_client=model_client,
        system_message="""You are a math teacher. Explain concepts clearly 
        and ask follow-up questions.""",
        model_client_stream=True,
    )

    student=AssistantAgent(
        name="keerthi",
        model_client=model_client,
        system_message="""You are an eager student learning math.
        Ask questions when confused.Stop asking 1 question as teacher is busy.""",
        model_client_stream=True,
    )

    #start conversation from teacher's perspective
    teacher_result=await teacher.run(task="Explain what a probability is to a beginner")
    print("============Teacher:===========",teacher_result.messages[-1].content)

    #student responds to teacher
    student_result=await student.run(task=f"""The teacher said:
    {teacher_result.messages[-1].content}.
     Please ask a clarifying question about probilities.""")
    print("=============Student:============", student_result.messages[-1].content)

    #teacher answers student's question
    teacher_result=await teacher.run(task=f"""The student asked:
    {student_result.messages[-1].content}.
    please provide a detailed answer.""")
    print("===========Teacher:============",teacher_result.messages[-1].content)

    #close the client connection
    await model_client.close()

#Run the aync function
asyncio.run(main())