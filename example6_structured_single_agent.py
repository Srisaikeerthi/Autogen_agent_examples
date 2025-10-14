import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os 
from autogen_core.models import ModelInfo
from dotenv import load_dotenv   
from autogen_agentchat.messages import StructuredMessage
from pydantic import BaseModel
from typing import List,Literal
load_dotenv()
openai_api_key=os.environ.get("GEMINI_API_KEY")
class MovieReview(BaseModel):
    title:str
    genre:List[str]
    rating:int
    sentiment:Literal["Positive","Negative","Mixed"]
    summary:str
    pros:List[str]
    cons:List[str]
    recommendation:str
 
async def main() -> None:
    # we are using OpenAi for google gemini 2.5 pro --because it is OpenAi compatable
    model_client=OpenAIChatCompletionClient(  #set standards by openAI
        model="gemini-2.5-pro",
        model_info=ModelInfo(vision=True, function_calling=True, json_output=True,
        family="unknown",structured_output=True),
        api_key=openai_api_key
    )
 
    movie=input("Enter movie name: ").strip()
    movie_critic_agent = AssistantAgent(
        name="movie_critic",
        model_client=model_client,
        system_message="""You are a professional movie critic.
        Analyze movies throughly and provide structured reviews.you must review this {movie} movie only and rate it out of 100""",
        output_content_type=MovieReview,
    )
    print("===Movie Review===")
    movie_result= await movie_critic_agent.run(task="Review the movie {movie} this movie only. ")
    if isinstance(movie_result.messages[-1],StructuredMessage):
        review= movie_result.messages[-1].content
        print(f"===Title===: {review.title}")
        print(f"===Genre===: {', '.join(review.genre)}")
        print(f"===Rating===: {review.rating}/100")
        print(f"===Sentiment===: {review.sentiment}") 
        print(f"===Summary===: {review.summary}")
        print(f"===Pros===: {', '.join(review.pros)}")
        print(f"===Cons===: {', '.join(review.cons)}")
        print(f"===Recommendation===: {review.recommendation}")
 
        await model_client.close()
asyncio.run(main())