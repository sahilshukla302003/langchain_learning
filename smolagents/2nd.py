from smolagents import CodeAgent, VisitWebpageTool,InferenceClientModel


from dotenv import load_dotenv
import os

load_dotenv()



HF_Token=os.getenv("HF_API_TOKEN")


model=InferenceClientModel(token=HF_Token)

agent = CodeAgent(tools=[VisitWebpageTool()],
                   model=model,
                   additional_authorized_imports=["requests","markdownify"],
                   )



agent.run("Visit https://psit.ac.in and tell me what this college is about.")

