from smolagents import CodeAgent, DuckDuckGoSearchTool,InferenceClientModel


from dotenv import load_dotenv
import os

load_dotenv()



HF_Token=os.getenv("HF_API_TOKEN")


model=InferenceClientModel(token=HF_Token)


agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)


agent.run("what is psit kanpur?")

