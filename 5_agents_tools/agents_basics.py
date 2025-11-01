from dotenv import load_dotenv
from langsmith import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
import os
from langchain.agents import create_react_agent, AgentExecutor


load_dotenv()


client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

@tool
def get_current_time() -> str:
    """Returns the current local time."""
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

tools = [get_current_time]

# ✅ now this will work
prompt = client.pull_prompt("hwchase17/react", include_model=True)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

executor = create_react_agent_executor(llm, tools)

result = executor.invoke({"messages": [{"role": "user", "content": "What time is it?"}]})
print(result)
