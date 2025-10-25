from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

result = llm.invoke("add 1 and 2")
print(result)
print(result.content)
