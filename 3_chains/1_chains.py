from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

message=[
    ("system","write joke on {topic}"),
    ("human","{joke} jokes"),
]

prompt=ChatPromptTemplate.from_messages(message)

chain=prompt|llm

result=chain.invoke({"topic":"epam","joke":"3"})
print(result.content)



