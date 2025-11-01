from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

message=[
    ("system","write joke on {topic}"),
    ("human","{joke} jokes"),
]

parser=StrOutputParser()

prompt=ChatPromptTemplate.from_messages(message)

chain=prompt|llm|parser

result=chain.invoke({"topic":"epam","joke":"3"})
print(result.content)



