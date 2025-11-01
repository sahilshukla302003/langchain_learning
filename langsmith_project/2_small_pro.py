from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser



load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser=StrOutputParser()

messages=[
    ("system","you are a report generator!!"),
    ("human","generate a thorough report about the topic: {topic}")
]



messages1=[
    ("system","you would extract important information as points!!"),
    ("human","take out 5 crucial information from the report: {text}")
]

prompt1=ChatPromptTemplate.from_messages(messages)

prompt2=ChatPromptTemplate.from_messages(messages1)


chain=prompt1|llm|parser|RunnableLambda(lambda x:{"text":x})|prompt2|llm|parser


res=chain.invoke({"topic":"psit kanpur"})

print(res)


