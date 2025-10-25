from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")


message=[
    ("system","tell me jokse about {topic}"),
    HumanMessage(content="tell 3 jokes"),
]



prompt_temp=ChatPromptTemplate.from_messages(message)

print("====Promp temp====")
prompt=prompt_temp.invoke({"topic": "epam"})
print(prompt)

print("====output using ai======")

res=llm.invoke(prompt)
print(res.content)
