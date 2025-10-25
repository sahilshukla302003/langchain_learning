from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")



template="write a mail about {topic}"

prompt_temp=ChatPromptTemplate.from_template(template)

print("====Promp temp====")
prompt=prompt_temp.invoke({"topic": "HR mail to epam for offer letter pls write full mail the hr manager name is rahul my name is sa"})
print(prompt)

print("====output using ai======")

res=llm.invoke(prompt)
print(res.content)
