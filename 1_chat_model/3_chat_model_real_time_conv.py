from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage


load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


system_message=SystemMessage(content="You are a helpful assistent")

print("Type exit to come out of chat")

chat_his=[]
while True:
    query=input("You::")
    if query.lower()=="exit":
        break
    
    chat_his.append(HumanMessage(content=query))
    res=llm.invoke(chat_his)
    response=res.content
    chat_his.append(AIMessage(content=response))
    print(f"AI Ans:: {res.content}")


print("--------Message History---------")
print(chat_his)

