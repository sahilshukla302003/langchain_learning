from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Prompt with variables
message = [
    ("system", "write jokes on {topic}"),
    ("human", "tell me {joke} jokes"),
]

prompt = ChatPromptTemplate.from_messages(message)

# Convert output to uppercase
upper_case = RunnableLambda(lambda x: x.content.upper())

# Create the chain
chain = prompt | llm | upper_case

# Invoke the chain
result = chain.invoke({"topic": "EPAM", "joke": "3"})

print(result)  # ✅ Just print the string
