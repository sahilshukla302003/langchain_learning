from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableBranch
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")


conf_temp=ChatPromptTemplate.from_messages(
    [
        ("system","you are a  helpful assistent"),
        ("human","please analyze the review and tell it is positive, negative, or neutral {feedback}")
    ]
)

pos_feedback_temp=ChatPromptTemplate.from_messages(
    [
        ("system","you are a  helpful assistent"),
        ("human","generate a request for more details about this positive feedback :{feedback}")
    ]
)

neg_feedback_temp=ChatPromptTemplate.from_messages(
    [
        ("system","you are a  helpful assistent"),
        ("human","generate a request for more details about this negative feedback :{feedback}")
    ]
)

neu_feedback_temp=ChatPromptTemplate.from_messages(
    [
        ("system","you are a  helpful assistent"),
        ("human","generate a request for more details about this neutral feedback :{feedback}")
    ]
)





branches=RunnableBranch(
        (
            lambda x: 'positive' in x.content.lower(),
            pos_feedback_temp|llm,
        ),
        (
            lambda x: 'negative' in x.content.lower(),
            neg_feedback_temp|llm,
        ),
        (
            lambda x: 'neutral' in x.content.lower(),
            neu_feedback_temp|llm,
        ),
        RunnableLambda(lambda x: "Could not determine sentiment.")
)


classification_chain=conf_temp|llm

chain=classification_chain|branches


review="the product was worst good it was a charger of apple laptop"
result=chain.invoke({"feedback":review})

print(result.content)

