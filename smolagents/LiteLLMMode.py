from smolagents import LiteLLMModel
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("⚠️ GOOGLE_API_KEY not found. Please set it in your .env file.")

# Force LiteLLM to use Gemini REST API, not Vertex AI
os.environ["LITELLM_API_BASE"] = "https://generativelanguage.googleapis.com/v1beta"

model = LiteLLMModel(
    model="gemini-2.5-flash",
    api_key=api_key,
    provider="google",
    temperature=0.2
)

messages = []

while True:
    user_input = input("enter a message::: ")
    if user_input.lower() == "exit":
        break

    messages.append({"role": "user", "content": user_input})
    resp = model(messages, max_tokens=500)
    assistant_message = resp.content

    print("Assistant:", assistant_message)
    messages.append({"role": "assistant", "content": assistant_message})
