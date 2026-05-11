from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


client = ChatOpenAI(
    model="moonshotai/kimi-k2.6",
    base_url="https://openrouter.ai/api/v1",
    api_key="",
    reasoning_effort="minimal",
    extra_body={"reasoning": {"enabled": True}}
)

DATA = ""

with open("data/readme.txt", "r") as f:
    DATA = f.read().strip()

def build_prompt(role: str, instruct: str, constraints: str):
    messages = [
        ("system", role),
        ("user", instruct + "\n" + constraints + "\n" + "Описание данных:\n" + DATA)
    ]
    return ChatPromptTemplate.from_messages(messages)

def execute(prompt: str):
    chain = build_prompt() | client
    chain.invoke()