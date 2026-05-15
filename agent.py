from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import re
from score import ScoringEngine

load_dotenv()
score = ScoringEngine()


client = ChatOpenAI(
    model="moonshotai/kimi-k2.6",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("API_KEY"),
    reasoning_effort="minimal",
    extra_body={
        "reasoning": {"enabled": False},
        "provider": {
            "order": ["fireworks"],
            "allow_fallbacks": False
        },
    },
    temperature=1.1,
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

def execute(prompt: ChatPromptTemplate):
    chain = prompt | client
    print("=====Вызов агента=====")
    answer = chain.invoke({})
    code_match = re.search(r'```python(.*?)```', answer.content, re.DOTALL)
    print("=====Исполнение кода=====")
    if code_match:
        clean_code = code_match.group(1).strip()
    else:
        clean_code = answer.content.strip()

    try:
        namespace = {"__name__": "__main__"}
        exec(clean_code, namespace)
    except Exception as e:
        print("Ошибка исполнения, код не рабочий", e)
        return 0

    try:
        res = score.score()
        return res.roc_auc
    except Exception as e:
        print("Ошибка датасета", e)
    print("Код не рабочий")
    return 0


if __name__ == "__main__":
    answ = client.invoke("Привет, как дела?")
    print(answ)