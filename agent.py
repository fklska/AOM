from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import re
from score import ScoringEngine

load_dotenv()

client = ChatOpenAI(
    model="moonshotai/kimi-k2.6",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("API_KEY"),
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

def execute(prompt: ChatPromptTemplate):
    chain = prompt | client
    answer = chain.invoke({})
    code_match = re.search(r'```python(.*?)```', answer.content, re.DOTALL)

    if code_match:
        clean_code = code_match.group(1).strip()
    else:
        clean_code = answer.content.strip()

    try:
        namespace = {"__name__": "__main__"}
        exec(clean_code, namespace)
    except Exception as e:
        print("Ошибка", e)


    if os.path.exists("output/test.csv") & os.path.exists("output/train.csv"):
        res = ScoringEngine.score()

        return res.roc_auc

    print("Код не рабочий")
    return 0