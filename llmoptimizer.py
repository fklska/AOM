from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from agent import execute, build_prompt, DATA
from score import ScoringEngine
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

load_dotenv()
score = ScoringEngine()
OPTIMIZER_SYSTEM_PROMPT = """
Ты — оптимизатор промптов для LLM-агента. Твоя единственная цель — сгенерировать новый, максимально эффективный промпт (состоящий из трёх частей), который повысит ROC-AUC модели CatBoost, обучаемой **на стандартных параметрах** после предобработки данных агентом.

**Как работает пайплайн:**
1. Ты получаешь историю предыдущих попыток.
2. Каждая попытка — это полный текст промпта для агента (состоит из роли, инструкции, ограничений + описание данных), а также итоговый ROC-AUC.
3. Ты анализируешь эту историю и создаёшь улучшенный промпт.
4. Агент, получив твой промпт + описание данных, пишет Python-скрипт предобработки.
5. CatBoost обучается на результате и измеряется ROC-AUC.

**Формат входных данных (история):**
```json
[
  {{"prompt": "<role>\\n<instruction>\\n<constraints>\\nОписание данных:\\n<DATA>", "roc-auc": 0.637}},
  {{"prompt": "...", "roc-auc": 0.584}}
]
```

**Что тебе нужно сделать:**
- Изучи историю: определи, какие подходы к предобработке (обработка пропусков, кодирование категорий, feature engineering, отбор признаков, работа с выбросами, порядок объединения таблиц) привели к высокому ROC-AUC, а какие — к провалу.
- Скомбинируй удачные идеи из топовых попыток, отбрось вредные или бесполезные.
- Ты **не ограничен** исходным набором фраз. Можешь генерировать принципиально новые инструкции, усложнять или упрощать пайплайн, добавлять нестандартные эвристики предобработки — главное, чтобы это логически могло улучшить качество бинарной классификации на CatBoost.
- Помни, что данные всегда табличные, реляционные, без временных рядов и аномальных структур.
- Важно чтобы код был рабочим, лучше он будет простым, но рабочим
- Если roc-auc у какого-либо промпта равен 0, это значит что агент на основе этого промпта написал нерабочий код

**Жёсткие технические требования (должны быть соблюдены в `task` и/или `constraints`):**
- Исходные таблицы лежат в папке `data/`, включая `train.csv`, `test.csv` и вспомогательные CSV-файлы.
- Агент должен сохранить итоговые обработанные таблицы строго как `output/train.csv` и `output/test.csv`.
- При объединении вспомогательных таблиц с `train.csv` и `test.csv` используется **только `inner` join** — иначе размерность данных неконтролируемо взрывается.
- Агент выполняет **исключительно** предобработку данных. Никакого обучения моделей, никакой кросс-валидации, никакой визуализации или анализа в коде.
- Агент возвращает **только** исполняемый Python-код, без комментариев, без markdown-блоков и без пояснительного текста вне кода.
- Описание конкретных колонок и таблиц (`readme.txt`) будет добавлено к твоему промпту автоматически на стороне пользователя. **Не включай описание данных и не ссылайся на конкретные названия колонок в своём ответе** — промпт должен быть универсальным и работать с любым подобным датасетом.

**Формат твоего ответа:**
Верни строго три текстовых поля:

1. `role` — роль агента (например, кем он является и в чём его экспертиза).
2. `task` — детальные инструкции по предобработке данных. Это основное место для улучшений: опиши пошагово, что агент должен сделать с данными (загрузка, объединение, очистка, кодирование, feature engineering, сохранение).
3. `constraints` — ограничения, запреты и требования к коду/поведению агента.

Каждое поле — одна строка (многострочная, если требуется). Не добавляй ничего кроме этих трёх компонентов. Не пиши вступлений и заключений.

**Ключевой наказ:** твоя цель — максимизировать ROC-AUC. Если видишь, что простые подходы работают лучше сложных — упрости. Если видишь, что агент теряет полезную информацию — добавь инструкции по её сохранению. Если CatBoost мог бы выиграть от специфичного кодирования или генерации признаков — обяжи агента сделать это.
"""

OPTIMIZER_HUMAN_PROMPT = """
На основе истории оптимизируй и создай новый промпт для LLM агента:

{history}
"""

PromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", OPTIMIZER_SYSTEM_PROMPT),
        ("user", OPTIMIZER_HUMAN_PROMPT)
    ]
)
class PromptAnswer(BaseModel):
    role: str = Field(description="Роль агента, который будет предобрабатывать данные")
    task: str = Field(description="Инструкции для агента по предобработке данных")
    constraints: str = Field(description="Ограничения для агента")

optimizer = ChatOpenAI(
    model="moonshotai/kimi-k2.6",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("API_KEY"),
    reasoning_effort="minimal",
    extra_body={
        "reasoning": {"enabled": True},
        "provider": {
            "order": ["fireworks"],
            "allow_fallbacks": False
        },
    },
)
chain = PromptTemplate | optimizer.with_structured_output(PromptAnswer)

# role, task, constraint, roc-auc
BASIC = [
    [0, 0, 3, 0],
    [0, 5, 2, 0.584295],
    [0, 5, 9, 0.637306],
    [0, 2, 5, 0.655533]
]

prompts = pd.read_csv("prompts.csv", sep=";")
roles = prompts.role.to_list()
instructs = prompts.instruct.to_list()
constraints = prompts.constraints.to_list()

def main():
    history = list(map(lambda x: {"prompt": roles[x[0]] + "\n" + instructs[x[1]] + "\n" + constraints[x[2]] + "\n" + "Описание данных:\n" + DATA, "roc-auc": x[3]}, BASIC))
    for i in range(2):
        print("Вызов оптимизатора")
        answer = chain.invoke({
            "history": history
        })
        prompt = build_prompt(answer.role, answer.task, answer.constraints)
        auc = execute(prompt)
        print(auc)

        history.append({"prompt": answer.role + "\n" + answer.task + "\n" + answer.constraints + "\n" + "Описание данных:\n" + DATA, "roc-auc": auc})

        log_entry = {
            "role": answer.role,
            "instruct": answer.task,
            "constraint": answer.constraints,
            "roc_auc": auc
        }
        with open("llm_optimizer.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")



main()
