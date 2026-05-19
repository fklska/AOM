from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
SOURCE_CSV = Path("DATA.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42

README = """Titanic — бинарная классификация выживания.

Файлы в data/:
- train.csv — обучающая выборка
- test.csv — тестовая выборка

Колонки:
- client_id — идентификатор пассажира (бывший PassengerId)
- target — целевая переменная: выжил (1) или нет (0), бывший Survived
- Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked — признаки

Вспомогательных таблиц нет: достаточно загрузить train.csv и test.csv,
применить препроцессинг и сохранить результат в output/train.csv и output/test.csv
с теми же именами колонок client_id и target.
"""


def main():
    df = pd.read_csv(SOURCE_CSV)
    df = df.rename(columns={"PassengerId": "client_id", "Survived": "target"})

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["target"],
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)
    (DATA_DIR / "readme.txt").write_text(README, encoding="utf-8")

    print(f"Сохранено {DATA_DIR / 'train.csv'}: {train_df.shape}")
    print(f"Сохранено {DATA_DIR / 'test.csv'}: {test_df.shape}")
    print(f"Сохранено {DATA_DIR / 'readme.txt'}")


if __name__ == "__main__":
    main()
