Titanic — бинарная классификация выживания.

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
