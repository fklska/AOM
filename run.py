import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Перфекционистский препроцессор для задачи бинарной классификации
    повторных покупок (reorder prediction).

    Этапы:
    1. Загрузка всех таблиц из `data/`.
    2. Агрегация `orders` + `order_items` (только `prior`-заказы —
       защита от утечки целевой информации).
    3. INNER JOIN справочников и агрегатов к `train`/`test`.
    4. Автоматический анализ типов колонок с эвристикой для ID и
       низкокардинальных целочисленных признаков.
    5. Imputation: медиана для числовых, "missing" для категориальных.
    6. Кодирование: OneHotEncoder (< 20 уникальных), FrequencyEncoder (≥ 20).
    7. Стандартизация числовых признаков.
    8. Сохранение результатов в `output/`.
    """

    def __init__(
        self,
        data_dir: str = 'data',
        output_dir: str = 'output',
        cat_uniq_threshold: int = 20,
        numeric_as_cat_threshold: int = 25,
        random_state: int = 42
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.cat_uniq_threshold = cat_uniq_threshold
        self.numeric_as_cat_threshold = numeric_as_cat_threshold
        self.random_state = random_state

        # Колонки, которые никогда не трогаем как признаки
        self.exclude_cols = {'row_id', 'target'}
        # Явные внешние ключи — исключаем из масштабирования/кодирования
        self.id_keys = {'user_id', 'product_id', 'order_id'}

        # Метаданные, заполняемые в процессе
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.id_cols: List[str] = []
        self.medians: Dict[str, float] = {}
        self.freq_maps: Dict[str, Dict] = {}
        self.ohe: OneHotEncoder | None = None
        self.ohe_feature_names: List[str] = []
        self.scaler: StandardScaler | None = None

    # ------------------------------------------------------------------
    # 1. Загрузка
    # ------------------------------------------------------------------
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Загружает все CSV из `data_dir` в словарь."""
        files = {
            'train': 'train.csv',
            'test': 'test.csv',
            'users': 'users.csv',
            'orders': 'orders.csv',
            'order_items': 'order_items.csv',
            'products': 'products.csv',
            'aisles': 'aisles.csv',
            'departments': 'departments.csv',
            'data_dictionary': 'data_dictionary.csv'
        }
        data: Dict[str, pd.DataFrame] = {}
        for key, fname in files.items():
            path = self.data_dir / fname
            if not path.exists():
                raise FileNotFoundError(f"Отсутствует обязательный файл: {path}")
            data[key] = pd.read_csv(path)
            print(f"[+] Загружен `{fname}`: {data[key].shape}")
        return data

    # ------------------------------------------------------------------
    # 2. Агрегация истории заказов (data-leakage-safe)
    # ------------------------------------------------------------------
    def build_transaction_features(
        self, data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Создаёт признаки на основе `order_items` и `orders`.
        Используется только `eval_set == 'prior'`, чтобы исключить
        попадание информации из целевого заказа в признаки.
        """
        orders = data['orders'].copy()
        items = data['order_items'].copy()

        # Защита от утечки: оставляем только историю
        if 'eval_set' in orders.columns:
            orders = orders[orders['eval_set'] == 'prior'].copy()

        # INNER JOIN: только товары, присутствующие в обеих таблицах
        tx = items.merge(orders, on='order_id', how='inner')

        # --- User × Product агрегаты ---
        up_agg = (
            tx.groupby(['user_id', 'product_id'])
            .agg(
                up_orders_count=('order_id', 'nunique'),
                up_reordered_mean=('reordered', 'mean'),
                up_add_to_cart_mean=('add_to_cart_order', 'mean'),
                up_last_order_number=('order_number', 'max'),
                up_avg_days_since_prior=('days_since_prior_order', 'mean')
            )
            .reset_index()
        )

        # --- Product-агрегаты (глобальная популярность товара) ---
        prod_agg = (
            tx.groupby('product_id')
            .agg(
                p_total_orders=('order_id', 'nunique'),
                p_unique_users=('user_id', 'nunique'),
                p_reordered_mean=('reordered', 'mean'),
                p_avg_add_to_cart=('add_to_cart_order', 'mean')
            )
            .reset_index()
        )

        print(f"[+] Агрегаты: user-product {up_agg.shape}, product {prod_agg.shape}")
        return up_agg, prod_agg

    # ------------------------------------------------------------------
    # 3. INNER JOIN всех справочников к train/test
    # ------------------------------------------------------------------
    def merge_all(
        self,
        data: Dict[str, pd.DataFrame],
        up_agg: pd.DataFrame,
        prod_agg: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Последовательно выполняет INNER JOIN базовых выборок со справочниками
        и агрегатами. Отсутствующие связи отбрасываются.
        """
        out = {}
        for split in ['train', 'test']:
            base = data[split].copy()
            df = base

            # Справочники
            df = df.merge(data['users'], on='user_id', how='inner')
            df = df.merge(data['products'], on='product_id', how='inner')
            df = df.merge(data['aisles'], on='aisle_id', how='inner')
            df = df.merge(data['departments'], on='department_id', how='inner')

            # Агрегированные транзакции
            df = df.merge(up_agg, on=['user_id', 'product_id'], how='inner')
            df = df.merge(prod_agg, on='product_id', how='inner')

            # Воспроизводимый порядок
            df = df.sort_values('row_id').reset_index(drop=True)

            out[split] = df
            print(f"[+] `{split}` после объединения: {df.shape}")
        return out['train'], out['test']

    # ------------------------------------------------------------------
    # 4. Автоматический анализ типов колонок
    # ------------------------------------------------------------------
    def analyze_columns(self, df: pd.DataFrame):
        """
        Распределяет колонки на три группы:
        - id_cols — ключи связей, не участвуют в обучении как признаки;
        - numeric_cols — масштабируются StandardScaler;
        - categorical_cols — кодируются OHE или частотой.

        Эвристика: целочисленные колонки с числом уникальных значений
        < `numeric_as_cat_threshold` считаются категориальными.
        """
        self.id_cols = []
        self.numeric_cols = []
        self.categorical_cols = []

        for col in df.columns:
            if col in self.exclude_cols:
                continue

            if col in self.id_keys:
                self.id_cols.append(col)
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                nuniq = df[col].nunique(dropna=True)
                if nuniq < self.numeric_as_cat_threshold:
                    self.categorical_cols.append(col)
                else:
                    self.numeric_cols.append(col)
            else:
                self.categorical_cols.append(col)

        print(
            f"[+] Типы: numeric={len(self.numeric_cols)}, "
            f"categorical={len(self.categorical_cols)}, id={len(self.id_cols)}"
        )
        print(f"    Числовые: {self.numeric_cols}")
        print(f"    Категориальные: {self.categorical_cols}")

    # ------------------------------------------------------------------
    # 5. Обработка пропусков
    # ------------------------------------------------------------------
    def handle_missing(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Числовые → медиана (по train); категориальные → 'missing'."""
        for col in self.numeric_cols:
            if train[col].isna().any() or test[col].isna().any():
                med = train[col].median()
                self.medians[col] = med
                train[col] = train[col].fillna(med)
                test[col] = test[col].fillna(med)

        for col in self.categorical_cols:
            if train[col].isna().any() or test[col].isna().any():
                train[col] = train[col].fillna('missing')
                test[col] = test[col].fillna('missing')

        print("[+] Пропуски обработаны")
        return train, test

    # ------------------------------------------------------------------
    # 6. Кодирование категориальных признаков
    # ------------------------------------------------------------------
    def encode_features(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Разделение по кардинальности:
        - < threshold уникальных → OneHotEncoder;
        - ≥ threshold уникальных → FrequencyEncoder,
          где частота $f_i = \frac{n_i}{N}$ вычисляется только на `train`.
        """
        low_card = [
            c for c in self.categorical_cols
            if train[c].nunique() < self.cat_uniq_threshold
        ]
        high_card = [
            c for c in self.categorical_cols
            if train[c].nunique() >= self.cat_uniq_threshold
        ]

        print(f"[+] Кодирование: OneHot {len(low_card)}, Frequency {len(high_card)}")

        # --- OneHotEncoder ---
        if low_card:
            self.ohe = OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False,
                dtype=np.float32
            )
            ohe_train = self.ohe.fit_transform(train[low_card])
            ohe_test = self.ohe.transform(test[low_card])

            self.ohe_feature_names = self.ohe.get_feature_names_out(low_card).tolist()

            ohe_train_df = pd.DataFrame(
                ohe_train, columns=self.ohe_feature_names, index=train.index
            )
            ohe_test_df = pd.DataFrame(
                ohe_test, columns=self.ohe_feature_names, index=test.index
            )

            train = train.drop(columns=low_card)
            test = test.drop(columns=low_card)
            train = pd.concat([train, ohe_train_df], axis=1)
            test = pd.concat([test, ohe_test_df], axis=1)

        # --- FrequencyEncoder ---
        for col in high_card:
            freq = train[col].value_counts(normalize=True).to_dict()
            self.freq_maps[col] = freq
            train[col] = train[col].map(freq).astype(np.float32)
            test[col] = test[col].map(freq).astype(np.float32)
            # Категории, отсутствующие в train, получают вес 0.0
            test[col] = test[col].fillna(0.0)

        return train, test

    # ------------------------------------------------------------------
    # 7. Масштабирование числовых признаков
    # ------------------------------------------------------------------
    def scale_features(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        StandardScaler для всех числовых колонок:
        $$z = \frac{x - \mu}{\sigma}$$
        где $\mu$ и $\sigma$ — среднее и стандартное отклонение `train`.
        """
        if not self.numeric_cols:
            return train, test

        self.scaler = StandardScaler()
        train[self.numeric_cols] = self.scaler.fit_transform(
            train[self.numeric_cols]
        )
        test[self.numeric_cols] = self.scaler.transform(
            test[self.numeric_cols]
        )
        print("[+] Числовые признаки отмасштабированы")
        return train, test

    # ------------------------------------------------------------------
    # 8. Сохранение результатов
    # ------------------------------------------------------------------
    def save(self, train: pd.DataFrame, test: pd.DataFrame):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_path = self.output_dir / 'train.csv'
        test_path = self.output_dir / 'test.csv'

        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)

        print(f"[+] Сохранено `{train_path}`: {train.shape}")
        print(f"[+] Сохранено `{test_path}`: {test.shape}")

    # ------------------------------------------------------------------
    # Полный пайплайн
    # ------------------------------------------------------------------
    def run(self):
        print("=" * 60)
        print("Запуск препроцессинга")
        print("=" * 60)

        data = self.load_data()
        up_agg, prod_agg = self.build_transaction_features(data)
        train, test = self.merge_all(data, up_agg, prod_agg)

        self.analyze_columns(train)
        train, test = self.handle_missing(train, test)
        train, test = self.encode_features(train, test)
        train, test = self.scale_features(train, test)

        self.save(train, test)

        print("=" * 60)
        print("Готово. Обработанные данные находятся в `output/`.")
        print("=" * 60)


if __name__ == '__main__':
    processor = DataPreprocessor()
    processor.run()
