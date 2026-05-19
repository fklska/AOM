import os
import json
import glob
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Загрузка данных
data_dir = Path('data')
csv_files = [f for f in data_dir.glob('*.csv')]

train = None
test = None
auxiliary = []

for f in csv_files:
    df = pd.read_csv(f)
    name = f.name
    if name == 'train.csv':
        train = df.copy()
    elif name == 'test.csv':
        test = df.copy()
    else:
        auxiliary.append((name, df))

# 2. Объединение вспомогательных таблиц
for i, (name, aux_df) in enumerate(auxiliary):
    common_cols = [c for c in train.columns if c in aux_df.columns and c != 'target']
    if len(common_cols) == 0:
        continue
    join_col = 'client_id' if 'client_id' in common_cols else common_cols[0]
    
    if i == 0 and len(auxiliary) > 1:
        orig_train_rows = len(train)
        orig_test_rows = len(test)
        train = train.merge(aux_df, on=join_col, how='left', suffixes=('', f'_{name.replace(".csv", "")}'))
        test = test.merge(aux_df, on=join_col, how='left', suffixes=('', f'_{name.replace(".csv", "")}'))
        if len(train) != orig_train_rows:
            train = train.drop_duplicates(subset=[c for c in train.columns if not c.startswith('target')])
        if len(test) != orig_test_rows:
            test = test.drop_duplicates(subset=[c for c in test.columns if not c.startswith('target')])
    else:
        train = train.merge(aux_df, on=join_col, how='inner', suffixes=('', f'_{name.replace(".csv", "")}'))
        test = test.merge(aux_df, on=join_col, how='inner', suffixes=('', f'_{name.replace(".csv", "")}'))

# 3. Очистка
if 'target' in train.columns:
    train = train[train['target'].notna()]
train = train.drop_duplicates()
test = test.drop_duplicates()

# 4. Идентификация типов колонок
exclude_cols = ['target', 'client_id']
feature_cols = [c for c in train.columns if c not in exclude_cols]

cat_cols = []
num_cols = []

for col in feature_cols:
    if col not in test.columns:
        continue
    if train[col].dtype == 'object' or train[col].dtype == 'bool' or pd.api.types.is_categorical_dtype(train[col].dtype):
        cat_cols.append(col)
    elif pd.api.types.is_numeric_dtype(train[col].dtype):
        n_unique = train[col].nunique(dropna=True)
        if n_unique <= 10:
            # Проверим, похожи ли на категориальные
            cat_cols.append(col)
        else:
            num_cols.append(col)

# Пересмотр: булевы и очевидно категориальные числовые
for col in list(num_cols):
    if train[col].nunique(dropna=True) <= 10:
        num_cols.remove(col)
        if col not in cat_cols:
            cat_cols.append(col)

for col in list(cat_cols):
    if col in num_cols:
        num_cols.remove(col)

# 5. Числовые колонки - заполнение и винзоризация
num_medians = {}

for col in num_cols:
    median_val = train[col].median()
    num_medians[col] = median_val
    train[col] = train[col].fillna(median_val)
    if col in test.columns:
        test[col] = test[col].fillna(median_val)
    
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    
    train[col] = train[col].clip(lower=lower, upper=upper)
    if col in test.columns:
        test[col] = test[col].clip(lower=lower, upper=upper)

# 6. Категориальные колонки - заполнение
cat_fills = {}
for col in cat_cols:
    fill_val = "MISSING"
    cat_fills[col] = fill_val
    train[col] = train[col].fillna(fill_val).astype(str)
    if col in test.columns:
        test[col] = test[col].fillna(fill_val).astype(str)

# 7. Синтетические признаки

# а) Отношения между ключевыми числовыми парами
key_num_pairs = []
for i, c1 in enumerate(num_cols):
    for c2 in num_cols[i+1:]:
        ratio_name = f'ratio_{c1}_to_{c2}'
        train_ratio = np.where(train[c2] != 0, train[c1] / train[c2], 0)
        test_ratio = np.where(test[c2] != 0, test[c1] / (test[c2].replace(0, np.nan)), 0)
        train_ratio = np.nan_to_num(train_ratio, nan=0, posinf=0, neginf=0)
        test_ratio = np.nan_to_num(test_ratio, nan=0, posinf=0, neginf=0)
        train[ratio_name] = train_ratio
        test[ratio_name] = test_ratio
        if ratio_name not in num_cols:
            num_cols.append(ratio_name)

# б) Индикаторы пропусков (исходных)
original_columns = set([c.replace('ratio_', '').split('_to_')[0] for c in train.columns if c.startswith('ratio_')])
for col in feature_cols:
    if col in cat_cols or col in num_cols:
        miss_col = f'is_missing_{col}'
        if miss_col not in train.columns:
            train[miss_col] = 0
            test[miss_col] = 0

# Добавляем индикаторы для исходных пропущенных значений (сохраняем в отдельном словаре)
orig_missing_indicator = {}
for col in feature_cols:
    miss_col = f'was_missing_{col}'
    if col in train.columns:
        train[miss_col] = 0
        test[miss_col] = 0

# в) Target encoding с кросс-валидационной защитой
def target_encode_cv(df_train, df_test, cat_col, target_col, min_samples_leaf=10, smoothing=10, n_folds=5):
    global_mean = df_train[target_col].mean()
    encoded_train = np.zeros(len(df_train))
    encoded_test = np.zeros(len(df_test))
    
    # Для test: используем полный train
    stats = df_train.groupby(cat_col)[target_col].agg(['mean', 'count'])
    stats['smooth'] = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
    test_map = stats['smooth'].to_dict()
    encoded_test = df_test[cat_col].map(test_map).fillna(global_mean).values
    
    # Для train: CV
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(df_train):
        tr = df_train.iloc[train_idx]
        val = df_train.iloc[val_idx]
        
        stats_fold = tr.groupby(cat_col)[target_col].agg(['mean', 'count'])
        stats_fold['smooth'] = (stats_fold['count'] * stats_fold['mean'] + smoothing * global_mean) / (stats_fold['count'] + smoothing)
        
        fold_map = stats_fold['smooth'].to_dict()
        encoded_train[val_idx] = val[cat_col].map(fold_map).fillna(global_mean).values
    
    return encoded_train, encoded_test

if 'target' in train.columns:
    for col in cat_cols:
        if col in test.columns and train[col].nunique() > 1:
            enc_col = f'te_{col}'
            enc_train, enc_test = target_encode_cv(train, test, col, 'target')
            train[enc_col] = enc_train
            test[enc_col] = enc_test
            num_cols.append(enc_col)

# 8. Удаление константных колонок
all_cols = [c for c in train.columns if c not in ['target', 'client_id'] and c in test.columns]
for col in all_cols:
    if train[col].nunique() <= 1 and test[col].nunique() <= 1:
        train = train.drop(columns=[col])
        if col in test.columns:
            test = test.drop(columns=[col])
        if col in cat_cols:
            cat_cols.remove(col)
        if col in num_cols:
            num_cols.remove(col)

# 9. Синхронизация колонок
final_feature_cols = [c for c in train.columns if c not in ['target', 'client_id'] and c in test.columns]
final_feature_cols = sorted(final_feature_cols)  # для стабильности порядка
final_feature_cols_test = [c for c in final_feature_cols if c in test.columns]

# Определяем итоговые cat_cols
final_cat_cols = [c for c in cat_cols if c in final_feature_cols]

# Приводим test к тому же порядку колонок, что и train (по признакам)
train_features = train[final_feature_cols]
test_features = test[final_feature_cols]

# Добавляем target обратно
if 'target' in train.columns:
    train_out = pd.concat([train[['target']], train_features], axis=1)
else:
    train_out = train_features

if 'target' in test.columns:
    test_out = pd.concat([test[['target']], test_features], axis=1)
else:
    test_out = test_features

# 10. Сохранение
os.makedirs('output', exist_ok=True)

with open('output/cat_features.json', 'w') as f:
    json.dump(final_cat_cols, f)

train_out.to_csv('output/train.csv', index=False)
test_out.to_csv('output/test.csv', index=False)