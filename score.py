import os
from dataclasses import dataclass, field

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

CATBOOST_PARAMS = {
    "iterations": 100,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_seed": 42,
    "verbose": 0,
    "thread_count": 1,
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
}


@dataclass
class ScoringResult:
    roc_auc: float = 0.0
    gini: float = 0.0
    primary_score: float = 0.0
    details: dict = field(default_factory=dict)


class ScoringEngine:
    def __init__(self):
        self.target_column = "target"
        self.id_column = "client_id"
        self.metric = "roc_auc"

    def score(self, output_dir: str = "output/") -> ScoringResult:
        train_df = pd.read_csv(os.path.join(output_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(output_dir, "test.csv"))

        feature_cols = [
            c for c in train_df.columns if c not in (self.id_column, self.target_column)
        ]

        X_train = train_df[feature_cols].copy()
        y_train = train_df[self.target_column].copy()
        X_test = test_df[feature_cols].copy()

        X_train = X_train.fillna(-999)
        X_test = X_test.fillna(-999)

        cat_features = []
        for i, col in enumerate(X_train.columns):
            if X_train[col].dtype == "str":
                X_train[col] = X_train[col].astype(str)
                X_test[col] = X_test[col].astype(str)
                cat_features.append(i)

        hidden_labels = test_df[self.target_column]

        fit_params = {"cat_features": cat_features} if cat_features else {}
        try:
            model = CatBoostClassifier(**CATBOOST_PARAMS)
            model.fit(X_train, y_train, **fit_params)
        except Exception as e:
            print("Ошибка c данными")
            result = ScoringResult(
                roc_auc=0,
                gini=0,
                primary_score=0,
                details={
                    "n_features": 0,
                    "train_rows": 0,
                    "test_rows": 0,
                    "top_features": 0,
                },
            )
            return result
        
        
        test_probas = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(hidden_labels, test_probas)
        gini = 2 * roc_auc - 1

        feature_importance = dict(
            zip(feature_cols, model.get_feature_importance().tolist(), strict=False)
        )
        top_features = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
        )

        primary_score = roc_auc if self.metric == "roc_auc" else gini

        result = ScoringResult(
            roc_auc=round(roc_auc, 6),
            gini=round(gini, 6),
            primary_score=round(primary_score, 6),
            details={
                "n_features": len(feature_cols),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "top_features": top_features,
            },
        )

        return result

if __name__ == "__main__":
    score = ScoringEngine()
    res = score.score()
    print(res)