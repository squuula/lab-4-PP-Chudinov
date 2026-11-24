"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from grader_contracts.ensembles import EnsembleInput, EnsembleResult

TARGET = "Creditability"

# Признаки german.csv, которые закодированы числами, но по смыслу являются
# категориями (номинальными или "ступенчатыми"). Для MLP их кодируем one-hot.
NOMINAL = [
    "Account_Balance",
    "Payment_Status_of_Previous_Credit",
    "Purpose",
    "Value_Savings_Stocks",
    "Length_of_current_employment",
    "Sex_Marital_Status",
    "Guarantors",
    "Most_valuable_available_asset",
    "Concurrent_Credits",
    "Type_of_apartment",
    "Occupation",
]
ENGINEERED_BASE = ["Credit_Amount", "Duration_of_Credit_monthly", "Instalment_per_cent", "Account_Balance"]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Добавляет производные признаки. Использует только строку объекта, без target."""
    X = X.copy()
    X["log_amount"] = np.log1p(X["Credit_Amount"])
    X["monthly_payment"] = X["Credit_Amount"] / X["Duration_of_Credit_monthly"].clip(lower=1)
    X["log_monthly_payment"] = np.log1p(X["monthly_payment"])
    X["burden"] = X["Instalment_per_cent"] * X["monthly_payment"]
    X["no_account"] = (X["Account_Balance"] == 4).astype(int)
    return X


def _to_frame(features) -> pd.DataFrame:
    X = features.copy() if isinstance(features, pd.DataFrame) else pd.DataFrame(np.asarray(features))
    if TARGET in X.columns:  # защита: целевая переменная не должна попасть в признаки
        X = X.drop(columns=TARGET)
    return X


def build_model(columns, seed: int = 42) -> VotingClassifier:
    """Soft Voting: Random Forest + Gradient Boosting + полносвязная MLP."""
    known = all(c in columns for c in ENGINEERED_BASE)
    fe = [FunctionTransformer(add_features)] if known else []
    nominal = [c for c in NOMINAL if c in columns]

    # Деревьям масштабирование и one-hot не нужны.
    rf = make_pipeline(
        *fe,
        RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
    )
    gb = make_pipeline(
        *fe,
        GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            random_state=seed,
        ),
    )
    # Нейросети нужны one-hot для категорий и стандартизация числовых признаков.
    mlp_pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), nominal)],
        remainder=StandardScaler(),
    )
    mlp = make_pipeline(
        *fe,
        mlp_pre,
        MLPClassifier(
            hidden_layer_sizes=(16,),
            alpha=3.0,  # сильная L2-регуляризация: данных всего ~700 строк
            max_iter=3000,
            random_state=seed,
        ),
    )
    return VotingClassifier([("rf", rf), ("gb", gb), ("mlp", mlp)], voting="soft")


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучите выбранную модель и верните вероятности, метки и метрики на test.

    Разрешены Random Forest, Extra Trees, AdaBoost, Gradient Boosting,
    HistGradientBoosting, Voting/Stacking или полносвязная MLP. Не используйте
    test_target при обучении модели или подборе её параметров.
    """
    X_train = _to_frame(data.train_features)
    X_test = _to_frame(data.test_features)
    y_train = np.asarray(data.train_target).ravel()
    y_test = np.asarray(data.test_target).ravel()

    model = build_model(list(X_train.columns), seed=data.random_seed)
    model.fit(X_train, y_train)  # обучение только на train

    classes = list(model.classes_)
    pos_label = 1 if 1 in classes else classes[-1]
    pos_idx = classes.index(pos_label)
    neg_label = classes[1 - pos_idx]

    probabilities = model.predict_proba(X_test)[:, pos_idx]
    predictions = np.where(probabilities >= 0.5, pos_label, neg_label)

    # test_target используется только здесь — для подсчёта итоговых метрик.
    return EnsembleResult(
        model_name="Soft Voting (RandomForest + GradientBoosting + MLP)",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(y_test == pos_label, probabilities)),
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, pos_label=pos_label, zero_division=0)),
        recall=float(recall_score(y_test, predictions, pos_label=pos_label, zero_division=0)),
        f1=float(f1_score(y_test, predictions, pos_label=pos_label, zero_division=0)),
    )
