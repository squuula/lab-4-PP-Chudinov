"""Воспроизводимый прогон всех экспериментов лабораторной №4.

Запуск из корня репозитория:  python experiments.py
Результат: таблица в консоли и файл experiment_results.json.

Схема: german.csv -> stratified train/test 70/30 (random_state=42).
Модели сравниваются по ROC-AUC на 5-fold x 3 RepeatedStratifiedKFold
ТОЛЬКО на train. Метрики на test считаются для отчёта и не участвуют в выборе.
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ensemble_models import NOMINAL, TARGET, add_features, build_model, run_ensemble_experiment
from grader_contracts.ensembles import EnsembleInput

warnings.filterwarnings("ignore", category=UserWarning)
SEED = 42

df = pd.read_csv("german.csv", sep=";")
X = df.drop(columns=TARGET)
y = df[TARGET].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)

fe = lambda: FunctionTransformer(add_features)
ohe = lambda scale=True: ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), NOMINAL)],
    remainder=StandardScaler() if scale else "passthrough")

final = build_model(list(X.columns), SEED)
rf_f, gb_f, mlp_f = (clone(e) for _, e in final.estimators)

experiments = {
    "E0 LogisticRegression (baseline, не ансамбль)":
        make_pipeline(fe(), ohe(), LogisticRegression(C=0.3, max_iter=2000)),
    "E1 RandomForest, исходные признаки, по умолчанию":
        RandomForestClassifier(500, random_state=SEED, n_jobs=-1),
    "E2 RandomForest + признаки, leaf=3, sqrt, balanced": rf_f,
    "E3 ExtraTrees + one-hot, leaf=3, max_features=0.3":
        make_pipeline(fe(), ohe(False), ExtraTreesClassifier(500, min_samples_leaf=3, max_features=0.3,
                                                             random_state=SEED, n_jobs=-1)),
    "E4 AdaBoost (300, lr=0.05)":
        make_pipeline(fe(), AdaBoostClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED)),
    "E5 GradientBoosting (300, lr=0.03, depth=3, subsample=0.8)": gb_f,
    "E6 HistGradientBoosting (300, lr=0.03, depth=3, l2=1)":
        make_pipeline(fe(), HistGradientBoostingClassifier(learning_rate=0.03, max_iter=300, max_depth=3,
                                                           l2_regularization=1.0, random_state=SEED)),
    "E7 MLP (32), alpha=1": make_pipeline(fe(), ohe(), MLPClassifier((32,), alpha=1.0, max_iter=3000,
                                                                      random_state=SEED)),
    "E8 MLP (16), alpha=3": mlp_f,
    "E9 Soft Voting RF+GB": VotingClassifier([("rf", clone(rf_f)), ("gb", clone(gb_f))], voting="soft"),
    "E10 Stacking RF+GB+MLP -> LogReg": StackingClassifier(
        [("rf", clone(rf_f)), ("gb", clone(gb_f)), ("mlp", clone(mlp_f))],
        final_estimator=LogisticRegression(), cv=StratifiedKFold(5, shuffle=True, random_state=SEED)),
    "E11 Soft Voting RF+GB+MLP (ВЫБРАНА)": final,
}


def test_metrics(y_true, proba):
    pred = (proba >= 0.5).astype(int)
    return dict(roc_auc=roc_auc_score(y_true, proba), accuracy=accuracy_score(y_true, pred),
                precision=precision_score(y_true, pred), recall=recall_score(y_true, pred),
                f1=f1_score(y_true, pred))


rows = []
for name, model in experiments.items():
    t0 = time.time()
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc")
    model.fit(X_tr, y_tr)
    m = test_metrics(y_te, model.predict_proba(X_te)[:, 1])
    rows.append(dict(experiment=name, cv_roc_auc_mean=cv_scores.mean(), cv_roc_auc_std=cv_scores.std(),
                     **{f"test_{k}": v for k, v in m.items()}, seconds=time.time() - t0))
    print(f"{name:60s} CV={cv_scores.mean():.4f}±{cv_scores.std():.3f} "
          f"test AUC={m['roc_auc']:.4f} acc={m['accuracy']:.3f} F1={m['f1']:.3f}", flush=True)

# Финальный запуск через контрактную функцию (тот же split).
t0 = time.time()
res = run_ensemble_experiment(EnsembleInput(X_tr, y_tr, X_te, y_te, random_seed=SEED))
final_seconds = time.time() - t0
print("\nКонтрактная функция:", res.model_name, f"ROC-AUC={res.roc_auc:.4f}", f"{final_seconds:.1f}s")

# Проверка устойчивости выбранной модели на 10 других случайных разбиениях 70/30
# (делается ПОСЛЕ выбора модели, на выбор не влияет).
robust = []
for s in range(10):
    a, b, c, d = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1000 + s)
    robust.append(run_ensemble_experiment(EnsembleInput(a, c, b, d, random_seed=SEED)).roc_auc)
print(f"Устойчивость (10 разбиений): ROC-AUC {np.mean(robust):.4f} ± {np.std(robust):.4f}, "
      f"min={np.min(robust):.4f}, max={np.max(robust):.4f}")

out = {
    "dataset": "german.csv",
    "split": {"test_size": 0.3, "stratify": True, "random_state": SEED,
              "n_train": int(len(y_tr)), "n_test": int(len(y_te))},
    "positive_class": 1,
    "selection_criterion": "mean ROC-AUC on RepeatedStratifiedKFold(5x3) on train only",
    "final_model": {
        "model_name": res.model_name,
        "roc_auc": round(res.roc_auc, 4), "accuracy": round(res.accuracy, 4),
        "precision": round(res.precision, 4), "recall": round(res.recall, 4), "f1": round(res.f1, 4),
        "threshold": 0.5, "fit_predict_seconds": round(final_seconds, 1),
    },
    "robustness_10_random_splits": {"roc_auc_mean": round(float(np.mean(robust)), 4),
                                    "roc_auc_std": round(float(np.std(robust)), 4),
                                    "roc_auc_min": round(float(np.min(robust)), 4),
                                    "roc_auc_max": round(float(np.max(robust)), 4)},
    "experiments": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows],
}
with open("experiment_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("Сохранено в experiment_results.json")
