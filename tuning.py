"""Предварительный подбор гиперпараметров RF, GB и MLP (только train, CV 5x3)."""
import warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from ensemble_models import add_features, NOMINAL
warnings.filterwarnings("ignore")
df = pd.read_csv("german.csv", sep=";")
y = df["Creditability"].values; X = df.drop(columns="Creditability")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
fe = lambda: FunctionTransformer(add_features)
ohe = lambda: ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), NOMINAL)], remainder=StandardScaler())
print("MLP grid (train CV only)")
for h in [(8,), (16,), (32,), (16, 8)]:
    for a in [1.0, 3.0, 10.0]:
        m = make_pipeline(fe(), ohe(), MLPClassifier(h, alpha=a, max_iter=3000, random_state=42))
        print(h, a, round(cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc").mean(), 4), flush=True)
print("RF grid")
for leaf in [1, 3, 5]:
    for mf in ["sqrt", 0.3, 0.5]:
        m = make_pipeline(fe(), RandomForestClassifier(300, min_samples_leaf=leaf, max_features=mf, class_weight="balanced_subsample", random_state=42, n_jobs=-1))
        print(leaf, mf, round(cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc").mean(), 4), flush=True)
print("GB grid")
for d in [2, 3]:
    for lr_, n in [(0.03, 300), (0.05, 200), (0.01, 600)]:
        m = make_pipeline(fe(), GradientBoostingClassifier(n_estimators=n, learning_rate=lr_, max_depth=d, subsample=0.8, random_state=42))
        print(d, lr_, n, round(cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc").mean(), 4), flush=True)
