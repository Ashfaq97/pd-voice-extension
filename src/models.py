"""Model definitions and hyper-parameter grids, shared by every phase.

Kept in one place so that Phase 2 (regime comparison), Phase 3 (recording
type) and Phase 5 (interpretability) cannot silently disagree about what
"logistic regression" or "XGBoost" means.

Every estimator is wrapped in a Pipeline whose first step is StandardScaler,
so scaling is fit per fold and never on full data. Grids are addressed with
the `clf__` prefix accordingly.
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from config import SEED

# Display order, chosen to match the tables in the paper.
MODEL_ORDER = [
    "Logistic Regression",
    "KNN",
    "Decision Tree",
    "Gradient Boosting",
    "XGBoost",
]


def _pipe(clf):
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def get_models():
    """Return {name: (pipeline, param_grid)}.

    Grid sizes are deliberately modest (5-12 points each, 43 fits per inner
    CV across all five models). The paper's claim is about the gap between
    validation regimes, not about squeezing the last point of accuracy out of
    any one model, and the grids are identical across regimes so the contrast
    stays like-for-like.
    """
    return {
        "Logistic Regression": (
            _pipe(LogisticRegression(max_iter=5000, random_state=SEED)),
            {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        "KNN": (
            _pipe(KNeighborsClassifier()),
            {
                "clf__n_neighbors": [3, 5, 7, 9, 11],
                "clf__weights": ["uniform", "distance"],
            },
        ),
        "Decision Tree": (
            _pipe(DecisionTreeClassifier(random_state=SEED)),
            {
                "clf__max_depth": [3, 5, 7, None],
                "clf__min_samples_leaf": [1, 5, 10],
            },
        ),
        "Gradient Boosting": (
            _pipe(GradientBoostingClassifier(random_state=SEED)),
            {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [2, 3],
            },
        ),
        "XGBoost": (
            _pipe(
                XGBClassifier(
                    random_state=SEED,
                    eval_metric="logloss",
                    verbosity=0,
                    n_jobs=1,
                )
            ),
            {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [3, 5],
            },
        ),
    }


# Fixed hyper-parameters for phases where tuning inside every fold is too
# costly to be worth it (Phase 3 runs 4 recording types x 2 regimes x 5
# models). These are the library defaults except where a grid above has a
# clearly preferred region; using fixed values keeps the per-type numbers
# comparable to each other, which is what that analysis is about.
def get_fixed_models():
    return {
        "Logistic Regression": _pipe(
            LogisticRegression(C=1.0, max_iter=5000, random_state=SEED)
        ),
        "KNN": _pipe(KNeighborsClassifier(n_neighbors=5, weights="uniform")),
        "Decision Tree": _pipe(
            DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=SEED)
        ),
        "Gradient Boosting": _pipe(
            GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=3, random_state=SEED
            )
        ),
        "XGBoost": _pipe(
            XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=SEED,
                eval_metric="logloss",
                verbosity=0,
                n_jobs=1,
            )
        ),
    }
