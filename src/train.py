from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def tune_knn(X_train, y_train):
    param_grid = {
        "n_neighbors": range(1, 21),
        "weights": ["uniform", "distance"],
        "p": [1, 2],  # 1 = Manhattan distance, 2 = Euclidean distance
    }
    grid = GridSearchCV(
        KNeighborsClassifier(), param_grid,
        cv=5, scoring="f1_weighted", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print(f"[KNN] Best params: {grid.best_params_} | Best CV F1: {grid.best_score_:.3f}")
    return grid.best_estimator_


def tune_decision_tree(X_train, y_train):
    param_grid = {
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"],
    }
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42), param_grid,
        cv=5, scoring="f1_weighted", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print(f"[Decision Tree] Best params: {grid.best_params_} | Best CV F1: {grid.best_score_:.3f}")
    return grid.best_estimator_


def tune_logistic_regression(X_train, y_train):
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    }
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42), param_grid,
        cv=5, scoring="f1_weighted", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print(f"[Logistic Regression] Best params: {grid.best_params_} | Best CV F1: {grid.best_score_:.3f}")
    return grid.best_estimator_


def tune_random_forest(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42), param_grid,
        cv=5, scoring="f1_weighted", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print(f"[Random Forest ] Best params: {grid.best_params_} | Best CV F1: {grid.best_score_:.3f}")
    return grid.best_estimator_