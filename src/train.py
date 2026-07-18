import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def find_best_k(X_train, y_train, k_range=range(1, 21)):
    """K-sweep specifically for KNN."""
    scores = []
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        scores.append(cv_scores.mean())
    best_k = list(k_range)[int(np.argmax(scores))]
    return best_k, scores


def train_knn(X_train, y_train, n_neighbors: int = 5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, max_depth: int = 5):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model
