import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import random
from sklearn.metrics import accuracy_score

n_datapoints = 15000

records = []

for i in range(n_datapoints):
    n_samples = random.randint(100, 10000)
    n_features = random.randint(5, 20)
    n_informative = random.randint(2, min(10, n_features - 1))
    class_sep = random.uniform(0.5, 2.0)
    flip_y = random.uniform(0.0, 0.15)
    max_classes = min(2**n_informative, n_samples // 10)
    n_classes = random.randint(2, max(2, max_classes))


    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=flip_y,
        n_classes=n_classes,
        class_sep=class_sep,
        random_state=i
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

    max_depth = random.choice([None, 2, 3, 5, 8, 10, 15, 20])
    min_samples_split = random.randint(2, 10)
    min_samples_leaf = random.randint(1, 5)
    criterion = random.choice(["gini", "entropy", "log_loss"])

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=i
    )
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    acc_gap = abs(train_acc - test_acc)

    if train_acc <= 0.75 and test_acc <= 0.75:
        label = "underfit"
    elif acc_gap > 0.10:
        label = "overfit" if train_acc > test_acc else "abnormalfit"            
    else:
        label = "normalfit"

    records.append({
        "max_depth": -1 if max_depth is None else max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "criterion": criterion,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_informative": n_informative,
        "class_sep": round(class_sep, 3),
        "n_classes": n_classes,
        "label": label
    })

df = pd.DataFrame(records)
df.to_csv("decision_tree_fit_prediction_dataset.csv", index=False)

print(f"Dataset samples: {len(df)}\n")
print(df.head(10))
print("\nLabel distribution:\n", df['label'].value_counts())
