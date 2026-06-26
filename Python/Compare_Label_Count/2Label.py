import os
import glob
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


RANDOM_SEED = 42


#########################################################
# LOAD DATA
#########################################################

def load_features(master_path):

    df = pd.read_excel(master_path)

    # remove non-numeric columns
    df = df.drop(
        columns=[
            "source_file",
            "prefix",
            "idx"
        ],
        errors="ignore"
    )

    y = df["label"].astype(int)

    X = df.drop(columns=["label"]).copy()


    # convert everything to numeric
    for c in X.columns:
        X[c] = pd.to_numeric(
            X[c],
            errors="coerce"
        )


    X = X.replace(
        [np.inf,-np.inf],
        np.nan
    )

    X = X.fillna(
        X.median(
            numeric_only=True
        )
    )

    return X.values, y.values


#########################################################
# SPLIT
#########################################################

def fixed_per_class_split(
        X,
        y,
        train_per_class=20,
        test_per_class=5,
        seed=42):

    rng=np.random.default_rng(seed)

    train=[]
    test=[]

    classes=np.unique(y)

    for c in classes:

        idx=np.where(y==c)[0]

        perm=rng.permutation(idx)

        train.extend(
            perm[:train_per_class]
        )

        test.extend(
            perm[
            train_per_class:
            train_per_class+test_per_class
            ]
        )

    return np.array(train),np.array(test)


#########################################################
# MODELS
#########################################################

def get_models():

    return {

        "LogReg":
        Pipeline([
            ("scaler",StandardScaler()),
            ("clf",LogisticRegression(max_iter=5000))
        ]),

        "SVM":
        Pipeline([
            ("scaler",StandardScaler()),
            ("clf",SVC(
                kernel="rbf",
                probability=True
            ))
        ]),

        "KNN":
        Pipeline([
            ("scaler",StandardScaler()),
            ("clf",KNeighborsClassifier())
        ]),

        "RF":
        RandomForestClassifier(),

        "ExtraTrees":
        ExtraTreesClassifier(),

        "NB":
        GaussianNB(),

        "MLP":
        Pipeline([
            ("scaler",StandardScaler()),
            ("clf",MLPClassifier(
                max_iter=2000
            ))
        ]),

        "Tree":
        DecisionTreeClassifier()
    }


#########################################################
# NORMAL 3-CLASS TRAINING
#########################################################

def run_normal(X,y):

    tr,te=fixed_per_class_split(
        X,y,
        train_per_class=20,
        test_per_class=5
    )

    Xtr,ytr=X[tr],y[tr]
    Xte,yte=X[te],y[te]

    print("\n")
    print("="*80)
    print("NORMAL TRAINING")
    print("="*80)

    for name,model in get_models().items():

        model.fit(Xtr,ytr)

        pred=model.predict(Xte)

        print("\n",name)

        print(
            classification_report(
                yte,
                pred
            )
        )


#########################################################
# TRAIN ONLY ON CLASS 0 & 2
#########################################################

def run_extreme_training(
        X,
        y,
        threshold=0.70):

    train_mask=(y==0)|(y==2)

    X02=X[train_mask]
    y02=y[train_mask]


    tr,_=fixed_per_class_split(
        X02,
        y02,
        train_per_class=20,
        test_per_class=0
    )

    Xtr=X02[tr]
    ytr=y02[tr]


    _,te=fixed_per_class_split(
        X,
        y,
        train_per_class=20,
        test_per_class=5
    )

    Xte=X[te]
    yte=y[te]


    print("\n")
    print("="*80)
    print("TRAIN ONLY ON 0 & 2")
    print("="*80)


    for name,model in get_models().items():

        if not hasattr(
            model,
            "predict_proba"
        ):
            continue


        model.fit(Xtr,ytr)

        probs=model.predict_proba(
            Xte
        )

        pred=[]


        for p in probs:

            conf=max(p)

            if conf < threshold:

                pred.append(1)

            else:

                pred.append(
                    model.classes_[
                    np.argmax(p)
                    ]
                )

        pred=np.array(pred)


        print("\n",name)

        print(
            classification_report(
                yte,
                pred
            )
        )

        print(
            confusion_matrix(
                yte,
                pred
            )
        )


#########################################################
# MAIN
#########################################################

if __name__=="__main__":

    MASTER_PATH = r"/Users/mohammad/University/Bachelor Project/Final/Data/Stand/Features/MASTER_Features_Stand.xlsx"

    X,y=load_features(
        MASTER_PATH
    )


    # Experiment 1
    run_normal(X,y)


    # Experiment 2
    run_extreme_training(
        X,
        y,
        threshold=0.70
    )