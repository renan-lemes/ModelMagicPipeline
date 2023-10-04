import dataset

# ---------- Libs sklearn  pipes ----------------

from sklearn.pipeline import Pipeline, make_pipeline  # carregar as pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
)  # transformar as class do df como numerica
from sklearn.compose import (
    ColumnTransformer,
)  # transformacoes especifica das colunas formatando os dados
from sklearn.impute import KNNImputer, SimpleImputer  # KNN para inputar dados faltantes
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    GridSearchCV,
)

## ------------ Libs sklearn models -------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier


def models_class():
    train, test = dataset.LoadData()

    data_set = dataset.DataStruct(train, test).data_cat__num()

    categoric_features, numeric_features = (
        data_set.categoric_features,
        data_set.numeric_features,
    )

    def Transforms(categoric_features, numeric_features):
        numeric_transform = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]
        )

        ## Aqui to criando os transformadores não numericos
        catecoric_transform = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        ## Aqui to unindo todos com
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transform, numeric_features),
                ("cat", catecoric_transform, categoric_features),
            ]
        )
        return preprocessor

    preprocessor = Transforms(categoric_features, numeric_features)

    def models_classifiers_pipes(preprocessor):
        rf_pipeline = Pipeline(
            [("preprocessor", preprocessor), ("classifier", RandomForestClassifier())]
        )

        knn_pipeline = Pipeline(
            [("preprocessor", preprocessor), ("classifier", KNeighborsClassifier())]
        )

        lr_pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        )

        dt_pipeline = Pipeline(
            [("preprocessor", preprocessor), ("classifier", DecisionTreeClassifier())]
        )

        classifiers = [
            ("RandomForest", rf_pipeline),
            ("KNeighbors", knn_pipeline),
            ("LogisticRegression", lr_pipeline),
            ("DecisionTree", dt_pipeline),
        ]

        voting_classifier = VotingClassifier(estimators=classifiers)
        return voting_classifier, classifiers

    voting_classifier, classifiers = models_classifiers_pipes(
        preprocessor
    )  ## Aqui esta todos os modelos salvos

    y = data_set.train["Survived"]
    X = data_set.train.drop("Survived", axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    validation_cruz = KFold(n_splits=10, shuffle=True)

    for name, clf in classifiers:
        scores = cross_val_score(
            clf, X_train, y_train, cv=validation_cruz
        )  # Use validação cruzada k-fold (cv=5 como exemplo)
        print(f"{name}: Scores de Validação Cruzada: {scores}")
        print(f"{name}: Média dos Scores de Validação Cruzada: {scores.mean()}")

    def Save_models(classifiers):
        import joblib
        import json
        import pickle

        model1 = classifiers.rf_pipeline
        model2 = classifiers.dt_pipeline

        joblib.dump(model1, "..\\model\\RandomForest.pkl")
        joblib.dump(model1, "..\\model\\DecisionTree.pkl")

    Save_models(classifiers)
