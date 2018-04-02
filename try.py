import tensorflow as tf
import numpy as np
import pandas as pd


def train_data_baseline(file_path):
    try:

        dataframe = pd.read_table(file_path, sep="\t")
        print("Baseline Training data read!!!")
        return dataframe[["SentenceId", "Phrase", "Sentiment"]].drop_duplicates("SentenceId")

    except Exception as e:

        print("Error in reading dataframe")
        print(e)


def test_data_baseline(file_path):
    try:

        dataframe = pd.read_table(file_path, sep="\t")
        print("Baseline test data read")
        return dataframe[["SentenceId", "Phrase"]].drop_duplicates("SentenceId")

    except Exception as e:

        print("Error in reading test baseline dataframe")
        print(e)


def training_baseline_models(train_path, test_path, model_name, train_param, test_param):
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.svm import SVC
    from nltk.classify import MaxentClassifier
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.metrics import accuracy_score

    train_df, test_df = train_data_baseline(train_path), test_data_baseline(test_path)

    train_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)

    print(train_df.shape)
    print(test_df.shape)

    X_data, y_data = train_df.values[:, 1], train_df.values[:, 2].astype('int')

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1)

    print(X_train.shape, y_train.shape)

    ##### Training on baseline models 1. SVM  2. MaxEnt classifiers (Both perform better than Standard Naive Bayes for sentiment classification over large corpora)#####

    #           1. SVM        #

    if model_name in ["SVM", "svm"]:

        if train_param is True:

            svm_param_grid = {"kernel": ['rbf', 'poly', 'linear'],
                              "C": [50, 40, 30, 20, 10, 5, 1],
                              'gamma': [0.001, 0.005, 0.01, 0.1]}

            svm_grid = GridSearchCV(SVC(class_weight='balanced', probability=True,max_iter=2500), param_grid=svm_param_grid,
                                    n_jobs=-1, verbose=1)

            # svm_pipeline = Pipeline([("count_vectorizer",CountVectorizer(ngram_range=(1,3))),("SVM",svm_grid)])

            count_vec = CountVectorizer(max_df=0.5, ngram_range=(1, 3))
            train_data = count_vec.fit_transform(X_train)

            print("Starting SVM Training")

            svm_grid.fit(train_data, y_train)

            print("Training Over")
            print("BEST PARAMETERS : ")
            print(svm_grid.best_params_)

            if test_param is True:
                pred = svm_grid.predict(count_vec.transform(X_test))
                print(accuracy_score(y_test, pred))


        else:

            return "Please mention either train or test param as True"


training_baseline_models("Sentiment_Analysis_Challenge/train.tsv", "Sentiment_Analysis_Challenge/test.tsv",
                         model_name="svm", train_param=True, test_param=True)