
import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import os, json, sys
import joblib
import mlflow
import argparse

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# define functions
def main(args):
           
    # enable auto logging
    mlflow.autolog()

    # read in data
    df = pd.read_csv(args.spam_csv)
    df['text'] = df['text'].apply(lambda text: re.sub('[^A-Za-z]+', ' ', text.lower()))
    
    X = df[['text']]
    y = df[['label']]

    # Use 1/5 of the data for testing later
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state, stratify=y)

    # Print number of comments for each set
    print(f"There are {X_train.shape[0]} comments for training.")
    print(f"There are {X_test.shape[0]} comments for testing")

    clf = make_pipeline(
        #TfidfVectorizer(stop_words=get_stop_words('en')),
        TfidfVectorizer(),
        SVC(kernel='linear', probability=True)
    )

    clf = clf.fit(X=X_train['text'], y=y_train['label'])
    
    # Make predictions for the test set
    y_pred = clf.predict(X_test['text'])

    # Return accuracy score
    true_acc = accuracy_score(y_pred, y_test)
    
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

    # Print the confusion matrix using Matplotlib
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()

    plt.savefig("confusion_matrix.png")

    print('Accuracy: %.3f' % true_acc)

    precision = precision_score(y_test, y_pred)
    print('Precision: %.3f' % precision)

    recall = recall_score(y_test, y_pred)
    print('Recall: %.3f' % recall)

    f1 = f1_score(y_test, y_pred)
    print('f1: %.3f' % f1)

    #### MODEL
    # Registering the model to the workspace
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=clf,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name,
    )
    
    # Saving the model to a file
    print("Saving the model via MLFlow")
    mlflow.sklearn.save_model(
        sk_model=clf,
        path=os.path.join(args.registered_model_name, "trained_model"),
    )
        
# run script
if __name__ == "__main__":
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--spam-csv", type=str)
    parser.add_argument("--registered_model_name", type=str, help="model name")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, required=False, default=0.20)

    # parse args
    args = parser.parse_args()
    
    # run main function
    main(args)
