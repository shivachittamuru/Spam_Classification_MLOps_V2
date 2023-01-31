
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, json, sys
import joblib
import mlflow
import argparse

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# define functions
def main(args):
    
    # Start Logging
    mlflow.start_run()
    
    # enable auto logging
    #mlflow.autolog()

    # read in data
    df = pd.read_csv(args.spam_csv)
    
    X_train, X_test, y_train, y_test = process_data(df, args.random_state)    

    # Print number of comments for each set
    print(f"There are {X_train.shape[0]} comments for training.")
    print(f"There are {X_test.shape[0]} comments for testing")

    # Allow unigrams and bigrams
    vectorizer = CountVectorizer(ngram_range=(1, 5))

    # Encode train text
    X_train_vect = vectorizer.fit_transform(X_train.text.tolist())

    # Fit model
    clf=MultinomialNB()
    clf.fit(X=X_train_vect, y=y_train)

    # Vectorize test text
    X_test_vect = vectorizer.transform(X_test.text.tolist())

    # Make predictions for the test set
    preds = clf.predict(X_test_vect)

    # Return accuracy score
    true_acc = accuracy_score(preds, y_test)
    true_acc

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=preds)

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

    precision = precision_score(y_test, preds)
    print('Precision: %.3f' % precision)

    recall = recall_score(y_test, preds)
    print('Recall: %.3f' % recall)

    f1 = f1_score(y_test, preds)
    print('f1: %.3f' % f1)
    
    
    mlflow.log_metric("Accuracy", true_acc)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1", f1)
    


    
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
    mlflow.sklearn.save_model(clf, "../../model")

    #### VECTORIZER
    # Registering the vectorizer to the workspace
    print("Registering the vectorizer via MLFlow")

    mlflow.sklearn.log_model(
        sk_model=vectorizer,
        registered_model_name=args.registered_vec_name,
        artifact_path=args.registered_vec_name,
    )
    
    # Saving the vectorizer to a file
    mlflow.sklearn.save_model(vectorizer, "../../model")

    
    # os.makedirs('./outputs', exist_ok=True)
    # with open(args.registered_model_name, 'wb') as file:
    #     joblib.dump(value=clf, filename='outputs/' + args.registered_model_name)    

    # with open(args.registered_vec_name, 'wb') as file:
    #     joblib.dump(value=vectorizer, filename='outputs/' + args.registered_vec_name)
        
    # Stop Logging
    mlflow.end_run()
    
def process_data(df, random_state):
    X = df.drop(["label"], axis=1)
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test

# run script
if __name__ == "__main__":
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--spam-csv", type=str)
    parser.add_argument("--registered_model_name", type=str, help="model name")
    parser.add_argument("--registered_vec_name", type=str, help="vectorizer name")
    parser.add_argument("--random_state", type=int, default=42)

    # parse args
    args = parser.parse_args()
    
    # run main function
    main(args)