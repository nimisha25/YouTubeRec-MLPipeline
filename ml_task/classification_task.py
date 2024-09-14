import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold,cross_validate, GridSearchCV
from sklearn import linear_model
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier


def run_logistic_regression(X_train, y_train) :
    lm = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear',C=0.1, penalty='l2', class_weight='balanced')
    lm.fit(X_train, y_train)
    return lm

def run_random_forest(X_train, y_train) :
    classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42, class_weight='balanced')
    classifier.fit(X_train, y_train)
    return classifier

def run_svm(X_train, y_train) :
    rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1, probability=True, class_weight='balanced')
    rbf.fit(X_train, y_train)
    return rbf

# def run_xgboost(X_train, y_train) :
#     xgb_clf = XGBClassifier(objective='multi:softmax', 
#                             num_class=3, 
#                             missing=1, 
#                             eval_metric=['merror','mlogloss'], 
#                             seed=42)
#     xgb_clf.fit(X_train,y_train, early_stopping_rounds=10)
#     return xgb_clf

def evaluate_model(model, X, y) :
    # Define the metrics you want to track
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_ovr']
    
    # Use cross_validate to compute all metrics at once
    scores = cross_validate(model, X, y, cv=kf, scoring=scoring)
    
    # Calculate the mean for each metric
    metrics = {
        'accuracy': scores['test_accuracy'].mean(),
        'precision': scores['test_precision_weighted'].mean(),
        'recall': scores['test_recall_weighted'].mean(),
        'f1_score': scores['test_f1_weighted'].mean(),
        'roc_auc': scores['test_roc_auc_ovr'].mean()
    }
    
    return metrics


def compare_models(X_train, y_train) :
    models = {
        'Logistic Regression': run_logistic_regression(X_train, y_train),
        'Random Forest': run_random_forest(X_train, y_train),
        'SVM': run_svm(X_train, y_train),
        # 'XGBoost': run_xgboost(X_train, y_train)
    }
    comparison_metrics = {}

    for model_name, model in models.items() :
        print(f"Evaluating {model_name}...")
        metrics = evaluate_model(model, X_train, y_train)
        comparison_metrics[model_name] = metrics

    return comparison_metrics

def tune_hyperparameters(X_train, y_train) :
    param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
    }

    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                            param_grid, cv=5, verbose=3)
    rf_grid.fit(X_train, y_train)
    return rf_grid

if __name__ == '__main__' :
    df = pd.read_csv('data/clustered_dataset.csv')
    # print("columns: " , df.columns)
    y = df['cluster']
    X = df.drop(columns = ['cluster', 'cluster_label'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y , 
                                    random_state=104,  
                                    test_size=0.25,  
                                    shuffle=True)
    
    # comparison_metrics = compare_models(X_train, y_train)
    # print(comparison_metrics)
    # svm = run_svm(X_train, y_train)
    # metrics_test = evaluate_model(svm, X_test, y_test)
    # print("SVM Test Metrics:", metrics_test)
    # rf_grid = tune_hyperparameters(X_train, y_train)
    # print(rf_grid.best_params_)
    # print(y_train.value_counts())
    # print(y_test.value_counts())


    
