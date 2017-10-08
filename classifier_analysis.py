import numpy as np
import multiprocessing
from transformers import FilterSimu

n_jobs = multiprocessing.cpu_count()

def classifier_analysis(X, label, methodType):
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import ShuffleSplit
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    #rng = None
    rng = np.random.RandomState(1)

    if methodType == 0:
        # random forest
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2,
                                            min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                            max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                            bootstrap=True, oob_score=False, n_jobs=n_jobs, random_state=rng, verbose=0,
                                            warm_start=False, class_weight=None)

        param_grid = {
            'filter__threshold': [0.95, 0.97, 0.99],
            'classifier__n_estimators': [5, 10, 20],
            'classifier__max_depth': [None, 10, 5, 3],
            'classifier__max_features': ['auto', 10, 5]
        }
    elif methodType == 1:
        # adaboost
        from sklearn.ensemble import AdaBoostClassifier
        classifier = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=rng)
        param_grid = {
            'filter__threshold': [0.95, 0.97, 0.99],
            'classifier__n_estimators': [5, 10, 20],
            'classifier__learning_rate': [0.8, 0.9, 1.0]
        }
    elif methodType == 2:
        # GBC
        from sklearn.ensemble import GradientBoostingClassifier
        classifier = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                                criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
                                                min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
                                                min_impurity_split=None, init=None, random_state=rng, max_features=None,
                                                verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
        param_grid = {
            'filter__threshold': [0.95, 0.97, 0.99],
            'classifier__n_estimators': [50, 100, 150],
            'classifier__max_depth': [None, 10, 5, 3],
            'classifier__learning_rate': [0.8, 0.9, 1.0]
        }
    elif methodType == 3:
        # logtistic regression
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                                        intercept_scaling=1, class_weight=None, random_state=rng, solver='saga',
                                        max_iter=100, multi_class='multinomial', verbose=0, warm_start=False, n_jobs=n_jobs)
        param_grid = {
            'filter__threshold': [0.95, 0.97, 0.99],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': [0.9, 1.0, 1.1]
        }
    elif methodType == 4:
        # SVM
        from sklearn.svm import SVC
        classifier = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,
                         tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                         decision_function_shape='ovr', random_state=rng)
        param_grid = {
            'filter__threshold': [0.95, 0.97, 0.99],
            'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'classifier__C': [0.9, 1.0, 1.1]
        }
    elif methodType == 5:
        # MLP
        from sklearn.neural_network import MLPClassifier
        classifier = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001,
                                   batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5,
                                   max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False,
                                   warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
                                   validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        param_grid = {
            'filter__threshold': [0.95, 0.97, 0.99],
            'classifier__hidden_layer_sizes': [(100, ), (50, ), (20, )],
            'classifier__learning_rate_init': [0.0001, 0.001, 0.01]
        }
    elif methodType == 6:
        # linear SVM
        from sklearn.svm import LinearSVC
        classifier = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, C=1.0, multi_class='ovr',
                               fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=rng,
                               max_iter=1000)
        param_grid = {
            'filter__threshold': [0.95, 0.97, 0.99],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': [0.9, 1.0, 1.1]
        }
    elif methodType == 7:
        # Bernoulli Naive Bayes
        from sklearn.naive_bayes import BernoulliNB
        classifier = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
        param_grid = {
            'filter__threshold': [0.95, 0.97, 0.99],
            'classifier__alpha': [0.90, 0.95, 1.0],
            'classifier__fit_prior': [True, False]
        }
    elif methodType == 8:
        # multinomial Naive Bayes
        from sklearn.naive_bayes import MultinomialNB
        classifier = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
        param_grid = {
            'classifier__alpha': [0.90, 0.95, 1.0],
            'classifier__fit_prior': [True, False]
        }
    else:
        return

    if methodType == 8:
        pipe = Pipeline([
            ('classifier', classifier)
        ])
    else:
        pipe = Pipeline([
            ('scale', StandardScaler()),
            ('filter', FilterSimu()),
            ('classifier', classifier)
        ])


    grid = GridSearchCV(pipe, cv=ShuffleSplit(n_splits=4, test_size=0.25, random_state=rng), n_jobs=1, param_grid=param_grid)
    grid.fit(X, label)
    best_estimator = grid.best_estimator_

    #mean_scores = np.array(grid.cv_results_['mean_test_score'])
    #mean_tscores = np.array(grid.cv_results_['mean_train_score'])
    #print mean_scores
    #print mean_tscores

    print grid.best_params_
    score = grid.best_score_
    #print grid.cv_results_['params']

    return best_estimator, grid.predict(X), score