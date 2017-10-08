import numpy as np


from sklearn.preprocessing import StandardScaler
from transformers import FilterSimu

def regression_analysis(X, y, methodType):
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    rng = np.random.RandomState(1)

    if methodType == 0:
        # linear regression
        from sklearn.linear_model import LinearRegression

        regressor = LinearRegression()
        regressor.fit(X, y)

        R2 = regressor.score(X, y)
        yp = regressor.predict(X)
        score = R2

        return regressor, yp, score
    elif methodType == 1:
        # decision tree
        from sklearn.tree import DecisionTreeRegressor

        regressor = DecisionTreeRegressor(max_depth=4)
        regressor.fit(X, y)
        yp = regressor.predict(X)
        R2 = regressor.score(X, y)

        score = R2

        return regressor, yp, score
    elif methodType == 2:
        # adaboost
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor
        regressor = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=4), n_estimators=300, learning_rate=1.0, loss='linear', random_state=rng)
        regressor.fit(X, y)
        yp = regressor.predict(X)
        R2 = regressor.score(X, y)

        score = R2

        return regressor, yp, score
    elif methodType == 3:
        # random forest
        from sklearn.ensemble import RandomForestRegressor

        regressor = RandomForestRegressor(n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2,
                                          min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                          max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                          bootstrap=True, oob_score=False, n_jobs=1, random_state=rng, verbose=0,
                                          warm_start=False)

        pipe = Pipeline([
            #('scale', StandardScaler()),
            ('filter', FilterSimu()),
            #('normalizer', Normalizer()),
            ('regression', regressor)
        ])

        nEstimators = [10, 50, 100]

        param_grid = {
            'filter__threshold': [0.9, 0.95, 0.97, 0.99],
            #'normalizer__norm': ['l1'],
            'regression__n_estimators': nEstimators,
            'regression__max_depth': [None, 10, 5, 3]
        }

        grid = GridSearchCV(pipe, cv=3, n_jobs=1, param_grid=param_grid)
        grid.fit(X, y)

        print X[:10]

        print grid.cv_results_.keys()
        mean_scores = np.array(grid.cv_results_['mean_test_score'])
        mean_tscores = np.array(grid.cv_results_['mean_train_score'])
        print mean_scores
        print mean_tscores

        print grid.best_params_
        print grid.best_score_
        print grid.cv_results_['params']



        regressor.fit(X, y)
        yp = regressor.predict(X)
        R2 = regressor.score(X, y)

        score = R2

        return regressor, yp, score
    elif methodType == 4:
        # GBR
        from sklearn.ensemble import GradientBoostingRegressor
        regressor = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                              criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
                                              min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
                                              min_impurity_split=None, init=None, random_state=None, max_features=None,
                                              alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
        regressor.fit(X, y)
        yp = regressor.predict(X)
        R2 = regressor.score(X, y)

        score = R2

        return regressor, yp, score
    elif methodType == 5:
        # Bayesian Ridge
        from sklearn.linear_model import BayesianRidge
        regressor = BayesianRidge(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06,
                                  compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False)
        regressor.fit(X, y)
        yp = regressor.predict(X)
        R2 = regressor.score(X, y)

        score = R2

        return regressor, yp, score
    elif methodType == 6:
        # Lasso
        from sklearn.linear_model import Lasso
        regressor = Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000,
                          tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
        regressor.fit(X, y)
        yp = regressor.predict(X)
        R2 = regressor.score(X, y)

        score = R2

        return regressor, yp, score
    elif methodType == 7:
        # logistic regression
        from sklearn.linear_model import LogisticRegression
        regressor = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                                       intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear',
                                       max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
        regressor.fit(X, y)
        yp = regressor.predict(X)
        R2 = regressor.score(X, y)

        score = R2

        return regressor, yp, score
    elif methodType == 8:
        # SVR
        from sklearn.svm import SVR
        regressor = SVR(kernel='poly', degree=10, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True,
                        cache_size=200, verbose=False, max_iter=-1)
        regressor.fit(X, y)
        yp = regressor.predict(X)
        R2 = regressor.score(X, y)

        score = R2

        return regressor, yp, score
    elif methodType == 9:
        # MLP
        from sklearn.neural_network import MLPRegressor
        regressor = MLPRegressor(hidden_layer_sizes=(200, ), activation='relu', solver='adam', alpha=0.0001,
                                 batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5,
                                 max_iter=200, shuffle=True, random_state=rng, tol=0.0001, verbose=False,
                                 warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
                                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        regressor.fit(X, y)
        yp = regressor.predict(X)
        R2 = regressor.score(X, y)

        score = R2

        return regressor, yp, score
    elif methodType == 10:
        # KN
        from sklearn.neighbors import KNeighborsRegressor
        regressor = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                        metric='minkowski', metric_params=None, n_jobs=1)
        regressor.fit(X, y)
        yp = regressor.predict(X)
        R2 = regressor.score(X, y)

        score = R2

        return regressor, yp, score
    else:
        return None, None, -1


