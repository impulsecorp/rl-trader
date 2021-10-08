from project_common import *

param_spaces = {}

param_spaces['AdaBoostClassifier'] = {}

param_spaces['BaggingClassifier'] = {'n_estimators': {'space': [10, 50, 100, 200],
                                                      'mut_prob': 0.5,
                                                      'default': 50},
                                     'max_samples': {'space': [0.1, 0.33, 0.75, 1.0],
                                                     'mut_prob': 0.25,
                                                     'default': 1.0},
                                     }

param_spaces['DecisionTreeClassifier'] = {'criterion': {'space': ['gini', 'entropy'],
                                                        'mut_prob': 0.15,
                                                        'default': 'gini'},
                                          'splitter': {'space': ['best', 'random'],
                                                       'mut_prob': 0.15,
                                                       'default': 'best'},
                                          'max_depth': {'space': [None, 1, 3, 5],
                                                        'mut_prob': 0.25,
                                                        'default': None},
                                          'max_features': {'space': [None, 0.25, 0.5, 0.8],
                                                           'mut_prob': 0.25,
                                                           'default': None},
                                          'min_samples_split': {'space': [2, 3],
                                                                'mut_prob': 0.5,
                                                                'default': 2},
                                          }

param_spaces['BernoulliNB'] = {'alpha': {'space': [0.0, 0.5, 1.0, 2.0],
                                         'mut_prob': 0.15,
                                         'default': 1.0},
                               'binarize': {'space': [0.0, 0.5, 1.0],
                                            'mut_prob': 0.15,
                                            'default': 0.0},
                               'fit_prior': {'space': [True, False],
                                             'mut_prob': 0.15,
                                             'default': True},
                               }

param_spaces['ExtraTreesClassifier'] = {'n_estimators': {'space': [10, 25, 50, 75],
                                                         'mut_prob': 0.15,
                                                         'default': 10},
                                        }

param_spaces['GaussianNB'] = {}

param_spaces['GaussianProcessClassifier'] = {}

param_spaces['GradientBoostingClassifier'] = {
    'loss': {'space': ['deviance', 'exponential']},
    'max_depth': {'space': [1, 5, 10, 15]},
    'max_features': {'space': ['sqrt', 'log2', None]},
    'learning_rate': {'space': [0.001, 0.05, 0.2]},
    'subsample': {'space': [0.5, 0.8, 0.95, 1.0]},
    'n_estimators': {'space': [10, 100, 500, 1000]},
}

param_spaces['KNeighborsClassifier'] = {'weights': {'space': ['uniform', 'distance'],
                                                    'mut_prob': 0.15,
                                                    'default': 'uniform'},
                                        'algorithm': {'space': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                                                      'mut_prob': 0.15,
                                                      'default': 'auto'},
                                        'n_neighbors': {'space': [2, 4, 8, 15, 25],
                                                        'mut_prob': 0.15,
                                                        'default': 5},
                                        'leaf_size': {'space': [10, 25, 50, 75],
                                                      'mut_prob': 0.15,
                                                      'default': 30},
                                        'p': {'space': [2, 3],
                                              'mut_prob': 0.15,
                                              'default': 2},
                                        }

param_spaces['LinearDiscriminantAnalysis'] = {}

param_spaces['LinearRegression'] = {
    'fit_intercept': {'space': [True, False],
                      'mut_prob': 0.15,
                      'default': True},
    'normalize': {'space': [True, False],
                  'mut_prob': 0.15,
                  'default': False},
}

param_spaces['LinearSVC'] = {'C': {'space': [0.5, 0.75, 0.95, 1.0],
                                   'mut_prob': 0.15,
                                   'default': 1.0},
                             }

param_spaces['LogisticRegression'] = {'solver': {'space': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                                                 'mut_prob': 0.15,
                                                 'default': 'liblinear'},
                                      'C': {'space': [0.5, 0.75, 0.95, 1.0],
                                            'mut_prob': 0.15,
                                            'default': 1.0},
                                      'fit_intercept': {'space': [True, False],
                                                        'mut_prob': 0.15,
                                                        'default': True},
                                      }

param_spaces['MLPClassifier'] = {'hidden_layer_sizes': {'space': [(10,), (30,), (50,), (100,),
                                                                  (10, 30), (50, 50), (80, 40)],
                                                        'mut_prob': 0.15,
                                                        'default': (100,)},
                                 'activation': {'space': ['identity', 'logistic', 'tanh', 'relu'],
                                                'mut_prob': 0.15,
                                                'default': 'relu'},
                                 'solver': {'space': ['lbfgs', 'sgd', 'adam'],
                                            'mut_prob': 0.15,
                                            'default': 'adam'},
                                 'learning_rate': {'space': ['constant', 'invscaling', 'adaptive'],
                                                   'mut_prob': 0.15,
                                                   'default': 'constant'},
                                 'learning_rate_init': {'space': [0.001 / 2, 0.001, 0.002],
                                                        'mut_prob': 0.15,
                                                        'default': 0.001},
                                 }

param_spaces['NuSVC'] = {}

param_spaces['RandomForestClassifier'] = {'criterion': {'space': ['gini', 'entropy'],
                                                        'mut_prob': 0.15,
                                                        'default': 'gini'},
                                          'n_estimators': {'space': [10, 25, 50, 75],
                                                           'mut_prob': 0.15,
                                                           'default': 10},
                                          'max_depth': {'space': [None, 1, 3, 5],
                                                        'mut_prob': 0.25,
                                                        'default': None},
                                          'max_features': {'space': [None, 'auto', 0.25, 0.5, 0.8],
                                                           'mut_prob': 0.25,
                                                           'default': 'auto'},
                                          'min_samples_split': {'space': [2, 3],
                                                                'mut_prob': 0.5,
                                                                'default': 2},
                                          }

param_spaces['SGDClassifier'] = {
    'loss': {'space': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber',
                       'epsilon_insensitive', 'squared_epsilon_insensitive']},
    'penalty': {'space': ['none', 'l2', 'l1', 'elasticnet']},
    'alpha': {'space': [ 0.00001, 0.0001, 0.001]},
    'learning_rate': {'space': ['constant', 'optimal', 'invscaling']},
    'class_weight': {'space': ['balanced', None]},
}

param_spaces['SVC'] = {'C': {'space': [0.1, 1.0, 10.0],
                             'mut_prob': 0.15,
                             'default': 1.0},
                       'kernel': {'space': ['linear', 'rbf'],
                                  'mut_prob': 0.15,
                                  'default': 'rbf'},

                       }

param_spaces['XGBClassifier'] = {'max_depth': {'space': [1, 3, 5, 7, 10, 15],
                                               'mut_prob': 0.15,
                                               'default': 3},
                                 'learning_rate': {'space': [0.01, 0.05, 0.1, 0.2],
                                                   'mut_prob': 0.15,
                                                   'default': 0.05},
                                 'n_estimators': {'space': [50, 100, 200,],
                                                  'mut_prob': 0.15,
                                                  'default': 50},
                                 'min_child_weight': {'space': [1, 5, 10, 50],
                                                      'mut_prob': 0.15,
                                                      'default': 5},
                                 'subsample': {'space': [0.5, 0.8, 1.0],
                                               'mut_prob': 0.15,
                                               'default': 0.5},
                                 'colsample_bytree': {'space': [0.5, 0.8, 1.0],
                                                      'mut_prob': 0.15,
                                                      'default': 0.5},
                                 }

param_spaces['XGBRegressor'] = {
    'max_depth': {'space': [1, 3, 8, 25],
                  'mut_prob': 0.15,
                  'default': 3},
    'subsample': {'space': [0.5, 1.0],
                  'mut_prob': 0.15,
                  'default': 0.5},
}

###########################
aps = {

    'GradientBoostingRegressor': {
        # Add in max_delta_step if classes are extremely imbalanced
        'max_depth': [1, 2, 3, 4, 5, 7, 10, 15],
        'max_features': ['sqrt', 'log2', None],
        'loss': ['ls', 'huber'],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
        'n_estimators': [10, 50, 100, 200, 500, 1000, 2000],
        'subsample': [0.5, 0.65, 0.8, 0.9, 0.95, 1.0]
    },

    'RandomForestRegressor': {
        'max_features': ['auto', 'sqrt', 'log2', None],
        'min_samples_split': [2, 5, 20, 50, 100],
        'min_samples_leaf': [1, 2, 5, 20, 50, 100],
        'bootstrap': [True, False]
    },
    'RidgeClassifier': {
        'alpha': [.0001, .001, .01, .1, 1, 10, 100, 1000],
        'class_weight': [None, 'balanced'],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
    },
    'Ridge': {
        'alpha': [.0001, .001, .01, .1, 1, 10, 100, 1000],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
    },
    'ExtraTreesRegressor': {
        'max_features': ['auto', 'sqrt', 'log2', None],
        'min_samples_split': [2, 5, 20, 50, 100],
        'min_samples_leaf': [1, 2, 5, 20, 50, 100],
        'bootstrap': [True, False]
    },
    'AdaBoostRegressor': {
        'base_estimator': [None, LinearRegression(n_jobs=-1)],
        'loss': ['linear', 'square', 'exponential']
    },
    'RANSACRegressor': {
        'min_samples': [None, .1, 100, 1000, 10000],
        'stop_probability': [0.99, 0.98, 0.95, 0.90]
    },
    'Lasso': {
        'selection': ['cyclic', 'random'],
        'tol': [.0000001, .000001, .00001, .0001, .001],
        'positive': [True, False]
    },

    'ElasticNet': {
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'selection': ['cyclic', 'random'],
        'tol': [.0000001, .000001, .00001, .0001, .001],
        'positive': [True, False]
    },

    'LassoLars': {
        'positive': [True, False],
        'max_iter': [50, 100, 250, 500, 1000]
    },

    'OrthogonalMatchingPursuit': {
        'n_nonzero_coefs': [None, 3, 5, 10, 25, 50, 75, 100, 200, 500]
    },

    'BayesianRidge': {
        'tol': [.0000001, .000001, .00001, .0001, .001],
        'alpha_1': [.0000001, .000001, .00001, .0001, .001],
        'lambda_1': [.0000001, .000001, .00001, .0001, .001],
        'lambda_2': [.0000001, .000001, .00001, .0001, .001]
    },

    'ARDRegression': {
        'tol': [.0000001, .000001, .00001, .0001, .001],
        'alpha_1': [.0000001, .000001, .00001, .0001, .001],
        'alpha_2': [.0000001, .000001, .00001, .0001, .001],
        'lambda_1': [.0000001, .000001, .00001, .0001, .001],
        'lambda_2': [.0000001, .000001, .00001, .0001, .001],
        'threshold_lambda': [100, 1000, 10000, 100000, 1000000]
    },

    'SGDRegressor': {
        'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
        'penalty': ['none', 'l2', 'l1', 'elasticnet'],
        'learning_rate': ['constant', 'optimal', 'invscaling'],
        'alpha': [.00001, .0001, .001]
    },

    'PassiveAggressiveRegressor': {
        'epsilon': [0.01, 0.05, 0.1, 0.2, 0.5],
        'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
        'C': [.0001, .001, .01, .1, 1, 10, 100, 1000],
    },

    'Perceptron': {
        'penalty': ['none', 'l2', 'l1', 'elasticnet'],
        'alpha': [.0000001, .000001, .00001, .0001, .001],
        'class_weight': ['balanced', None]
    },

    'PassiveAggressiveClassifier': {
        'loss': ['hinge', 'squared_hinge'],
        'class_weight': ['balanced', None],
        'C': [0.01, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    }

    , 'LightGBM': {
        'boosting_type': ['gbdt', 'dart']
        , 'min_child_samples': [1, 5, 7, 10, 15, 20, 35, 50, 100, 200, 500, 1000]
        , 'num_leaves': [2, 4, 7, 10, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100, 125, 150, 200, 250]
        , 'colsample_bytree': [0.7, 0.9, 1.0]
        , 'subsample': [0.7, 0.9, 1.0]
        , 'learning_rate': [0.01, 0.05, 0.1]
        , 'n_estimators': [5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000]
    }

    , 'LightGBMRegression': {
        'boosting_type': ['gbdt', 'dart']
        , 'min_child_samples': [1, 5, 7, 10, 15, 20, 35]#, 50, 100, 200, 500, 1000]
        , 'num_leaves': [2, 4, 7, 10, 15, 20, 25, 30]#, 35, 40, 50, 65, 80, 100, 125, 150, 200, 250]
        , 'colsample_bytree': [0.7, 0.9, 1.0]
        , 'subsample': [0.7, 0.9, 1.0]
        , 'learning_rate': [0.01, 0.05, 0.1]
        #, 'n_estimators': [5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000]
    }

    , 'LightGBMMultiClass': {
        'boosting_type': ['gbdt', 'dart']
        , 'min_child_samples': [1, 5, 7, 10, 15, 20, 35, 50, 100, 200, 500, 1000]
        , 'num_leaves': [2, 4, 7, 10, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100, 125, 150, 200, 250]
        , 'colsample_bytree': [0.7, 0.9, 1.0]
        , 'subsample': [0.7, 0.9, 1.0]
        , 'learning_rate': [0.01, 0.05, 0.1]
        , 'n_estimators': [5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000]
    }
    , 'CatBoostClassifier': {
        'depth': [1, 3, 5, 7, 9, 12, 15, 20, 32]
        , 'l2_leaf_reg': [.0000001, .000001, .00001, .0001, .001, .01, .1]
        , 'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]

    }

    , 'CatBoostRegressor': {
        'depth': [1, 3, 5, 7, 9, 12, 15, 20, 32]
        , 'l2_leaf_reg': [.0000001, .000001, .00001, .0001, .001, .01, .1]
        , 'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]

    }

    , 'LinearSVR': {
        'C': [0.5, 0.75, 0.85, 0.95, 1.0]
        , 'epsilon': [0, 0.05, 0.1, 0.15, 0.2]
    }
}
for k in aps.keys():
    param_spaces[k] = {}
    pd = aps[k]
    for pk, pv in pd.items():
        param_spaces[k][pk] = {}
        param_spaces[k][pk]['space'] = pv
        param_spaces[k][pk]['mut_prob'] = 0.15
        param_spaces[k][pk]['default'] = rnd.choice(pv)
###############################

param_spaces['DecisionTreeRegressor'] = {
                                          'splitter': {'space': ['best', 'random'],
                                                       'mut_prob': 0.15,
                                                       'default': 'best'},
                                          'max_depth': {'space': [None, 1, 3, 5],
                                                        'mut_prob': 0.25,
                                                        'default': None},
                                          'max_features': {'space': [None, 0.25, 0.5, 0.8],
                                                           'mut_prob': 0.25,
                                                           'default': None},
                                          'min_samples_split': {'space': [2, 3],
                                                                'mut_prob': 0.5,
                                                                'default': 2},
                                          }

param_spaces['KNeighborsRegressor'] = {'weights': {'space': ['uniform', 'distance'],
                                                    'mut_prob': 0.15,
                                                    'default': 'uniform'},
                                        'algorithm': {'space': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                                                      'mut_prob': 0.15,
                                                      'default': 'auto'},
                                        'n_neighbors': {'space': [2, 4, 8, 15, 25],
                                                        'mut_prob': 0.15,
                                                        'default': 5},
                                        'leaf_size': {'space': [10, 25, 50, 75],
                                                      'mut_prob': 0.15,
                                                      'default': 30},
                                        'p': {'space': [2, 3],
                                              'mut_prob': 0.15,
                                              'default': 2},
                                        }

param_spaces['BaggingRegressor'] = {'n_estimators': {'space': [10, 50, 100, 200],
                                                      'mut_prob': 0.5,
                                                      'default': 50},
                                     'max_samples': {'space': [0.1, 0.33, 0.75, 1.0],
                                                     'mut_prob': 0.25,
                                                     'default': 1.0},
                                     }


# the rest
akk = {
       'SnapshotEnsembles': SnapshotEnsembles,
       'ArcX4': ArcX4,
       'ProbabilisticClustering': ProbabilisticClustering,
       'InfiniteBoostingWithHoldoutCLF': InfiniteBoostingWithHoldoutCLF,
       'IsolationForest': IsolationForest,
       'LabelSpreading': LabelSpreading,
       'Lars': Lars,
       'OrthogonalMatchingPursuitCV': OrthogonalMatchingPursuitCV,
       'InfiniteBoostingCLF': InfiniteBoostingCLF,
       'ResearchGradientBoostingBaseCLF': ResearchGradientBoostingBaseCLF,
       'KmeansClustering': KmeansClustering,
       'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis,
       'DefragTreesClassifier': DefragTreesClassifier,
       'MILBoostClassifier': MILBoostClassifier,
       'FuzzyClustering': FuzzyClustering,
       'ClassSwitching': ClassSwitching,
       'Perceptron': Perceptron,
       'BinaryVadaboost': BinaryVadaboost,
       'BinaryLogitBoost': BinaryLogitBoost,
       'mlxtendMLP': mlxtendMLP,
       'mlxtendPerceptron': mlxtendPerceptron,
       'GentleBoostClassifier': GentleBoostClassifier,
       'LogitBoostClassifier': LogitBoostClassifier,
       'ARDRegression': ARDRegression,
       #'BaggingRegressor': BaggingRegressor,
       #'DecisionTreeRegressor': DecisionTreeRegressor,
       'GaussianProcessRegressor': GaussianProcessRegressor,
       'HuberRegressor': HuberRegressor,
       'MLPRegressor': MLPRegressor,
       'SVR': SVR,
       'TheilSenRegressor': TheilSenRegressor,
       'mlxtendLogisticRegression': mlxtendLogisticRegression,
       #'KNeighborsRegressor': KNeighborsRegressor,
       'SymbolicRegressor': SymbolicRegressor,
       #'PassiveAggressiveRegressor': PassiveAggressiveRegressor,
       }
for k, v in akk.items():
    param_spaces[k] = {}
