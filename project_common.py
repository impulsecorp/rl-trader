import sys
sys.path.append('/home/peter/code/projects/')
sys.path.append('/home/peter/code/work/automl/')
sys.path.append('/home/ubuntu/')
sys.path.append('/home/ubuntu/new/automl')
from aidevutil import *
from massimport import *
from numpy import hstack, vstack, array, arange
import numpy as np
from keras.metrics import *

sk1_classes = { 
        # These Give Various Errors: 
        #
        #'SKBernoulliRBM':SKBernoulliRBM,
        #'gcForest':gcForest,
        # 'MGCForest':MGCForest,
        # 'RotationAdaBoostClassifier':RotationAdaBoostClassifier,
        # 'RotationForestClassifier':RotationForestClassifier,
        # 'BrownBoost':BrownBoost,
        #  'RotationForestClassifier2':RotationForestClassifier2,
        # 'FastFM':FastFM,
        #'RRForestClassifier':RRForestClassifier, 
        #'RRExtraTreesClassifier':RRExtraTreesClassifier,
        #'RGForest':RGForest,
        #'lBARN':lBARN,
        # 'AdalineClosed':Adaline,

        # These Give Key Errors:
        #
        #'CatBoostClassifier':CatBoostClassifier,
        #'InfiniteBoostingWithHoldoutCLF':InfiniteBoostingWithHoldoutCLF,
        #'InfiniteBoostingCLF':InfiniteBoostingCLF,
        #'ResearchGradientBoostingBaseCLF':ResearchGradientBoostingBaseCLF,
        #'DefragTreesClassifier':DefragTreesClassifier,
        #'MILBoostClassifier':MILBoostClassifier,
        #'Perceptron':Perceptron,
        #'mlxtendMLP':mlxtendMLP,
        #'mlxtendPerceptron':mlxtendPerceptron,
        #'BinaryLogitBoost':BinaryLogitBoost,
        }

class CatBoostClassifierWrapper:
    def __init__(self, **kw):
        self.clf = CatBoostClassifier(**kw)
    def fit(self, x, y):
        self.clf.fit(pd.DataFrame(x), pd.DataFrame(y), verbose=False)
    def predict(self, x):
        return self.clf.predict(pd.DataFrame(x))

class CatBoostRegressorWrapper:
    def __init__(self, **kw):
        self.clf = CatBoostRegressor(**kw)
    def fit(self, x, y):
        self.clf.fit(pd.DataFrame(x), pd.DataFrame(y), verbose=False)
    def predict(self, x):
        return self.clf.predict(pd.DataFrame(x))

class mlxtendMLPWrapper:
    def __init__(self, **kw):
        self.clf = mlxtendMLP(**kw)
    def fit(self, x, y):
        self.clf.fit(x, to_categorical(y))
    def predict(self, x):
        return np.argmax(self.clf.predict(x), axis=1)

class PerceptronWrapper:
    def __init__(self, **kw):
        self.clf = Perceptron(**kw)
    def fit(self, x, y):
        self.clf.fit(x, to_categorical(y))
    def predict(self, x):
        return np.argmax(self.clf.predict(x), axis=1)


class LightGBMWrapper:
    def __init__(self, **kw):
        self.kw = kw
    def fit(self, x, y):
        # specify your configurations as a dict
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_error',
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }
        params = {**params, **self.kw}
        lgb_train = lgb.Dataset(x, y.astype(int))
        self.gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=100,
                        valid_sets=None,
                        early_stopping_rounds=None)
    def predict(self, x):
        return self.gbm.predict(x, num_iteration=self.gbm.best_iteration).astype(np.int)


class LightGBMMultiClassWrapper:
    def __init__(self):
        pass
    def fit(self, x, y):
        # specify your configurations as a dict
        params = {
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'num_class': 14,
            'metric': 'multi_logloss',
            'subsample': .9,
            'colsample_bytree': .7,
            'reg_alpha': .01,
            'reg_lambda': .02,
            'num_leaves': 31,
            'min_split_gain': 0.01,
            'min_child_weight': 10,
            'silent': True,
            'verbosity': -1,}

        lgb_train = lgb.Dataset(x, y)
        self.gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=100,
                        valid_sets=None,
                        early_stopping_rounds=None)
    def predict(self, x):
        return self.gbm.predict(np.array(x), num_iteration=self.gbm.best_iteration).astype(np.int)

class LightGBMRegressionWrapper:
    def __init__(self, **kw):
        self.kw = kw
    def fit(self, x, y):
        # specify your configurations as a dict
        params = {'objective':'regression',
         'num_leaves': 31, #31
         'min_data_in_leaf': 30, #30
         'max_depth': 7,
         'learning_rate': 0.01,
         'lambda_l1':0.13,
         "boosting": "gbdt",
         "feature_fraction":0.85,
         'bagging_freq':8,
         "bagging_fraction": 0.9 ,
         "metric": 'rmse',
         "verbosity": -1,
         "random_state": None}
        params = {**params, **self.kw}
        lgb_train = lgb.Dataset(x, y.reshape(-1))
        self.gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=10000,
                        valid_sets=None,
                        early_stopping_rounds=None)
    def predict(self, x):
        return self.gbm.predict(np.array(x), num_iteration=self.gbm.best_iteration)#.astype(np.int)

class XGBRegressorWrapper:
    def __init__(self, **kw):
        kw = {'objective':'reg:linear',
                'eval_metric':'rmse',
                'learning_rate':.01,
                'eta': 0.15, # Step size shrinkage used in update to prevents overfitting
                'max_depth':5, #  - or 8
                'subsample': 0.6, # sample of rows  - or 1
                'colsample_bytree': 0.6, # sample of features - or .95
                'lambda':1, # l2 regu - or 25
                'random_state': rnd.randint(0,10000),
                'silent':True,
                'n_thread':1,
                'min_child_weight':100,
                'gamma':5,
                'colsample_bylevel':0.35,
                'alpha':25}
        self.clf = XGBRegressor(**kw)
    def fit(self, x, y):
        # specify your configurations as a dict
        self.clf.fit(x, y)#, early_stopping_rounds=10, )

    def predict(self, x):
        return self.clf.predict(x)


sk_classes = {
       'BaggingClassifier':BaggingClassifier,
       'DecisionTreeClassifier':DecisionTreeClassifier,
       'BernoulliNB':BernoulliNB,
       'ExtraTreesClassifier':ExtraTreesClassifier,
       'GaussianNB':GaussianNB,
    #   'LinearDiscriminantAnalysis':LinearDiscriminantAnalysis,
       'LogisticRegression':LogisticRegression,
       'RandomForestClassifier':RandomForestClassifier,
       'AdaBoostClassifier':AdaBoostClassifier,
       'GaussianProcessClassifier':GaussianProcessClassifier,
       'GradientBoostingClassifier':GradientBoostingClassifier,
       'KNeighborsClassifier':KNeighborsClassifier,
       'LinearRegression':LinearRegression,
       'LinearSVC':LinearSVC,
       'MLPClassifier':MLPClassifier,
       'NuSVC':NuSVC,
       'SVC':SVC,
       'XGBClassifier':XGBClassifier,
       'SGDClassifier':SGDClassifier,
  #     'SnapshotEnsembles':SnapshotEnsembles,
       'ArcX4':ArcX4,
       'BayesianRidge':BayesianRidge,
  #     'ElasticNet':ElasticNet,
       'LassoLars':LassoLars,
  #    # 'ProbabilisticClustering':ProbabilisticClustering,
       'IsolationForest':IsolationForest,
   #    'LabelSpreading':LabelSpreading,
       'Lars':Lars,
    #   'Lasso':Lasso,
       'PassiveAggressiveClassifier':PassiveAggressiveClassifier,
       'OrthogonalMatchingPursuit':OrthogonalMatchingPursuit,
   #    'OrthogonalMatchingPursuitCV':OrthogonalMatchingPursuitCV,
       'KmeansClustering':KmeansClustering,
       'QuadraticDiscriminantAnalysis':QuadraticDiscriminantAnalysis,
     #  'FuzzyClustering':FuzzyClustering,
       'ClassSwitching':ClassSwitching,
       'BinaryVadaboost':BinaryVadaboost,
       'GentleBoostClassifier':GentleBoostClassifier,
       'LogitBoostClassifier':LogitBoostClassifier,

        'RidgeClassifier':RidgeClassifier,

        'CatBoostClassifier':CatBoostClassifierWrapper,
        'InfiniteBoostingWithHoldoutCLF':InfiniteBoostingWithHoldoutCLF,
        'InfiniteBoostingCLF':InfiniteBoostingCLF,
        'ResearchGradientBoostingBaseCLF':ResearchGradientBoostingBaseCLF,
        'DefragTreesClassifier':DefragTreesClassifier,
        'MILBoostClassifier':MILBoostClassifier,
        'Perceptron':PerceptronWrapper,
        'mlxtendMLP':mlxtendMLPWrapper,
        'mlxtendPerceptron':mlxtendPerceptron,
        'BinaryLogitBoost':BinaryLogitBoost,
        'LightGBM' : LightGBMWrapper,

        #'LSTM_TMS':LSTM_TMS,
        'FCN':FCN,
        'MLP':MLP,
        'ResNet':ResNet,

    }

sndlev_classes = {'Ridge': Ridge,#(alpha=0.001, normalize=True, random_state=None),
#'LogisticRegression':LogisticRegression,
#'XGBClassifier':XGBClassifier,
#'MLPClassifier':MLPClassifier,
#'ExtraTreeClassifier':ExtraTreeClassifier,
#'GradientBoostingClassifier':GradientBoostingClassifier,
#'RandomForestClassifier':RandomForestClassifier,
#'KNeighborsClassifier':KNeighborsClassifier,
                 }

rg_classes = {
    'AdaBoostRegressor': AdaBoostRegressor,
    #'ARDRegression': ARDRegression, #slow
    'BaggingRegressor': BaggingRegressor,
    'DecisionTreeRegressor': DecisionTreeRegressor,
#    'ExtraTreesRegressor': ExtraTreesRegressor, # slow
#    'GaussianProcessRegressor': GaussianProcessRegressor, # slow
    'HuberRegressor': HuberRegressor,
    'LinearRegression': LinearRegression,
    'MLPRegressor': MLPRegressor, 
    'RandomForestRegressor': RandomForestRegressor,
    'Ridge': Ridge,
    'SVR': SVR,
#    'TheilSenRegressor': TheilSenRegressor, # slow
    'XGBRegressor': XGBRegressorWrapper,
#    'mlxtendLogisticRegression': mlxtendLogisticRegression, # slow
    'KNeighborsRegressor': KNeighborsRegressor,
#    'SymbolicRegressor': SymbolicRegressor, # slow
#    'CatBoostRegressor': CatBoostRegressorWrapper, # slow
    'SGDRegressor':SGDRegressor,

    'Lasso': Lasso,
    'ElasticNet': ElasticNet,
    'LassoLars': LassoLars,
#    'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit,
    'BayesianRidge': BayesianRidge,
    'PassiveAggressiveRegressor': PassiveAggressiveRegressor,
#    'LightGBMRegression': LightGBMRegressionWrapper,
}

class FakeClassifier:
    def __init__(self, **kw):
        pass
    def fit(self, x, y):
        pass
    def predict(self, x):
        return x
    def predict_proba(self, x):
        return x

from keras.applications import *
import gsk
class MobileNetClassifier:
    def __init__(self, **kw):
        self.model = MobileNet(input_shape=(gsk.TARGET_SIZE, gsk.TARGET_SIZE, 1), alpha=1.,
                               weights=None, classes=gsk.NCATS)
        self.model.load_weights('models/mobilenet_128x128_3_new')
    def fit(self, x, y):
        pass
    def predict(self, x):
        return x
    def predict_proba(self, x):
        return self.model.predict(x, verbose=1)

class ResNet50Classifier:
    def __init__(self, **kw):
        self.model = ResNet50(input_shape=(gsk.TARGET_SIZE, gsk.TARGET_SIZE,1),
                              weights=None, pooling=False, classes=gsk.NCATS)
        self.model.load_weights('models/resnet50_128x128_1')
    def fit(self, x, y):
        pass
    def predict(self, x):
        return x
    def predict_proba(self, x):
        return self.model.predict(x, verbose=1)

class XceptionClassifier:
    def __init__(self, **kw):
        self.model = Xception(input_shape=(gsk.TARGET_SIZE, gsk.TARGET_SIZE,1),
                              weights=None, pooling=False, classes=gsk.NCATS)
        self.model.load_weights('models/xception_128x128_11')
    def fit(self, x, y):
        pass
    def predict(self, x):
        return x
    def predict_proba(self, x):
        return self.model.predict(x, verbose=1)

class InceptionResNetClassifier:
    def __init__(self, **kw):
        self.model = InceptionResNetV2(input_shape=(gsk.TARGET_SIZE, gsk.TARGET_SIZE,1),
                                       weights=None, pooling=False, classes=gsk.NCATS)
        self.model.load_weights('models/inceptionresnetv2_128x128_1')
    def fit(self, x, y):
        pass
    def predict(self, x):
        return x
    def predict_proba(self, x):
        return self.model.predict(x, verbose=1)

class DenseNet121Classifier:
    def __init__(self, **kw):
        self.model = DenseNet121(input_shape=(gsk.TARGET_SIZE, gsk.TARGET_SIZE,1),
                                       weights=None, pooling=False, classes=gsk.NCATS)
        self.model.load_weights('models/densnset121_128x128_1')
    def fit(self, x, y):
        pass
    def predict(self, x):
        return x
    def predict_proba(self, x):
        return self.model.predict(x, verbose=1)


mc_classes = ['MobileNetClassifier',
              'ResNet50Classifier',
              #'XceptionClassifier', 
              'InceptionResNetClassifier',
              'DenseNet121Classifier']

m1c_classes = ['BernoulliNB',
'DecisionTreeClassifier',
'ExtraTreeClassifier',
'GaussianNB',
'KNeighborsClassifier',
'LinearDiscriminantAnalysis',
'MLPClassifier',
'RandomForestClassifier',
'GradientBoostingClassifier',
'SGDClassifier',
'SVC',

'LabelPropagation',
'NearestCentroid',
'RadiusNeighborsClassifier',
'LinearSVC',
'LogisticRegression',
'GaussianProcessClassifier',
'NuSVC',

'LabelSpreading',
'QuadraticDiscriminantAnalysis',
]
mc_classes = {x : eval(x) for x in mc_classes}

#sk_classes = {**sk_classes, **sndlev_classes}


class Raw:
    def __init__(self):
        pass
    def fit(self, x):
        pass
    def transform(self, x):
        return x
    
    
class SymTrans:
    def __init__(self):
        self.c = SymbolicTransformer(generations=10, population_size=1000, hall_of_fame=100, n_components=50, function_set=function_set, parsimony_coefficient=0.0005, max_samples=0.99, verbose=1, random_state=0, n_jobs=1)
    def fit(self, x, y):
        self.c.fit(x, y)
    def transform(self, x):
        mx = x
        tx = self.c.transform(x)
        return hstack([mx,tx])

class PolyFeats:
    def __init__(self):
        self.c = PolynomialFeatures(degree=1, interaction_only=True, include_bias=False)
    def fit(self, x):
        self.c.fit(x)
    def transform(self, x):
        mx = x
        tx = self.c.transform(x)
        return hstack([mx,tx])
    
    
class QuantileTransf:
    def __init__(self):
        self.c = preprocessing.QuantileTransformer(random_state=0, n_quantiles=10)
    def fit(self, x):
        self.c.fit(x)
    def transform(self, x):
        mx = x
        tx = self.c.transform(x)
        return hstack([mx,tx])


function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min']

pc_classes = {
               'SymbolicTransformer' : SymTrans,
               'PolynomialFeatures' : PolyFeats,
               'QuantileTransformer' : QuantileTransf,
               'PCA' : PCA,
               #'SelectKBest' : SelectKBest,               
               #'SelectPercentile' : SelectPercentile,
               #'LinearDiscriminantAnalysis' : LinearDiscriminantAnalysis, # (n_components=5),
               #'RBFSampler' : RBFSampler,
               #'Isomap' : Isomap,
               #'FeatureAgglomeration' : FeatureAgglomeration,
               'Raw' : Raw, 
             }

               #'SelectFromModel' : SelectFromModel, #(estimator=LinearRegression()),
               #'RandomTreesEmbedding' : RandomTreesEmbedding,o
               #'SFS' : SFS, #(estimator=KNeighborsClassifier(n_neighbors=2), k_features=(3, 15), forward=True, floating=False, scoring='accuracy', cv=5),
               #'EFS' : EFS, #(KNeighborsClassifier(n_neighbors=3), min_features=3, max_features=15, scoring='accuracy', print_progress=True, cv=5),
               #'TSNE' : TSNE,
               #'RFECV' : RFECV,# (estimator=LinearRegression()),
               #'FastICA' : FastICA,
               #'SpectralEmbedding' : SpectralEmbedding,
               #'LocallyLinearEmbedding' : LocallyLinearEmbedding,
            
            
def datasplit(dx, dy, idx, num_trials=3, test_size=0.2, mode='optimal'):
    if mode == 'optimal':
        ts = int(dx.shape[0]*test_size)
        js = int(dx.shape[0]/num_trials)
        if js < 1: js = 1
        rws = arange(dx.shape[0])
        n = 0
        for i,s in enumerate(slide_window(zip(dx,dy,rws), n=ts)):
            if i % js == 0:
                if n == idx:
                    tsxs = array([x[0] for x in s])
                    tsys = array([x[1] for x in s])
                    rs = array([x[2] for x in s])
                    bidx = array([True]*dx.shape[0])
                    bidx[rs, ...] = False
                    bidy = array([True]*dx.shape[0])
                    bidy[rs, ...] = False
                    trxs = dx[bidx, ...]
                    trys = dy[bidy, ...]
                    return trxs, tsxs, trys, tsys
                n += 1
                
        # if still found nothing
        trxs, tsxs, trys, tsys = train_test_split(dx, dy, test_size=test_size, random_state = rnd.randint(0, 100000))
        return trxs, tsxs, trys, tsys

        
    elif mode == 'random':
        trxs, tsxs, trys, tsys = train_test_split(dx, dy, test_size=test_size, random_state = rnd.randint(0, 100000))
        return trxs, tsxs, trys, tsys

defaults_binary_classification = {}
defaults_regression = {}
defaults_multiclass_classification = {'SVC' : {'probability': True},
                                      'NuSVC' : {'probability': True},
                                      'SGDClassifier': {'loss': 'log'},
                                      'LinearSVC' : {'multi_class': 'ovr'},
                                      'GaussianProcessClassifier' : {'multi_class': 'one_vs_one'},
                                      'LogisticRegression' : {'multi_class': 'ovr'},
                                      }

def keydecode(key, prediction_mode):
    kw={}
    if ' ' in key:
        k = key.split(' ')
        vs = ''.join(k[1:])
        k = k[0]
        kw = eval(vs)
        key = k
    else:
        if prediction_mode == 'binary_classification' and key in defaults_binary_classification: kw = defaults_binary_classification[key]
        if prediction_mode == 'multiclass_classification' and key in defaults_multiclass_classification: kw = defaults_multiclass_classification[key]
        if prediction_mode == 'regression' and key in defaults_regression: kw = defaults_regression[key]
    return key, kw

def decode_classifier(key, prediction_mode):
    key, kw = keydecode(key, prediction_mode)
    return sk_classes[key](**kw)

def decode_classifier_mc(key, prediction_mode):
    key, kw = keydecode(key, prediction_mode)
    return mc_classes[key](**kw)

def decode_regressor(key, prediction_mode):
    key, kw = keydecode(key, prediction_mode)
    return rg_classes[key](**kw)

def decode_classifier2(key, prediction_mode):
    key, kw = keydecode(key, prediction_mode)
    return sndlev_classes[key](**kw)

def decode_preprocessor(key, prediction_mode):
    key, kw = keydecode(key, prediction_mode)
    return pc_classes[key](**kw)

def RMSE(yt,yp):
    #yp[yp<0]=0
    #yp = np.expm1(yp)
    return np.sqrt(metrics.mean_squared_error(yt, yp))

def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def preds2catids(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def voting_mixer(am, weights, prediction_mode):
    if prediction_mode != 'multiclass_classification':
        if weights is not None:
            y_mix = [weights[i] * am[i] for i,n in enumerate(weights)]
            y_mix = np.sum(vstack(y_mix), axis=0)
            a=y_mix
        else:
            am = np.mean(vstack(am), axis=0)
            a=am
        return a
    else:
        if weights is not None:
            y_mix = [weights[i] * am[i] for i,n in enumerate(weights)]
            a = np.zeros_like(am[0])
            for i in y_mix:
                a += i
        else:
            a = np.zeros_like(am[0])
            for i in am:
                a += i
            a /= len(am)
        return a



mappings = {'binary_classification': decode_classifier,
            'regression': decode_regressor,
            'multiclass_classification': decode_classifier_mc}

mappings_2nd_level = {'binary_classification': decode_classifier2,
                      'regression': decode_regressor,
                      'multiclass_classification': decode_classifier_mc}

ar_mapping =  { 'binary_classification': sk_classes, 
                                      'multiclass_classification':  mc_classes, 
                                      'regression':  rg_classes}

def mmet(yp,yt):
    if len(yt.shape)==1:
        yt = to_categorical(yt)
    if len(yp.shape)==1:
        yp = to_categorical(yp)
    return accuracy_score(np.argmax(yp,axis=1), np.argmax(yt, axis=1))

metrics_mapping = { 'binary_classification': accuracy_score, 
                                      'multiclass_classification': mmet, # change to top 3
                                      'regression': RMSE}

afterprocessing_mapping =  { 'binary_classification': lambda x: x, 
                                      'multiclass_classification':  lambda x: x, 
                                      'regression':  lambda x: x}

sort_mapping =  { 'binary_classification': 'max', 
                                      'multiclass_classification':  'max', 
                                      'regression':  'min'}

def predictor(clf, prediction_mode):
    if prediction_mode == 'binary_classification':
        return clf.predict
    elif prediction_mode == 'regression':
        return clf.predict
    else:
        return clf.predict_proba

from sklearn.model_selection import *

def general_compute(args):
    
    clf_code, clf_kw, dx, dy, kw = args
    
    needs_categorical = kw['needs_categorical'] if 'needs_categorical' in kw else False
    prediction_mode = kw['prediction_mode']
    num_trials = kw['num_trials'] if 'num_trials' in kw else 1
    use_CV = kw['use_CV'] if 'use_CV' in kw else False
    CV_splits = kw['CV_splits'] if 'CV_splits' in kw else 5
    #if prediction_mode=='multiclass_classification': needs_categorical=True
    
    if (prediction_mode == 'regression') and use_CV:
        pdy = np.zeros_like(dy)#np.digitize(dy, bins=np.linspace(min(dy), max(dy),10))
    else:
        pdy = dy
            
    if 1:
        
        avg = 0
        for trial in range(num_trials):

            if use_CV:                    
                skf = StratifiedKFold(n_splits=CV_splits, shuffle=True, 
                                      random_state = rnd.randint(0, 100000)
                                     )
                cvavg = 0
                for tr, ts in skf.split(dx, pdy):
                    x_train = dx[tr]
                    y_train = dy[tr]
                    x_test = dx[ts]
                    y_test = dy[ts]

                    if needs_categorical:
                        y_train = to_categorical(y_train)
                        y_test = to_categorical(y_test)

                    clf = mappings[prediction_mode](clf_code, prediction_mode)
                    clf.fit(x_train, y_train)
                    p = predictor(clf, prediction_mode)(x_test)
                    s = metrics_mapping[prediction_mode](y_test, p)
                    
                    cvavg += s
                cvavg /= CV_splits
                
                avg += cvavg

            else:

                x_train, x_test, y_train, y_test = train_test_split(dx, dy, test_size=0.2, random_state = rnd.randint(0, 100000))

                if needs_categorical:
                    y_train = to_categorical(y_train)
                    y_test = to_categorical(y_test)
                
                clf = mappings[prediction_mode](clf_code, prediction_mode)
                clf.fit(x_train, y_train)
                p = predictor(clf, prediction_mode)(x_test)
                avg += metrics_mapping[prediction_mode](y_test, p)
                
        avg /= num_trials

        return avg
        
    #except Exception as ex:
    #    print('ERROR:',ex)
    #    return -0.01
