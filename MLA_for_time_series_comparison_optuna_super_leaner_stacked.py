#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('time', '', "#sklearn\nfrom sklearn.model_selection import train_test_split,RandomizedSearchCV\nfrom sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error\nfrom sklearn.metrics import mean_absolute_percentage_error,explained_variance_score\nfrom sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve,f1_score\nfrom sklearn import ensemble, linear_model, neighbors, svm, tree, neural_network\nimport xgboost as xgb\n\nfrom sklearn.linear_model import Ridge\nfrom sklearn.preprocessing import MinMaxScaler, minmax_scale, KBinsDiscretizer\nfrom sklearn import svm,model_selection, tree, linear_model, neighbors, ensemble, gaussian_process\n# from sklearn_extensions.extreme_learning_machines.elm import ELMClassifier,GenELMClassifier\n# from sklearn_extensions.extreme_learning_machines.random_layer import  RBFRandomLayer, MLPRandomLayer\nfrom scipy import stats\nimport tsfel\nimport optuna as op\nfrom statsmodels.tsa.stattools import acf, pacf\n\n\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport datetime as dt\nfrom math import sqrt\nfrom pandas.plotting import autocorrelation_plot\nfrom utils import series_to_supervised, decompose_time_to_features\nfrom sklearn.metrics import mean_absolute_error\nfrom lightgbm import LGBMRegressor\n\nfrom datetime import datetime\nfrom pandas.plotting import autocorrelation_plot\nfrom matplotlib import pyplot\n\n# Make numpy values easier to read.\nnp.set_printoptions(precision=4, suppress=True)\n\n# global variables\nseed = 2122022\nn_job = 4\nn_trial = 100\nrandom_iterations = 100\nscore = 'neg_mean_squared_error'\ncv = 5\nlags = 5\n\nimport warnings\nwarnings.filterwarnings('ignore')")


# In[2]:


def parser(s):
    return datetime.strptime(s,'%m/%d/%Y')

series = pd.read_csv("dataset/uk_wg_p_kg_se.csv")#, 
#                                   parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
series.head(5)


# In[3]:


series = series.drop(["year"], axis=1)
series.head()


# In[4]:


# https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/
from statsmodels.tsa.stattools import acf, pacf
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(series, lags=50)
pyplot.show()

# https://www.kite.com/python/answers/how-to-remove-outliers-from-a-pandas-dataframe-in-python

z_scores = stats.zscore(series)

abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
series = series[filtered_entries]
# In[5]:


data = series_to_supervised(series,lags)
# data_date = decompose_time_to_features(series)
data.head(5)


# In[6]:


data.describe()


# In[7]:


data.plot()

from scipy import signal
sos=signal.butter(4, .8,'low',output='sos') #Wn= cutoff frequency between 0 and 1
data=signal.sosfilt(sos, og_data)
data = pd.DataFrame(data)
data.head()
# In[8]:


get_ipython().run_cell_magic('time', '', '\ndata_new = data.values\nX, y = data_new[:, :-1], data_new[:, -1]\n\ntrainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=seed)\n\n# from scipy import signal\n# sos=signal.butter(4, .8,\'low\',output=\'sos\') #Wn= cutoff frequency between 0 and 1\n\n# trainX=signal.sosfilt(sos, trainX)\n# testX=signal.sosfilt(sos, testX)\n\n# trainX = np.log(trainX)\n# testX = np.log(testX)\n\n\n# X_train, X_test, y_train, y_test\nprint("shape of our training set",trainX.shape)\nprint("shape of our testing set",testX.shape)\n\n# check if the slicing is correct\nprint("shape of our training label",trainy.shape)\nprint("shape of our testing label",testy.shape)')

from metric_learn import NCA, LMNN, LFDA, MLKR, MMC
# metric_learning = False
lmnn = LMNN()
lmnn.fit(trainX, trainy)
trainX = lmnn.transform(trainX)
testX = lmnn.transform(testX)
# In[9]:


scalerX = MinMaxScaler(feature_range=(0, 1))
trainX = scalerX.fit_transform(trainX)
testX = scalerX.transform(testX)

scalery = MinMaxScaler(feature_range=(0, 1))
trainy = scalery.fit_transform(np.reshape(trainy,(trainy.size,1))) 
testy = scalery.transform(np.reshape(testy,(testy.size,1))) 


# check if the slicing is correct
print("shape of our training set",trainX.shape)
print("shape of our testing set",testX.shape)

# check if the slicing is correct
print("shape of our training label",trainy.shape)
print("shape of our testing label",testy.shape)


# In[10]:


def multiple_metric_reg(testyy,predictedyy):
    EVS = explained_variance_score(testyy,predictedyy) # best score 1
    R2= r2_score(testyy,predictedyy) # best score 1
    MAE = mean_absolute_error(testyy,predictedyy)
    MAPE = mean_absolute_percentage_error(testyy,predictedyy)
    MSE = mean_squared_error(testyy,predictedyy)
    RMSE = sqrt(mean_squared_error(testyy,predictedyy))
    return EVS, R2, MAE, MAPE, MSE, RMSE


# ## SVM

# In[11]:


# http://education.abcom.com/hyper-parameter-tuning-using-optuna/
# https://stackoverflow.com/questions/69071684/how-to-optimize-for-multiple-metrics-in-optuna
# https://stackoverflow.com/questions/62144904/python-how-to-retrive-the-best-model-from-optuna-lightgbm-study

trainXX, valX, trainyy, valy = train_test_split(trainX, trainy, test_size=0.2, random_state=seed)

def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])
        
def objective(trial):
    kernel=trial.suggest_categorical('kernel',['rbf','poly','linear','sigmoid'])
    c=trial.suggest_float("C",0.1,3.0,log=True)
    gamma=trial.suggest_categorical('gamma',['auto','scale'])
    degree=trial.suggest_int("degree",1,3,log=True)
    model =svm.SVR(kernel=kernel,degree=degree,gamma=gamma,C=c)
    model.fit(trainXX,trainyy)
    pred = model.predict(valX)
    trial.set_user_attr(key="best_booster", value=model)
    _,_,_,_,mse,_ = multiple_metric_reg(valy, pred)
    return mse

study = op.create_study(direction='minimize')
study.optimize(objective, n_trials=n_trial,n_jobs=-1, callbacks=[callback])


# In[12]:


best_svm_model=study.user_attrs["best_booster"]
trial=study.best_trial
print("Best Tuning Parameters : {} \n with accuracy of : {:.3f} %".format(trial.params,trial.value))
op.visualization.matplotlib.plot_optimization_history(study)
op.visualization.matplotlib.plot_param_importances(study)
op.visualization.matplotlib.plot_parallel_coordinate(study)


# ## XGBOOST

# In[13]:


# https://www.analyticsvidhya.com/blog/2021/11/tune-ml-models-in-no-time-with-optuna/

def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])
        
def objective(trial):

    # XGBoost parameters
    params = {
        "verbosity": 0,  # 0 (silent) - 3 (debug)
        "objective": "reg:squarederror",
        "n_estimators": 10000,
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.05),
        "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.2, 0.6),
        "subsample": trial.suggest_loguniform("subsample", 0.4, 0.8),
        "alpha": trial.suggest_loguniform("alpha", 0.01, 10.0),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 10.0),
        "gamma": trial.suggest_loguniform("lambda", 1e-8, 10.0),
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 10, 1000),
        "seed": seed,
        "n_jobs": -1,
    }
    
    model = xgb.XGBRegressor(random_state=seed, **params)
    model.fit(trainXX,trainyy)
    pred = model.predict(valX)
    trial.set_user_attr(key="best_booster", value=model)
    _,_,_,_,mse,_ = multiple_metric_reg(valy, pred)
    return mse

study1 = op.create_study(direction='minimize')
study1.optimize(objective, n_trials=n_trial,n_jobs=-1, callbacks=[callback])


# In[14]:


best_xgb_model=study1.user_attrs["best_booster"]
trial=study1.best_trial
print("Best Tuning Parameters : {} \n with accuracy of : {:.3f} %".format(trial.params,trial.value))
op.visualization.matplotlib.plot_optimization_history(study1)
op.visualization.matplotlib.plot_param_importances(study1)
# op.visualization.matplotlib.plot_parallel_coordinate(study1)


# ## LGBM

# In[16]:


# https://www.kaggle.com/luisandresgarcia/optuna?scriptVersionId=83651474
# https://www.kaggle.com/hamzaghanmi/lgbm-hyperparameter-tuning-using-optuna
def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])

def objective(trial):

    # XGBoost parameters
    params = {
        'metric': 'rmse', 
        'n_estimators': 20000,
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.006,0.008,0.01,0.014,0.017,0.02]),
        'max_depth': trial.suggest_categorical('max_depth', [10,20,100]),
        'num_leaves' : trial.suggest_int('num_leaves', 1, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100)
    }
    
    model = LGBMRegressor(random_state=seed, **params)
    model.fit(trainXX,trainyy)
    pred = model.predict(valX)
    trial.set_user_attr(key="best_booster", value=model)
    _,_,_,_,mse,_ = multiple_metric_reg(valy, pred)
    return mse

study2 = op.create_study(direction='minimize')
study2.optimize(objective, n_trials=n_trial,n_jobs=-1, callbacks=[callback])


# In[17]:


best_lgbm_model=study2.user_attrs["best_booster"]
trial=study2.best_trial
print("Best Tuning Parameters : {} \n with accuracy of : {:.3f} %".format(trial.params,trial.value))
op.visualization.matplotlib.plot_optimization_history(study2)
op.visualization.matplotlib.plot_param_importances(study2)
# op.visualization.matplotlib.plot_parallel_coordinate(study1)


# ## RANDONM FOREST

# In[18]:


import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler

def objective(trial):
    
    params = {
        "n_estimators": trial.suggest_int(name="n_estimators", low=100, high=1000),
        "max_depth": trial.suggest_float("max_depth", 4, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2.0, 150.0),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2.0, 60.0),
        "max_features": trial.suggest_categorical(
            "max_features", choices=["auto", "sqrt", "log2"]
        ),
        "n_jobs": -1,
#         "random_state": seed,
    }
    
    clf = ensemble.RandomForestRegressor(random_state=seed, **params)
    clf.fit(trainXX,trainyy)
    pred = clf.predict(valX)
    trial.set_user_attr(key="best_booster", value=clf)
    _,_,_,_,mse,_ = multiple_metric_reg(valy, pred)
    return mse

study3 = op.create_study(direction='minimize')
study3.optimize(objective, n_trials=n_trial,n_jobs=-1, callbacks=[callback])


# In[19]:


best_rf_model=study3.user_attrs["best_booster"]
trial=study3.best_trial
print("Best Tuning Parameters : {} \n with accuracy of : {:.3f} %".format(trial.params,trial.value))
op.visualization.matplotlib.plot_optimization_history(study3)
op.visualization.matplotlib.plot_param_importances(study3)
op.visualization.matplotlib.plot_parallel_coordinate(study3)


# ## MultiLayer Perceptron

# In[20]:


def objective(trial):

    params = {
        'learning_rate_init': trial.suggest_float('learning_rate_init ', 0.0001, 0.1, step=0.005),
        'first_layer_neurons': trial.suggest_int('first_layer_neurons', 10, 100, step=10),
        'second_layer_neurons': trial.suggest_int('second_layer_neurons', 10, 100, step=10),
        'third_layer_neurons': trial.suggest_int('second_layer_neurons', 10, 100, step=10),
        'activation': trial.suggest_categorical('activation', ['identity', 'tanh', 'relu']),
    }

    model = neural_network.MLPRegressor(
        hidden_layer_sizes=(params['first_layer_neurons'], params['second_layer_neurons'], params['third_layer_neurons']),
        learning_rate_init=params['learning_rate_init'],
        activation=params['activation'],
        random_state=seed,
        max_iter=1000
    )

    model.fit(trainXX,trainyy)
    pred = model.predict(valX)
    trial.set_user_attr(key="best_booster", value=model)
    _,_,_,_,mse,_ = multiple_metric_reg(valy, pred)
    return mse

study4 = op.create_study(direction='minimize')
study4.optimize(objective, n_trials=n_trial,n_jobs=-1, callbacks=[callback])


# In[21]:


best_mlp_model=study4.user_attrs["best_booster"]
trial=study4.best_trial
print("Best Tuning Parameters : {} \n with accuracy of : {:.3f} %".format(trial.params,trial.value))
op.visualization.matplotlib.plot_optimization_history(study4)
op.visualization.matplotlib.plot_param_importances(study4)
op.visualization.matplotlib.plot_parallel_coordinate(study4)


# ## KNN

# In[22]:


# https://www.kaggle.com/sashatarakanova/knn-with-hyperparameter-tuning-using-optuna
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler

import optuna
from optuna.samplers import TPESampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
# https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/

def objective(trial):
                
    # -- Tune estimator algorithm
    n_neighbors = trial.suggest_int("n_neighbors", 1, 30)
    weights = trial.suggest_categorical("weights", ['uniform', 'distance'])
    metric = trial.suggest_categorical("metric", ['euclidean', 'manhattan', 'minkowski'])
    knn = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, metric=metric)
    
    knn.fit(trainXX,trainyy)
    pred = knn.predict(valX)
    trial.set_user_attr(key="best_booster", value=knn)
    _,_,_,_,mse,_ = multiple_metric_reg(valy, pred)
    return mse

study5 = op.create_study(direction='minimize')
study5.optimize(objective, n_trials=n_trial,n_jobs=-1, callbacks=[callback])


# In[23]:


best_knn_model=study5.user_attrs["best_booster"]
trial=study5.best_trial
print("Best Tuning Parameters : {} \n with accuracy of : {:.3f} %".format(trial.params,trial.value))
op.visualization.matplotlib.plot_optimization_history(study5)
op.visualization.matplotlib.plot_param_importances(study5)
op.visualization.matplotlib.plot_parallel_coordinate(study5)


# ## XTREES FOREST

# In[24]:


import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler

def objective(trial):
    
    params = {
        "n_estimators": trial.suggest_int(name="n_estimators", low=100, high=1000),
        "max_depth": trial.suggest_float("max_depth", 4, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2.0, 150.0),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2.0, 60.0),
        "max_features": trial.suggest_categorical(
            "max_features", choices=["auto", "sqrt", "log2"]
        ),
        "n_jobs": -1,
#         "random_state": seed,
    }
    
    clf = ensemble.ExtraTreesRegressor(random_state=seed, **params)
    clf.fit(trainXX,trainyy)
    pred = clf.predict(valX)
    trial.set_user_attr(key="best_booster", value=clf)
    _,_,_,_,mse,_ = multiple_metric_reg(valy, pred)
    return mse

study6 = op.create_study(direction='minimize')
study6.optimize(objective, n_trials=n_trial,n_jobs=-1, callbacks=[callback])


# In[25]:


best_ets_model=study6.user_attrs["best_booster"]
trial=study6.best_trial
print("Best Tuning Parameters : {} \n with accuracy of : {:.3f} %".format(trial.params,trial.value))
op.visualization.matplotlib.plot_optimization_history(study6)
op.visualization.matplotlib.plot_param_importances(study6)
op.visualization.matplotlib.plot_parallel_coordinate(study6)


# In[26]:


MLA = [best_svm_model,best_rf_model,best_xgb_model, best_lgbm_model, best_mlp_model,best_knn_model, best_ets_model]


# In[27]:


# https://machinelearningmastery.com/super-learner-ensemble-in-python/

from math import sqrt
from numpy import hstack
from numpy import vstack
from numpy import asarray
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression


# collect out of fold predictions form k-fold cross validation
def get_out_of_fold_predictions(X, y, models):
	meta_X, meta_y = list(), list()
	# define split of data
	kfold = KFold(n_splits=5, shuffle=True)
	# enumerate splits
	for train_ix, test_ix in kfold.split(X):
		fold_yhats = list()
		# get data
		train_X, test_X = X[train_ix], X[test_ix]
		train_y, test_y = y[train_ix], y[test_ix]
		meta_y.extend(test_y)
		# fit and make predictions with each sub-model
		for model in models:
# 			model.fit(train_X, train_y)
			yhat = model.predict(test_X)
			# store columns
			fold_yhats.append(yhat.reshape(len(yhat),1))
		# store fold yhats as columns
		meta_X.append(hstack(fold_yhats))
	return vstack(meta_X), asarray(meta_y)

# fit all base models on the training dataset
def fit_base_models(X, y, models):
	for model in models:
		model.fit(X, y)
 
# fit a meta model
def fit_meta_model(X, y):
	model = LinearRegression()
	model.fit(X, y)
	return model

# evaluate a list of models on a dataset
def evaluate_models(X, y, models):
	for model in models:
		yhat = model.predict(X)
		mse = mean_squared_error(y, yhat)
		print('%s: RMSE %.3f' % (model.__class__.__name__, sqrt(mse)))
        
# make predictions with stacked model
def super_learner_predictions(X, models, meta_model):
	meta_X = list()
	for model in models:
		yhat = model.predict(X)
		meta_X.append(yhat.reshape(len(yhat),1))
	meta_X = hstack(meta_X)
	# predict
	return meta_model.predict(meta_X), meta_X


# In[28]:


# Stacked Ensemble learning
# get out of fold predictions
meta_X, meta_y = get_out_of_fold_predictions(trainX, trainy, MLA)
print('Meta ', meta_X.shape, meta_y.shape)
# fit base models
# fit_base_models(trainX, trainy, models)
# fit the meta model
meta_model = fit_meta_model(meta_X, meta_y)
# evaluate base models
evaluate_models(testX, testy, MLA)
# evaluate meta model
yhat, meta_mx = super_learner_predictions(testX, MLA, meta_model)
print('Super Learner: RMSE %.3f' % (sqrt(mean_squared_error(testy, yhat))))


# In[29]:


# Results of super learner
EVS = explained_variance_score(testy, yhat) # best score 1
R2= r2_score(testy, yhat) # best score 1
MAE = mean_absolute_error(testy, yhat)
MAPE = mean_absolute_percentage_error(testy, yhat)
MSE = mean_squared_error(testy, yhat)
RMSE = sqrt(mean_squared_error(testy, yhat))
print('Test R2: %.3f' % R2)
print('Test MAE: %.3f' % MAE)
print('Test MAPE: %.3f' % MAPE)
print('Test MSE: %.3f' % MSE)
print('Test RMSE: %.3f' % RMSE)
print('Test Explained_Variance_Score: %.3f' % EVS)


# In[30]:


# Voting ensemble
syhat = np.sum( np.asarray(meta_mx), axis = 1)/meta_mx.shape[1]
EVS = explained_variance_score(testy, syhat) # best score 1
R2= r2_score(testy, syhat) # best score 1
MAE = mean_absolute_error(testy, syhat)
MAPE = mean_absolute_percentage_error(testy, syhat)
MSE = mean_squared_error(testy, syhat)
RMSE = sqrt(mean_squared_error(testy, syhat))
print('Test R2: %.3f' % R2)
print('Test MAE: %.3f' % MAE)
print('Test MAPE: %.3f' % MAPE)
print('Test MSE: %.3f' % MSE)
print('Test RMSE: %.3f' % RMSE)
print('Test Explained_Variance_Score: %.3f' % EVS)


# In[31]:


get_ipython().run_cell_magic('time', '', "MLA_columns = []\nMLA_compare = pd.DataFrame(columns = MLA_columns)\n\nrow_index = 0\nfor alg in MLA:\n    predictedy = alg.predict(testX)\n    testyy,predictedyy = scalery.inverse_transform(testy),scalery.inverse_transform(np.reshape(predictedy,(predictedy.size,1)))\n    \n    MLA_name = alg.__class__.__name__\n    MLA_compare.loc[row_index,'MLA Name'] = MLA_name\n    MLA_compare.loc[row_index,'Parameters'] = str(alg)\n    \n    EVS = explained_variance_score(testyy,predictedyy) # best score 1\n    R2= r2_score(testyy,predictedyy) # best score 1\n    MAE = mean_absolute_error(testyy,predictedyy)\n    MAPE = mean_absolute_percentage_error(testyy,predictedyy)\n    MSE = mean_squared_error(testyy,predictedyy)\n    RMSE = sqrt(mean_squared_error(testyy,predictedyy))\n    \n    MLA_compare.loc[row_index, 'R2 score'] = round(R2, 4)\n    MLA_compare.loc[row_index, 'MAE'] = round(MAE, 4)\n    MLA_compare.loc[row_index, 'MAPE'] = round(MAPE, 4)\n    MLA_compare.loc[row_index, 'MSE'] = round(MSE, 4)    \n    MLA_compare.loc[row_index, 'RMSE'] = round(RMSE, 4)\n    MLA_compare.loc[row_index, 'EVS'] = round(EVS, 4)\n    \n    YEVS = explained_variance_score(testy,predictedy) # best score 1\n    YR2= r2_score(testy,predictedy) # best score 1\n    YMAE = mean_absolute_error(testy,predictedy)\n    YMAPE = mean_absolute_percentage_error(testy,predictedy)\n    YMSE = mean_squared_error(testy,predictedy)\n    YRMSE = sqrt(mean_squared_error(testy,predictedy))\n    \n    MLA_compare.loc[row_index, 'YR2 score'] = round(YR2, 4)\n    MLA_compare.loc[row_index, 'YMAE'] = round(YMAE, 4)\n    MLA_compare.loc[row_index, 'YMAPE'] = round(YMAPE, 4)\n    MLA_compare.loc[row_index, 'YMSE'] = round(YMSE, 4)\n    MLA_compare.loc[row_index, 'YRMSE'] = round(YRMSE, 4)\n    MLA_compare.loc[row_index, 'YEVS'] = round(YEVS, 4)\n    \n    \n    row_index+=1\n    \nMLA_compare.sort_values(by = ['YR2 score'], ascending = False, inplace = True)    \nMLA_compare")


# In[ ]:





# In[ ]:




