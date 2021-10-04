# This is script is trains the models and determine the best parameters using a cross-validation


#imports
import geopandas as gpd
import pandas as pd
from helper import get_training_data, train_lasso_regression, train_ridge_regression, train_pca_regression
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,  RobustScaler
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

year = 2015

alpha_grid = [0.0001, 0.001, 0.01, 0.1,0.2,0.3,0.4,0.5, 0.6,0.7,0.8,0.9,1,2,3,5]
result = pd.DataFrame(columns= alpha_grid+['naive_mse','test_mse'])
target = 'income_levels'

for city in ['marseille', 'lyon', 'paris']:
    for dens in ['score', 'count']:
        for radius in [500, 1000, 2000]:
            agg = get_training_data(city, 'FR', radius, dens, year)
            result.loc[str(city)+'_'+str(radius)+'_'+str(dens)] = train_lasso_regression(agg, target, alpha_grid)

for city in ['berlin', 'bremen', 'hamburg']:
    for dens in ['score', 'count']:
        for radius in [500, 1000, 2000]:
            agg = get_training_data(city, 'DE', radius, dens, year)
            result.loc[str(city)+'_'+str(radius)+'_'+str(dens)] = train_lasso_regression(agg, target, alpha_grid)


result.to_csv(f'output/lasso_{target}.csv')


alpha_grid = [0.0001, 0.001, 0.01, 0.1,0.2,0.3,0.4,0.5, 0.6,0.7,0.8,0.9,1,2,3,5]
result = pd.DataFrame(columns= alpha_grid+['naive_mse','test_mse'])
target = 'unemployment_rate'

for city in ['marseille', 'lyon', 'paris']:
    for dens in ['score', 'count']:
        for radius in [500, 1000, 2000]:
            agg = get_training_data(city, 'FR', radius, dens, year)
            result.loc[str(city)+'_'+str(radius)+'_'+str(dens)] = train_lasso_regression(agg, target, alpha_grid)

for city in ['berlin', 'bremen', 'hamburg']:
    for dens in ['score', 'count']:
        for radius in [500, 1000, 2000]:
            agg = get_training_data(city, 'DE', radius, dens, year)
            result.loc[str(city)+'_'+str(radius)+'_'+str(dens)] = train_lasso_regression(agg, target, alpha_grid)

result.to_csv(f'output/lasso_{target}.csv')


alpha_grid = [0.0001, 0.001, 0.01, 0.1,0.2,0.3,0.4,0.5, 0.6,0.7,0.8,0.9,1,2,3,5]
result = pd.DataFrame(columns= alpha_grid+['naive_mse','test_mse'])
target = 'foreign_nationals'

for city in ['marseille', 'lyon', 'paris']:
    for dens in ['score', 'count']:
        for radius in [500, 1000, 2000]:
            agg = get_training_data(city, 'FR', radius, dens, year)
            result.loc[str(city)+'_'+str(radius)+'_'+str(dens)] = train_lasso_regression(agg, target, alpha_grid)

for city in ['berlin', 'bremen', 'hamburg']:
    for dens in ['score', 'count']:
        for radius in [500, 1000, 2000]:
            agg = get_training_data(city, 'DE', radius, dens, year)
            result.loc[str(city)+'_'+str(radius)+'_'+str(dens)] = train_lasso_regression(agg, target, alpha_grid)


result.to_csv(f'output/lasso_{target}.csv')




alpha_grid = [0.0001, 0.001, 0.01, 0.1,0.2,0.3,0.4,0.5, 0.6,0.7,0.8,0.9,1,2,3,5]
pca_eval =  pd.DataFrame(columns=alpha_grid + ['naive_mse', 'test_mse'])
target = 'income_levels'

for city in ['marseille', 'lyon', 'paris']:
    for dens in ['score', 'count']:
        for radius in [500, 1000, 2000]:
            agg = get_training_data(city, 'FR', radius, dens, year)
            pca_eval.loc[str(city)+'_'+str(radius)+'_'+str(dens)] = (train_pca_lasso(agg, target, alpha_grid))


for city in ['berlin', 'bremen', 'hamburg']:
    for dens in ['score', 'count']:
        for radius in [500, 1000, 2000]:
            agg = get_training_data(city, 'DE', radius, dens, year)
            pca_eval.loc[str(city)+'_'+str(radius)+'_'+str(dens)] =(train_pca_lasso(agg, target, alpha_grid))


pca_eval.to_csv(f'output/pca_lasso_{target}.csv')


alpha_grid = [0.0001, 0.001, 0.01, 0.1,0.2,0.3,0.4,0.5, 0.6,0.7,0.8,0.9,1,2,3,5]
pca_eval =  pd.DataFrame(columns=alpha_grid + ['naive_mse', 'test_mse'])
target = 'foreign_nationals'

for city in ['marseille', 'lyon', 'paris']:
    for dens in ['score', 'count']:
        for radius in [500, 1000, 2000]:
            agg = get_training_data(city, 'FR', radius, dens, year)
            pca_eval.loc[str(city)+'_'+str(radius)+'_'+str(dens)] = (train_pca_lasso(agg, target, alpha_grid))


for city in ['berlin', 'bremen', 'hamburg']:
    for dens in ['score', 'count']:
        for radius in [500, 1000, 2000]:
            agg = get_training_data(city, 'DE', radius, dens, year)
            pca_eval.loc[str(city)+'_'+str(radius)+'_'+str(dens)] =(train_pca_lasso(agg, target, alpha_grid))

pca_eval.to_csv(f'output/pca_lasso_{target}.csv')

alpha_grid = [0.0001, 0.001, 0.01, 0.1,0.2,0.3,0.4,0.5, 0.6,0.7,0.8,0.9,1,2,3,5]
pca_eval =  pd.DataFrame(columns=alpha_grid + ['naive_mse', 'test_mse'])
target = 'unemployment_rate'

for city in ['marseille', 'lyon', 'paris']:
    for dens in ['score', 'count']:
        for radius in [500, 1000, 2000]:
            agg = get_training_data(city, 'FR', radius, dens, year)
            pca_eval.loc[str(city)+'_'+str(radius)+'_'+str(dens)] = (train_pca_lasso(agg, target, alpha_grid))


for city in ['berlin', 'bremen', 'hamburg']:
    for dens in ['score', 'count']:
        for radius in [500, 1000, 2000]:
            agg = get_training_data(city, 'DE', radius, dens, year)
            pca_eval.loc[str(city)+'_'+str(radius)+'_'+str(dens)] =(train_pca_lasso(agg, target, alpha_grid))


pca_eval.to_csv(f'output/pca_lasso_{target}.csv')