# Imports
import geopandas as gpd
import pandas as pd
from shapely import wkt
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_validate
import multiprocessing as mp
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

# Constants
EPSG_METER = 3035
CATEGORIES = ['accomendation', 'catering', 'education', 'health', 'leisure',
              'money', 'other', 'public', 'religion', 'shopping', 'tourism',
              'traffic', 'transport']

SOCIO_YEAR_DE = 2015
SOCIO_YEAR_FR = 2015


### National Districts Data
def map_official_id(df: pd.DataFrame, id: str):
    # method maps column of official_ids to the city name and returns it in an extra column
    # df: dataframe with id column
    # str: name of column with id

    df['assigned_city'] = df[id].map({
        11: 'berlin',
        2: 'hamburg',
        4: 'bremen',
        13055: 'marseille',
        69123: 'lyon',
        75056: 'paris'})

    return df


### OSM raw data processing#########################################################

def add_poi_category(df: pd.DataFrame):
    # returns modified dataframe with addtional column called category
    df['cat_code'] = df.code.astype(str).str[:2].astype(int)
    df['category'] = df.cat_code.map(
        {20: 'public', 21: 'health', 22: 'leisure', 23: 'catering', 24: 'accomendation',
         25: 'shopping', 26: 'money', 27: 'tourism', 29: 'other'})

    df.loc[df['fclass'] == 'university', 'category'] = 'education'
    df.loc[df['fclass'] == 'school', 'category'] = 'education'
    df.loc[df['fclass'] == 'kindergarten', 'category'] = 'education'
    df.loc[df['fclass'] == 'college', 'category'] = 'education'

    return df


def merge_filter_osm_amenities(name: str):
    # reads in all available OSM files and filters for relevant attributes
    # name: name of osm region as states in the filenmae
    # created dataframes are concatenated and returns as one large dataframe

    # Read in files and add categories
    POI = f'../data/raw/OSM-extract/{name}-latest-free.shp/gis_osm_pois_free_1.shp'
    POI_A = f'../data/raw/OSM-extract/{name}-latest-free.shp/gis_osm_pois_a_free_1.shp'

    POFW = f'../data/raw/OSM-extract/{name}-latest-free.shp/gis_osm_pofw_free_1.shp'
    POFW_A = f'../data/raw/OSM-extract/{name}-latest-free.shp/gis_osm_pofw_a_free_1.shp'

    TRAFFIC = f'../data/raw/OSM-extract/{name}-latest-free.shp/gis_osm_traffic_free_1.shp'
    TRAFFIC_A = f'../data/raw/OSM-extract/{name}-latest-free.shp/gis_osm_traffic_a_free_1.shp'

    TRANSPORT = f'../data/raw/OSM-extract/{name}-latest-free.shp/gis_osm_transport_free_1.shp'
    TRANSPORT_A = f'../data/raw/OSM-extract/{name}-latest-free.shp/gis_osm_transport_a_free_1.shp'

    poi = gpd.read_file(POI, crs={'init': 'epsg:4326'})
    poi = add_poi_category(poi)

    poi_a = gpd.read_file(POI_A, crs={'init': 'epsg:4326'})
    poi_a = add_poi_category(poi_a)

    transport = gpd.read_file(TRANSPORT, crs={'init': 'epsg:4326'})
    transport['category'] = 'transport'
    transport_a = gpd.read_file(TRANSPORT_A, crs={'init': 'epsg:4326'})
    transport_a['category'] = 'transport'

    traffic = gpd.read_file(TRAFFIC, crs={'init': 'epsg:4326'})
    traffic['category'] = 'traffic'
    traffic_a = gpd.read_file(TRAFFIC_A, crs={'init': 'epsg:4326'})
    traffic_a['category'] = 'traffic'

    pofw = gpd.read_file(POFW, crs={'init': 'epsg:4326'})
    pofw['category'] = 'religion'
    pofw_a = gpd.read_file(POFW_A, crs={'init': 'epsg:4326'})
    pofw_a['category'] = 'religion'

    # Filter for relevant classes
    traffic = traffic[traffic.fclass.isin(['street_lamp', 'fuel_station', 'parking'])]
    traffic_a = traffic_a[traffic_a.fclass.isin(['street_lamp', 'fuel_station', 'parking'])]

    transport = transport[transport.fclass.isin(['railway_station', 'railway_halt', 'tram_stop',
                                                 'bus_stop', 'bus_station', 'taxi_rank', 'airport'])]
    transport_a = transport_a[transport_a.fclass.isin(['railway_station', 'railway_halt', 'tram_stop',
                                                       'bus_stop', 'bus_station', 'taxi_rank', 'airport'])]

    return pd.concat([poi, poi_a, transport, transport_a, traffic, traffic_a, pofw, pofw_a])


# Defintion of mapping dictionaries
CLUSTER_DICT_FCLASS = {'college': 'university',
                       'bus_station': 'bus_stop',
                       'railway_halt': 'railway_station',
                       'biergarten': 'restaurant',
                       'pub': 'bar',
                       'food_court': 'fast_food',
                       'christian_anglican': 'christian',
                       'christian_catholic': 'christian',
                       'christian_evangelical': 'christian',
                       'christian_lutheran': 'christian',
                       'christian_methodist': 'christian',
                       'christian_orthodox': 'christian',
                       'christian_protestant': 'christian',
                       'christian_baptist': 'christian',
                       'muslim_sunni': 'muslim',
                       'muslim_shia': 'muslim'

                       }

CLUSTER_DICT_CODE = {2084: 2081,
                     5622: 5621,
                     5602: 5601,
                     2307: 2301,
                     2304: 2305,
                     2306: 2302,
                     3101: 3100,
                     3102: 3100,
                     3103: 3100,
                     3104: 3100,
                     3105: 3100,
                     3106: 3100,
                     3107: 3100,
                     3108: 3100,
                     3109: 3100,
                     3301: 3300,
                     3302: 3300

                     }


# Remap OSM categories which are likely to be confused by creators
def cluster_amenities(df: pd.DataFrame):
    # clusters point of interest according to a dictionary with mapping rules
    df.fclass = df.fclass.replace(CLUSTER_DICT_FCLASS)
    df.code = df.code.replace(CLUSTER_DICT_CODE)
    return df


def process_osm_amenities(region_name: (str), name: str):
    # region_name = name was stated in OSM file
    # name = name of desired city, small cap
    # returns processed osm of city

    df = merge_filter_osm_amenities(region_name)
    df = cluster_amenities(df)
    df = check_geolocation(df, name)
    return df


def check_geolocation(poi: gpd.GeoDataFrame, name: str):
    # Function maps poi to a city
    # poi: set of point of interests which have to be mapped to cities
    # name = name of city
    # returns poi GeoDataFrame with boolean values for each city

    nd_master = pd.read_csv('../data/merged/national_districts')
    cities = nd_master[((nd_master.admin_level == 4) & (nd_master.country == 'DE')) |
                       ((nd_master.admin_level == 8) & (nd_master.country == 'FR'))]
    cities.geometry = cities.geometry.apply(wkt.loads)
    cities_gdf = gpd.GeoDataFrame(cities, geometry='geometry', crs={'init': 'epsg:4326'})
    cities_gdf = cities_gdf.to_crs(epsg=EPSG_METER)
    cities_gdf = cities_gdf[cities_gdf.geographical_name == name.capitalize()]

    # map POIs to city and add new columns to untransformed geodataframe
    poi_trans = poi.to_crs(epsg=EPSG_METER)
    within = poi_trans.within(cities_gdf.iloc[0, 1].buffer(3000))
    poi = poi[within]
    poi['assigned_city'] = name

    # add the ids for the assigned cities
    poi['id'] = poi.assigned_city.map({
        'berlin': 'DE-official_id-11-admin_level-4',
        'hamburg': 'DE-official_id-02-admin_level-4',
        'bremen': 'DE-official_id-04-admin_level-4',
        'marseille': 'FR-official_id-13055-admin_level-8',
        'lyon': 'FR-official_id-69123-admin_level-8',
        'paris': 'FR-official_id-75056-admin_level-8'})

    return poi


##### OSM feature generation ######################################################

def get_features(shapes: gpd.GeoDataFrame, poi: gpd.GeoDataFrame, city_center: gpd.GeoDataFrame, radius_list: list,
                 city: str):
    # method generates minimal distance, location count, location score and distance to city center features
    # shapes: dataframe with building shapes of city
    # poi: dataframe with poi of city
    # city_center: dataframe with all city centers
    # radius_list: list of radius which should be used for feature generation
    # city: name of city as string

    # Convert files into geographic projection of meters
    shapes = shapes.to_crs(epsg=EPSG_METER)
    poi = poi.to_crs(epsg=EPSG_METER)
    city_center = city_center.to_crs(epsg=EPSG_METER)

    # transform shapes coordinates into array
    shapes_array = np.array([shapes.geometry.centroid.x, shapes.geometry.centroid.y]).T

    # filter selected city and transform to array
    city_center = city_center[city_center.assigned_city == city]
    cc_array = np.array([city_center.centroid.geometry.x, city_center.centroid.geometry.y]).T

    # for each poi category
    for p in poi.fclass.unique().tolist():
        # Marker for script
        print(p)

        # filter poi category
        poi_subset = poi[poi.fclass == p]
        # generate array with coordinates
        poi_array = np.array([poi_subset.centroid.geometry.x, poi_subset.centroid.geometry.y]).T
        # calculate distance matrix from all shapes to all poi of selected category
        dist_matrix = cdist(shapes_array, poi_array)

        # filter for minimal distance and store as feature
        shapes[str(p) + '_min_dist'] = dist_matrix.min(axis=1).tolist()

        # for every radius in given radius_list
        for radius in radius_list:
            # get unweighted score(so actual count) by first setting all distances larger than radius to radius
            # then count the ones which are unequal to radius (meaning they are within the given radius)
            dist_matrix_manipulated = np.where(dist_matrix > radius, radius, dist_matrix)
            shapes[str(p) + f'_area_count_{radius}'] = np.count_nonzero(dist_matrix_manipulated != radius, axis=1)

            # get weighted score by transforming distances using kernel and adding up al scores
            # dist_matrix_weighted = np.vectorize(kernel)
            # shapes[str(p) + f'_area_score_{radius}'] = dist_matrix_weighted(dist_matrix_manipulated, radius).sum(axis=1).tolist()
            shapes[str(p) + f'_area_score_{radius}'] = kernel(dist_matrix_manipulated, radius).sum(axis=1).tolist()

    # calculate distance to city center and store as feature
    shapes['dist_to_cc'] = cdist(shapes_array, cc_array)

    return shapes


def kernel(dist, radius):
    # Kernel function transforming distance to a weight/score depending on radius
    return (1 - (dist / radius))# ** 2) ** 2


##### Data preparation #####################################


def process_income_data(socios):
    # this method processes the income data by calculating the weighted average
    # socios: Dataframe with all socioeconomics data

    # Filter for german income data
    income_de = socios[socios.types == 'income_levels']
    # Select the middle of the interval as reference value
    income_de.loc[:, 'interval_center'] = income_de.subtype.map(
        {'0_1000': 500, '1000_1500': 1250, '1500_2500': 2000, '2500_3500': 3000, '3500_5000': 4250, '5000_plus': 6000})
    # delete existing income data for germany because it will be replaced with the new one
    socios = socios.loc[socios.types != 'income_levels'].copy()

    # for every quarter/neighborhood filter all data and calculate the weighted average by multiplying the middle of the interval
    # with the relative distribution value
    for quarter in income_de.districts_admin_level_11_official_id.unique().tolist():
        
        subset = income_de.loc[income_de.districts_admin_level_11_official_id == quarter].copy()###change1
        subset.loc[:, 'weighted_income'] = subset.interval_center * subset.relative_value
        weighted_result = subset.groupby('districts_admin_level_11_official_id')['weighted_income'].sum()
        new_income_row = subset.iloc[0, :]
        
        # store weighted result in "absolute" variable and set relative value to 1, label subtype as weighted_sum
        new_income_row.loc['subtype'] = 'weighted_sum'
        new_income_row.loc['absolute_value'] = weighted_result.reset_index().iloc[0, 1]
        new_income_row.loc['relative_value'] = 1
        # delete unused columns and add transformed data to socios dataset
        new_income_row = new_income_row.drop(index=['weighted_income', 'interval_center'])
        socios = socios.append(new_income_row)
        
    # filter socio data to remove everything labeled as "income_level_distribution_by_decile" out be keep mediane_euros      
    socios = socios.loc[
        ~((socios.types == 'income_level_distribution_by_decile') & (socios.subtype != 'mediane_euros'))]
    socios.types = socios.types.replace({'income_level_distribution_by_decile': 'income_levels'})

    # return final dataset
    return socios


def aggregate_add_socio_data(feature: gpd.GeoDataFrame, socios: gpd.GeoDataFrame, socio_list: list, value_type: list,
                             country: str):
    # this method adds a list of socioeconomic data to the feature dataset
    # feature: DataFrame of calculated features
    # socios: dataframe of all socioeconomics
    # socio_list: list with all variable names of socios which are supposed to be added
    # value_type: list of value types (absolute or relative) for each socioeconomic indicator according to the order of the socio_list
    # country: string with country code of city

    # Filter the feature data and make sure neighborhood id is included
    FEATURE_MASK = feature.loc[:, 'assigned_city':].columns.tolist()
    FEATURE_MASK.insert(1, 'districts_admin_level_11_id')

    
    
    # Group data by neighborhood and calculate median for each value
    feature = feature.groupby(['assigned_city', 'districts_admin_level_11_id']).median().reset_index().loc[:,
              FEATURE_MASK]


    # join each socioeconomic data from socio_list to each quarter
    for i, socio in enumerate(socio_list):
        SOCIO_MASK = [value_type[i], 'ph_id']
        feature = pd.merge(feature, socios[(socios.country == country) & (socios.types == socio)][SOCIO_MASK],
                           how='inner', left_on='districts_admin_level_11_id', right_on='ph_id').rename(
            columns={value_type[i]: socio}).drop(columns=['ph_id'])
    
    # generate random feature
    feature['random_noise'] = np.random.normal(1, 1, len(feature))    
    
    
    # add geometry shape of quarter, filter for one type to not add multiple times
    feature = pd.merge(feature, socios[(socios.country == country)&(socios.types == socio)][['geometry', 'ph_id']],
                       how='left', left_on='districts_admin_level_11_id', right_on='ph_id').drop(columns=['ph_id'])
    # return augmented dataframe
    return feature


def get_feature_target_correlation(agg_in: gpd.GeoDataFrame):
    # This method calculated all correlation coefficients between all features and target variables
    # agg: Dataframe of aggregated data on neighborhood level

    # store city name
    city = agg_in.loc[0, 'assigned_city']
    # calculate correlation
    agg, p_val = correlation_and_pvalues(agg_in)
       
    agg = agg.rename(columns={'index': 'feature'})
    # add column with amenity type
    agg['amenity'] = agg.feature.str.partition('_area')[0]
    # add column with radius used
    agg['radius'] = agg.feature.str.rpartition('_')[2]
    # add column with score type (score/count)
    agg['kind'] = agg.feature.str.split('_').str[-2]
    # extract amenity for minimum distances
    agg.loc[agg.radius == 'dist', 'amenity'] = agg.loc[agg.radius == 'dist', 'amenity'].str.partition('_min')[0]
    # Set radius to 0 for all "min_distance rows
    agg.radius = agg.radius.replace({'dist': 0})
    # add city name in extra column
    agg['assigned_city'] = city

    # return new dataframe
    return agg, p_val


def correlation_and_pvalues(agg: pd.DataFrame):
    # this method calculates all correlation coefficients and respective p-values for the target variables
    
    corr = pd.DataFrame(columns = ['feature', 'foreign_nationals', 'unemployment_rate', 'income_levels'])
    p_values = pd.DataFrame(columns = ['feature', 'foreign_nationals', 'unemployment_rate', 'income_levels'])
    i = 0

    for col in agg.iloc[:,2:-5].columns.tolist():
        r_foreign, p_foreign = stats.pearsonr(agg[col], agg['foreign_nationals'])
        r_income,p_income = stats.pearsonr(agg[col], agg['income_levels'])
        r_unemployment,p_unemployment = stats.pearsonr(agg[col], agg['unemployment_rate'])

        corr.loc[i,'feature']=col
        corr.loc[i,'foreign_nationals']=r_foreign
        corr.loc[i,'income_levels']=r_income
        corr.loc[i,'unemployment_rate']=r_unemployment

        p_values.loc[i,'feature']=col
        p_values.loc[i,'foreign_nationals']=p_foreign
        p_values.loc[i,'income_levels']=p_income
        p_values.loc[i,'unemployment_rate']=p_unemployment

        i+=1
        
    return corr, p_values


def plot_feature_target_correlation(agg: gpd.GeoDataFrame, ignore_min_dist: bool = False):
    # this method plots calculated correlations between targets and features as bar charts
    # agg: dataframe with correlations
    # ignore_min_dist: boolean to decide if correlation of 'min_distance' feature is supposed to be included or not

    # filter min_dist if boolean ist true
    if (ignore_min_dist):
        agg = agg.loc[agg.kind != 'min']

    # for each amenity plot one bar chart for each target variable
    for amen in agg.amenity.unique().tolist():
        fig = plt.figure(figsize=(12, 3))
        fig.subplots_adjust(hspace=0.4, wspace=0.8)
        ax = fig.add_subplot(1, 3, 1)
        sns.scatterplot(
            data=agg[(agg.amenity == amen)][['assigned_city', 'amenity', 'radius', 'kind', 'foreign_nationals']],
            x='radius', y='foreign_nationals', style='kind', hue='assigned_city', ax=ax, legend=False).set_title(amen)
        ax = fig.add_subplot(1, 3, 2)
        sns.scatterplot(
            data=agg[(agg.amenity == amen)][['assigned_city', 'amenity', 'radius', 'kind', 'unemployment_rate']],
            x='radius', y='unemployment_rate', style='kind', hue='assigned_city', ax=ax, legend=False).set_title(amen)
        ax = fig.add_subplot(1, 3, 3)
        sns.scatterplot(
            data=agg[(agg.amenity == amen)][['assigned_city', 'amenity', 'radius', 'kind', 'income_levels']],
            x='radius', y='income_levels', style='kind', hue='assigned_city', ax=ax).set_title(amen)
        plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)

    return fig


def aggregate_feature_in_category(agg: gpd.GeoDataFrame, poi_path: str = '../data/merged/pois'):
    # this methods aggregates the caculates features based on the amenity category
    # agg: data aggregated on neighborhood level

    # create new row for each feature in each quarter
    df = agg.melt(id_vars=['assigned_city', 'districts_admin_level_11_id'])
    # extract amenity type
    df['amenity'] = df.variable.str.partition('_area')[0]
    # extract radius
    df['radius'] = df.variable.str.rpartition('_')[2]
    # extract kind of feature (score or count)
    df['kind'] = df.variable.str.split('_').str[-2]

    # filter all feature which include a radius
    df = df.loc[(df.radius == '500') | (df.radius == '1000') | (df.radius == '2000')]

    # load poi data
    poi = get_csv_as_gpd(poi_path)
    # get mapping from amenity to category
    poi_cats = poi.groupby(['category', 'fclass']).count().reset_index()[['category', 'fclass']].rename(
        columns={'fclass': 'amenity'})
    # add category data
    df = pd.merge(df, poi_cats, how='inner', on='amenity')
    # group data by category and calculate median
    df = df.groupby(
        ['assigned_city', 'districts_admin_level_11_id', 'category', 'radius', 'kind']).median().reset_index()
    # generate new feature names in new column
    df['new_features'] = df.category + '_area_' + df.kind + '_' + df.radius.map(str)
    # transform result to move new features to columns
    df = df.pivot(index=['assigned_city', 'districts_admin_level_11_id'], columns='new_features',
                  values='value').reset_index()

    # remove old feature on amenity level
    MASK = ~(agg.columns.str.contains('1000') | agg.columns.str.contains('500') | agg.columns.str.contains('2000'))
    agg = agg.loc[:, MASK]
    # add new featuers on category level
    agg = pd.merge(df.loc[:, df.columns != 'assigned_city'], agg, how='inner', on='districts_admin_level_11_id')

    # return dataframe with new columns
    return agg


def correlation_filter(agg: gpd.GeoDataFrame, corr: gpd.GeoDataFrame, corr_filter: pd.DataFrame, filter_type: str):
    # this method filters out all variables/olumns which have a correlation which is lower than a random generated noise variable
    # agg: dataframe of aggreagted data on quarter level
    # corr: correlation dataframe of all correlations from feature to the different target variables
    # corr_filter: dataframe including the correlation values for the noise variable --> threshold value

    
    if(filter_type == 'random'):
        # filter columns according to thresshold in corr_filter
        corr = corr[corr.feature.isin(agg.columns.tolist())]
        corr_f = corr[((abs(corr.foreign_nationals) > abs(corr_filter.iloc[0, 4]))) | (
        (abs(corr.unemployment_rate) > abs(corr_filter.iloc[0, 5]))) | ((abs(corr.income_levels) > abs(corr_filter.iloc[0, 6])))]
    if(filter_type =='p_val'):
        corr_f = corr_filter[(corr_filter.foreign_nationals < 0.1)&(corr_filter.income_levels < 0.1)&(corr_filter.unemployment_rate < 0.1)]

    
    # add all column names to one large list and return filtered dataset
    #ls = pd.concat([corr_f_min, corr_f_score, corr_f_cc]).feature.tolist()
    ls = corr_f.feature.tolist()
    ls = ['assigned_city', 'districts_admin_level_11_id'] + ls + ['foreign_nationals', 'unemployment_rate',
                                                                  'income_levels']

    return agg.reindex(columns=ls)


def evaluate_by_regression(agg: gpd.GeoDataFrame):
    # this method generates an evlauation for each possible combination of features by brute-forcing all options
    # agg: dataframe with aggreagted data on quarter level

    # generate all possible combinations
    variations = list(product([500, 1000, 2000], repeat=13))
    # prepare output data frame by creating a column for each feature, start at one because first column (district_admin) is not feature
    out = pd.DataFrame(columns=(agg.columns.tolist() + ['mae', 'mape', 'target'])[1:])

    count = 0
    # for every variation fit model and store output
    for var in variations[:100]:

        count += 1
        print('combination ' + str(count))
        # generate feature names based on radius combinations
        agg = agg.dropna()
        cols = pd.DataFrame({'category': CATEGORIES, 'radius': list(var)})
        cols['feature_selected'] = cols.category + '_area_count_' + cols.radius.map(str)

        # filter all columns which do not include a radius feature
        MASK = ~(agg.columns.str.contains('admin_level') | (agg.columns.str.contains('assigned_city')) | (
            agg.columns.str.contains('1000')) | (agg.columns.str.contains('500')) |
                 agg.columns.str.contains('2000') | (agg.columns.str.contains('foreign')) | agg.columns.str.contains(
                    'unemployment') |
                 agg.columns.str.contains('income'))
        # create new feature mask based on min_dist and selected radius combination
        FEATURES = (agg.columns[MASK].tolist()) + (cols['feature_selected'].tolist())

        # split in feature and targets
        X = agg.loc[:, FEATURES]
        y_unemployment = agg.iloc[:, [-2]]
        y_foreign_nationals = agg.iloc[:, [-3]]
        y_income = agg.iloc[:, [-1]]

        for target in [y_foreign_nationals, y_unemployment, y_income]:

            # fit model and calculate scores based on CV
            model = Lasso(alpha=1)  # , normalize = True)
            scores = cross_validate(model, X, target, cv=4,
                                    scoring=('neg_mean_absolute_error', 'neg_mean_absolute_percentage_error'),
                                    return_estimator=True,
                                    n_jobs=4)

            # store results and used columns in dataframe
            for i, estimator in enumerate(scores['estimator']):
                current_index = len(out.index)
                out.loc[current_index, X.loc[:, abs(scores['estimator'][i].coef_) > 0].columns.tolist()] = True
                out.loc[current_index, 'mae'] = scores['test_neg_mean_absolute_error'][i]
                out.loc[current_index, 'mape'] = scores['test_neg_mean_absolute_percentage_error'][i]
                out.loc[current_index, 'target'] = target.columns[0]

    return out


def evaluate_by_regression_parallel(agg: gpd.GeoDataFrame):
    # this methods call the method combinatin_evaluation to via multiple threads to speed up computation time

    # generate all possible combinations
    variations = list(product([500, 1000, 2000], repeat=13))
    # prepare output data frame by creating a column for each feature, start at one because first column (district_admin) is not feature
    out = pd.DataFrame(columns=(agg.columns.tolist() + ['mae', 'mape', 'target'])[1:])

    # count number of available cpu and split data accordingly
    cpus = mp.cpu_count()
    chunks = np.array_split(variations, cpus)
    pool = mp.Pool(processes=cpus)
    # start evaluation in each pool
    chunk_processes = [pool.apply_async(combination_evaluation, args=(chunk, agg)) for chunk in chunks]
    # aggregate results of different pools
    results = [chunk.get() for chunk in chunk_processes]

    evaluation = pd.DataFrame(pd.concat(results))
    pool.close()

    return evaluation


def combination_evaluation(variations: list, agg: gpd.GeoDataFrame):
    # this method generates an evlauation for each possible combination of features by brute-forcing all options
    # agg: dataframe with aggreagted data on quarter level

    out = pd.DataFrame(columns=(agg.columns.tolist() + ['mae', 'mape', 'target'])[1:])
    count = 0
    # for every variation fit model and store output
    for var in variations:

        count += 1
        print('combination ' + str(count))
        # generate feature names based on radius combinations
        agg = agg.dropna()
        cols = pd.DataFrame({'category': CATEGORIES, 'radius': list(var)})
        cols['feature_selected'] = cols.category + '_area_count_' + cols.radius.map(str)

        # filter all columns which do not include a radius feature
        MASK = ~(agg.columns.str.contains('admin_level') | (agg.columns.str.contains('assigned_city')) | (
            agg.columns.str.contains('1000')) | (agg.columns.str.contains('500')) |
                 agg.columns.str.contains('2000') | (agg.columns.str.contains('foreign')) | agg.columns.str.contains(
                    'unemployment') |
                 agg.columns.str.contains('income'))
        # create new feature mask based on min_dist and selected radius combination
        FEATURES = (agg.columns[MASK].tolist()) + (cols['feature_selected'].tolist())

        # split in feature and targets
        X = agg.loc[:, FEATURES]
        y_unemployment = agg.iloc[:, [-2]]
        y_foreign_nationals = agg.iloc[:, [-3]]
        y_income = agg.iloc[:, [-1]]

        for target in [y_unemployment, y_income]:

            # fit model and calculate scores based on CV
            model = Lasso(alpha=1)  # , normalize = True)
            scores = cross_validate(model, X, target, cv=4,
                                    scoring=('neg_mean_absolute_error', 'neg_mean_absolute_percentage_error'),
                                    return_estimator=True)

            # store results and used columns in dataframe
            for i, estimator in enumerate(scores['estimator']):
                current_index = len(out.index)
                out.loc[current_index, X.loc[:, abs(scores['estimator'][i].coef_) > 0].columns.tolist()] = True
                out.loc[current_index, 'mae'] = scores['test_neg_mean_absolute_error'][i]
                out.loc[current_index, 'mape'] = scores['test_neg_mean_absolute_percentage_error'][i]
                out.loc[current_index, 'target'] = target.columns[0]

    return out



##### Model training #########################################


def prepare_socios(CITY: str, radius: int, density_type: str, socio_year_de: int):
    # This method processes the socio-economic indicators for the training process
    # Select year, process income and relabel
    
    SOCIO_YEAR_FR = 2015
    SOCIO_YEAR_DE = socio_year_de
    SOCIOS_PATH = '../data/merged/socioeconomics'

    #load socio data
    socios = get_csv_as_gpd(SOCIOS_PATH, CITY)
    socios = socios.rename(columns={'type': 'types'})
    #filter targets
    socios = socios[socios.types.isin(
    ['foreign_nationals', 'unemployment_rate', 'income_levels', 'income_level_distribution_by_decile'])].copy()
    #filter year of socio data
    socios = socios.loc[
    ((socios.country == 'DE') & (socios.year == SOCIO_YEAR_DE)) | ((socios.country == 'FR') & (socios.year == SOCIO_YEAR_FR))]
    # remove foreign_national subclass 'personne_francais'
    socios = socios.loc[
    ~((socios.country == 'FR') & (socios.types == 'foreign_nationals') & (socios.subtype != 'personnes_etrangeres'))].copy()

    # correct wrong realtive values from 1 to 0
    socios.loc[(socios.types == 'foreign_nationals') & (socios.assigned_city == 'bremen') & (
            socios.relative_value == 1), 'relative_value'] = 0
    socios.loc[(socios.types == 'foreign_nationals') & (socios.assigned_city == 'hamburg') & (
            socios.relative_value == 1), 'relative_value'] = 0

    socios.loc[(socios.types == 'unemployment_rate') & (socios.assigned_city == 'bremen') & (
            socios.relative_value == 1), 'relative_value'] = 0
    socios.loc[(socios.types == 'unemployment_rate') & (socios.assigned_city == 'hamburg') & (
            socios.relative_value == 1), 'relative_value'] = 0
    socios.loc[(socios.types == 'unemployment_rate') & (socios.assigned_city == 'berlin') & (
            socios.relative_value == 1), 'relative_value'] = 0
    socios = process_income_data(socios)
    return socios



def get_training_data(CITY: str, COUNTRY: str, radius: int, density_type: str, socio_year_de: int, output:str = 'agg'):
    
    """ method generates training data for a selected city including specified columns
    CITY: name of city
    COUNTRY: name of country city is in
    radius: radius for density feature
    density_type: density_type used for feature
    socio_year_de: year for socio data to be used
    """
    
    # constants
    SOCIO_YEAR_DE = socio_year_de
    RADIUS = radius
    DENSITY_TYPE = density_type
    SOCIOS_PATH = '../data/merged/socioeconomics'
    FEATURE_PATH = f'../data/processed/{CITY}_features'
    
    # prepare socio data    
    socios = prepare_socios(CITY, radius, density_type, socio_year_de)
    
    #load feature
    feature = get_csv_as_gpd(FEATURE_PATH)
    #filter out density column names depending on parameter
    density_columns = feature.columns[(feature.columns.str.contains(str(RADIUS)))&(feature.columns.str.contains(str(DENSITY_TYPE)))].tolist()
    #filter min_dist columns out 
    min_dist_columns = feature.columns[(feature.columns.str.contains('min_dist'))|(feature.columns.str.contains('dist_to_cc'))].tolist()
    #add columns for data merging
    merge_columns = ['districts_admin_level_11_id','assigned_city']
    
    #combine the lists and reindex feature data
    feature_subset = merge_columns + density_columns + min_dist_columns
    feature = feature.reindex(columns = (merge_columns + density_columns + min_dist_columns))
    
    # aggregate data and calculate weighted value of income in germany
    agg = aggregate_add_socio_data(feature, socios,
                                         ['foreign_nationals', 'unemployment_rate', 'income_levels'],
                                         ['relative_value', 'relative_value', 'absolute_value'], COUNTRY)

    
    #drop quarter with missing data
    agg = agg.dropna(subset = ['unemployment_rate', 'income_levels', 'foreign_nationals']).reset_index(drop = True)
    
    #scale target data with robust scaler
    scaler = RobustScaler(with_centering=True, with_scaling=True,quantile_range=(25.0, 75.0),copy=True)
    scaler.fit(agg[['unemployment_rate', 'income_levels', 'foreign_nationals']])
    agg.loc[:,['unemployment_rate', 'income_levels', 'foreign_nationals']] = scaler.transform(agg[['unemployment_rate', 'income_levels', 'foreign_nationals']])
    
    scaler_feature = RobustScaler(with_centering=True, with_scaling=True,quantile_range=(25.0, 75.0),copy=True)
    agg.iloc[:, 2:-5]=scaler_feature.fit_transform(agg.iloc[:, 2:-5])
    
    # due to small IQR we need to filter out some features
    if (density_type == 'score'):
        cols = agg.iloc[:,:2].columns.tolist()+agg.iloc[:, 2:-5].loc[:,scaler_feature.scale_>0.05].columns.tolist() + agg.iloc[:,-5:].columns.tolist()
        agg = agg.loc[:,cols]
         
    print('shape of training data '+ str(agg.shape))

    if(output == 'agg'):
        return agg
    else:
        return scaler



def train_lasso_regression(agg:pd.DataFrame, target:str, alphas: list ):
    """
    this method trains a lasso regression on the given data set and target and validates the best paramater using a gridsearch
    """
    
    #split data in feature and target
    X = agg.iloc[:,2:-5]
    y = agg.loc[:,[target]]      
    
    # generate train test split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 41)
    
    #train model in grid search with alpha parameter
    lasso = linear_model.Lasso(max_iter = 50000)
    parameters = {'alpha':alphas}
    clf = GridSearchCV(lasso, parameters, scoring = ['neg_mean_squared_error'], refit ='neg_mean_squared_error')
    clf.fit(X_train, y_train)
    
    #use best model to predict out of bag error
    y_pred = clf.predict(X_test)
    naive_mse = metrics.mean_squared_error(y_test, [y_train.mean()]*len(y_test))
    test_mse = metrics.mean_squared_error(y_test, y_pred)
    #test_mae = metrics.mean_absolute_error(y_test, y_pred)
    scores = clf.cv_results_['mean_test_neg_mean_squared_error'].tolist()+[naive_mse, test_mse]#, test_mae]
    #return all scores from grid search and test error
    return scores


def train_ridge_regression(agg:pd.DataFrame, target:str, alphas: list ):
    """
    this method trains a ridge regression on the given data set and target and validates the best paramater using a gridsearch
    """
        
    X = agg.iloc[:,2:-5]
    y = agg.loc[:,[target]]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 41)
    
    ridge = linear_model.Ridge()
    parameters = {'alpha':alphas}
    clf = GridSearchCV(ridge, parameters, scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_absolute_percentage_error'], refit ='neg_mean_squared_error')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    test_mse = metrics.mean_squared_error(y_test, y_pred)
    test_mae = metrics.mean_absolute_error(y_test, y_pred)
    scores = clf.cv_results_['mean_test_neg_mean_squared_error'].tolist()+[test_mse, test_mae]
    
    return scores


def train_pca_regression(agg: pd.DataFrame, target: str, threshold_grid: list):
    
    """
    Method performs a pca on data and trains linear regression based on resulting components
    agg: dataset to be used
    target: name of target variable
    threshold_grid: list of threshold values for explained values to determine number of components
    """
   

    scores = []
    pca_variance = pd.DataFrame(columns = ['components', 'variance'])
    mses = pd.DataFrame(columns = threshold_grid+['naive_mse', 'test_mse']) 
    
    X = agg.iloc[:,2:-5]
    y = agg.loc[:,[target]] 
        
    # generate pca for all possible number of components and calculate explained variance
    for i in range (0, min(len(X),len(X.columns.tolist()))-1):
        pca = PCA(n_components=i)
        pca.fit(X)
        pca_variance.loc[i, 'components'] = i
        pca_variance.loc[i,'variance'] = pca.explained_variance_ratio_.sum()

    # for each threshold value perform pca and train model
    for t in threshold_grid:
        #perform pca
        pca = PCA(n_components = pca_variance[pca_variance.variance>t].iloc[0,0])
        pca.fit(X)
        X_red = pd.DataFrame(pca.fit_transform(X))
        #train pca on reduced data
        X_train, X_test, y_train, y_test = train_test_split(X_red,y, test_size = 0.2, random_state = 41)
        reg = LinearRegression().fit(X_train, y_train)
        #calculate scores
        scores_mse = cross_val_score(reg, X_train, y_train, scoring = 'neg_mean_squared_error' ,cv=5).mean()
        scores.append(scores_mse)

    #calculate test_Error for best model
    pca = PCA(n_components = pca_variance[pca_variance.variance>threshold_grid[scores.index(max(scores))]].iloc[0,0]) 
    pca.fit(X)
    X_red = pd.DataFrame(pca.fit_transform(X))
    X_train, X_test, y_train, y_test = train_test_split(X_red,y, test_size = 0.2, random_state = 41)
    reg = LinearRegression().fit(X_train, y_train)

    scores.append(metrics.mean_squared_error(y_test, [y_train.mean()]*len(y_test)))
    scores.append(metrics.mean_squared_error(y_test, reg.predict(X_test)))

    return scores


def train_pca_lasso(agg: pd.DataFrame, target: str, alpha_grid: list):
    """
    Method performs a pca on data and trains lasso regression based on resulting components
    agg: dataset to be used
    target: name of target variable
    threshold_grid: list of threshold values for explained values to determine number of components
    """

    scores = []
    pca_variance = pd.DataFrame(columns=['components', 'variance'])
    mses = pd.DataFrame(columns=alpha_grid + ['naive_mse', 'test_mse'])

    city = agg.assigned_city.values[0]
    X = agg.iloc[:, 2:-5]
    y = agg.loc[:, [target]]

    # generate pca for all possible number of components and calculate explained variance
    for i in range(0, min(len(X), len(X.columns.tolist())) - 1):
        pca = PCA(n_components=i, random_state = 41)
        pca.fit(X)
        pca_variance.loc[i, 'components'] = i
        pca_variance.loc[i, 'variance'] = pca.explained_variance_ratio_.sum()

    # for each threshold value perform pca and train model
    # perform pca
    pca = PCA(n_components=pca_variance[pca_variance.variance > 0.99].iloc[0, 0], random_state = 41)
    pca.fit(X)
    X_red = pd.DataFrame(pca.fit_transform(X))
    # train pca on reduced data
    X_train, X_test, y_train, y_test = train_test_split(X_red, y, test_size=0.2, random_state=41)

    lasso = linear_model.Lasso(max_iter=50000)
    parameters = {'alpha': alpha_grid}
    clf = GridSearchCV(lasso, parameters, scoring=['neg_mean_squared_error'], refit='neg_mean_squared_error')
    clf.fit(X_train, y_train)

    # use best model to predict out of bag error
    y_pred = clf.predict(X_test)
    naive_mse = metrics.mean_squared_error(y_test, [y_train.mean()] * len(y_test))
    test_mse = metrics.mean_squared_error(y_test, y_pred)
    return clf.cv_results_['mean_test_neg_mean_squared_error'].tolist() + [naive_mse, test_mse]  # , test_mae



def get_best_lasso_model(agg: pd.DataFrame, target: str, city: str, country: str, socio_year: int, density_type: str, radius: int, output: str='predicts', meta_learner: bool=False):
    
    """
    Method trains the best lasso method and returns columns, predictions or MSE
    """
    
    # split data in feature and target
    model = 'lasso'
    X = agg.iloc[:, 2:-5]
    y = agg.loc[:, [target]]
   
    #create train test split for meta learner
    if (meta_learner == False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=41)
        X_meta, X_test, y_meta, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=41)

    # load data with all results of possible training combinations and find best parameter
    df = pd.read_csv(f'output/{model}_{target}.csv')
    df.iloc[:,1:] = df.iloc[:,1:].astype(float)
    df.loc[:, 'best_param'] = df.iloc[:,1:-2].idxmax(axis = 1)
    df[['city', 'radius', 'density_type']] = df['Unnamed: 0'].str.split('_', 2, True)

    df = df[(df.density_type== density_type)&(df.radius == str(radius))]
    
    # train lasso model with best found parameter
    lasso = Lasso(max_iter = 50000, alpha = float(df.loc[df.city == city,'best_param'].values[0]), )
    print(df.loc[df.city == city,'best_param'].values[0])
    lasso.fit(X_train, y_train)
    
    #make prediction on test set and create naive prediction
    y_pred = lasso.predict(X_test)
    naive_pred = [y_train.mean().values[0]] * len(y_test)

    # store results in dataframe
    predicts = pd.DataFrame({'y_test': y_test[target], 'y_pred': y_pred, 'naive': naive_pred})
    
    if(output == 'classifier'):
        return lasso
    if(output == 'used_columns'):
        return X.loc[:,lasso.coef_!=0].columns.tolist()
    if(output == 'full_predicts'):
        return lasso.predict(X_meta), lasso.predict(X_test)# lasso.predict(X)
    if(output == 'coef'):
        return lasso.coef_
    else:
        return predicts
    

def get_best_pca_lasso_model(agg: pd.DataFrame, target: str, city: str, country: str, socio_year: int, density_type: str, radius: int, output: str='predicts', meta_learner: bool = False):
    
    """
    Method trains the best pca lasso model and returns pca components, columns, predictions or MSE
    """
    # split data in feature and target
    model = 'pca_lasso'
    X = agg.iloc[:, 2:-5]
    y = agg.loc[:, [target]]
    
    # create dataframe with all possible number of pca compnents and calculate the explained variance
    pca_variance = pd.DataFrame(columns=['components', 'variance'])   
    for i in range(0, min(len(X), len(X.columns.tolist())) - 1):
        pca = PCA(n_components=i, random_state = 41)
        pca.fit(X)
        pca_variance.loc[i, 'components'] = i
        pca_variance.loc[i, 'variance'] = pca.explained_variance_ratio_.sum()

        
    print('shape before pca: '+str(X.shape))
    # select number of components based on threshold of 99% explained variance
    # perform pca
    pca = PCA(n_components=pca_variance[pca_variance.variance > 0.99].iloc[0, 0], random_state = 41)
    print('number of pca components: '+str(pca_variance[pca_variance.variance > 0.99].iloc[0, 0]))
    pca.fit(X)
    X_red = pd.DataFrame(pca.fit_transform(X))
    
    print('shape after pca: '+str(X_red.shape))
    
    #create train test split for meta learner
    if (meta_learner == False):
        X_train, X_test, y_train, y_test = train_test_split(X_red, y, test_size=0.2, random_state=41)
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(X_red, y, test_size=0.4, random_state=41)
        X_meta, X_test, y_meta, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=41)
    
    
    # load best parameter
    df = pd.read_csv(f'output/{model}_{target}.csv')
    df.iloc[:,1:] = df.iloc[:,1:].astype(float)
    df.loc[:, 'best_param'] = df.iloc[:,1:-2].idxmax(axis = 1)
    df[['city', 'radius', 'density_type']] = df['Unnamed: 0'].str.split('_', 2, True)

    df = df[(df.density_type== density_type)&(df.radius == str(radius))]
    
    # train lasso regression on reduced data to filter most important components
    lasso = Lasso(max_iter = 50000, alpha = float(df.loc[df.city == city,'best_param'].values[0]))
    print(df.loc[df.city == city,'best_param'].values[0])
    lasso.fit(X_train, y_train)

    #make predictions and store them in dataframe
    y_pred = lasso.predict(X_test)
    naive_pred = [y_train.mean().values[0]] * len(y_test)

    predicts = pd.DataFrame({'y_test': y_test[target], 'y_pred': y_pred, 'naive': naive_pred})
    
    
    if(output == 'classifier'):
        return lasso
    if(output == 'components'):
        return X_red
    if(output == 'used_columns'):
        return X_red.loc[:,lasso.coef_!=0].columns.tolist()
    if(output == 'full_predicts'):
        return lasso.predict(X_meta), lasso.predict(X_test) #lasso.predict(X_red)#
    if (output =='pca_classifier'):
        return pca
    if(output =='pca_var'):
        return pca_variance
    else:
        return predicts


def train_xgboost(agg: pd.DataFrame, target: str, cols: list=[], output: str = 'predicts', learner_type:str='city', weights: list=[]):
    
    """
    Method trains a XGBoost on a dataset and a selection of columns given in the cols list
    Returns either all predicts or the mse
    
    agg: scaled data set
    """
    # Define parameter grid for xgboost grid search    
    params = {'eta': [0.01, 0.1, 0.3, 0.5], 'max_depth': [1,2,3,4,8]}

    #Split data in feature in target    
    X = agg.iloc[:,:-5]
    y = agg.loc[:,[target]]
    
    # Filter the components which are supposed to be selected based on lasso regression from previous step
    X_subset = X.loc[:,cols]     
       
    #create train test split depedning on use case
    if (learner_type == 'transfer'):
        X_train = X_subset
        y_train = y       
    if (learner_type == 'meta'):
        X_train, X_temp, y_train, y_temp = train_test_split(X_subset, y, test_size=0.4, random_state=41)
        X_meta, X_test, y_meta, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=41)
    if (learner_type == 'city'):
        X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=41)
    
    xgboost = XGBRegressor(verbosity = 0)
    
    #train grid search and predict with the best model
    clf = GridSearchCV(xgboost, params, scoring = ['neg_mean_squared_error'], refit ='neg_mean_squared_error')
    
    if ((len(weights) !=0)):
        clf.fit(X_train, y_train, sample_weight= weights)
    else:
        clf.fit(X_train, y_train)
        
    # make prediction for test set unless it is the transfer case where the full data was used to train the model    
    if (learner_type != 'transfer'):
        y_pred = clf.predict(X_test)
        naive_pred = [y_train.mean().values[0]] * len(y_test)
        predicts = pd.DataFrame({'y_test': y_test[target],'y_pred': y_pred, 'naive':naive_pred})

        
        
        
    if (output=='mse'):
        return metrics.mean_squared_error(y_test, y_pred)
    if (output == 'classifier'):
        return clf   
    if (output == 'full_predicts'):
        return clf.predict(X_meta), clf.predict(X_test) 
    else:
        return predicts
    
    
def train_mean_model(agg: pd.DataFrame, target: str, city: str, country: str):
    
    """
    This mothod trains the model m5 as a combination on the boosted lasso and the boosted pca lasso model
    """
    
    predicts_master = pd.DataFrame(columns = ['y_pred_lasso_boosted', 'y_pred_pca_lasso_boosted', 'y_pred', 'y_test', 'y_naive'])

    # Lasso Boosted
    cols = get_best_lasso_model(agg=agg, target=target, city=city, country=country, socio_year=2015, density_type='count', radius = 1000, output ='used_columns')
    predicts = train_xgboost(agg, target, cols, 'predicts')
    predicts_master.loc[:, 'y_pred_lasso_boosted'] = predicts.y_pred

    # PCA Lasso boosted
    comps = get_best_pca_lasso_model(agg=agg,target=target, city=city, country=country, socio_year=2015,density_type='count', radius = 1000, output = 'components')
    cols = get_best_pca_lasso_model(agg=agg,target=target, city=city, country=country, socio_year=2015, density_type='count', radius = 1000, output = 'used_columns')
    reduced_data = pd.DataFrame(comps)
    reduced_data = reduced_data.join(agg.iloc[:,-5:])
    predicts = train_xgboost(reduced_data, target, cols, 'predicts')
    predicts_master.loc[:,'y_pred_pca_lasso_boosted'] = predicts.y_pred

    predicts_master.loc[:, 'y_pred'] = predicts_master[['y_pred_lasso_boosted', 'y_pred_pca_lasso_boosted']].mean(axis = 1)
    predicts_master.loc[:, 'y_test'] = predicts['y_test']
    predicts_master.loc[:, 'y_naive'] = predicts.naive

    targets = ['unemployment_rate', 'income_levels', 'foreign_nationals']
    scaler = get_training_data(city, country, 1000, 'count', 2015, 'scaler')
    scaler_new = RobustScaler()
    scaler_new.center_, scaler_new.scale_ = scaler.center_[targets.index(target)], scaler.scale_[targets.index(target)]
    predicts_master = pd.DataFrame(scaler_new.inverse_transform(predicts_master), columns =predicts_master.columns)

    return predicts_master


####Plot sensitivities ##################################

def plot_mse_overview(model: str, target: str, scale: str):
    
    """
    Method plots the MSE of each model as bar chart
    """
        
    df = pd.read_csv(f'output/{model}_{target}.csv')
    df.iloc[:,1:] = df.iloc[:,1:].astype(float)
    df.loc[:, 'best_param'] = df.iloc[:,1:-2].idxmax(axis = 1)
    df[['city', 'radius', 'density_type', 'socio_year', 'scaled']] = df['Unnamed: 0'].str.split('_', 4, True)
    
    df = df[(df.city.isin(['marseille', 'paris', 'lyon']))|((df.socio_year == '2015'))]
    
    ax = sns.barplot(data = df[(df.radius ==str(1000))&(df.density_type=='count')&(df.scaled == scale)][['city','naive_mse', 'test_mse']].melt(['city']), x='city', y='value', hue = 'variable')
    ax.set_title('MSE for '+model +' predicting '+target)
    
    

def plot_result_correlation(predicts: pd.DataFrame, city:str):
    
    """
    Method plots the correlation of the y_test and the y_pred in scatter plot and box plot 
    """
    
    corr = stats.pearsonr(predicts.y_test, predicts.y_pred)
    corr = [np.round(c, 3) for c in corr]

    text = 'r=%s, p=%s' % (corr[0], corr[1])
    
    
    fig = plt.figure(figsize=(10,5))
    
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(2, 2, 1)
    sns.regplot('y_test', 'y_pred', predicts, ci = 90, ax=ax)
    ax.set_title(city +' '+text)
    ax.set(xlabel='Actual Value', ylabel='Predicted Value')
    
    ax = fig.add_subplot(2, 2, 2)
    
    sns.boxplot(data= predicts, ax=ax)
    ax.set_title(city)
    ax.set(ylabel='Socioeconomic value')
    #return fig
    
    
def plot_socio_year_impact(model: str, target: str):
    
    """
    Plot improvement in % by socio year
    """
    
    model = model
    target = target
    df = pd.read_csv(f'output/{model}_{target}.csv')
    df.iloc[:,1:] = df.iloc[:,1:].astype(float)
    df.loc[:, 'best_param'] = df.iloc[:,1:-2].idxmax(axis = 1)
    df[['city', 'radius', 'density_type', 'socio_year', 'scaled']] = df['Unnamed: 0'].str.split('_', 4, True)

    df = df[(df.radius ==str(1000))&(df.density_type=='count')]
    df.loc[:, 'improvement_pct'] = 100 - ((df.test_mse / df.naive_mse)*100)
    df.loc[:, 'model'] = 'lasso'
    full = df[['city','model', 'scaled', 'radius', 'density_type', 'socio_year','naive_mse', 'test_mse', 'improvement_pct']].copy()

    ax = sns.barplot(data = full, x='city', y='improvement_pct', hue = 'socio_year')
    
def plot_scaling_impact(model: str, target: str, year: int):
    
    """
    Method plots improvement based on the scaling used
    """
    
    model = model
    target = target
    df = pd.read_csv(f'output/{model}_{target}.csv')
    df.iloc[:,1:] = df.iloc[:,1:].astype(float)
    df.loc[:, 'best_param'] = df.iloc[:,1:-2].idxmax(axis = 1)
    df[['city', 'radius', 'density_type', 'socio_year', 'scaled']] = df['Unnamed: 0'].str.split('_', 4, True)

    df = df[(df.city.isin(['marseille', 'paris', 'lyon']))|((df.socio_year == str(year)))]


    df = df[(df.radius ==str(1000))&(df.density_type=='count')]
    df.loc[:, 'improvement_pct'] = 100 - ((df.test_mse / df.naive_mse)*100)
    df.loc[:, 'model'] = 'lasso'
    full = df[['city','model', 'scaled', 'radius', 'density_type', 'socio_year','naive_mse', 'test_mse', 'improvement_pct']].copy()

    ax = sns.barplot(data = full, x='city', y='improvement_pct', hue = 'scaled')
    
def plot_model_comparison(target: str, year: int, ignore_pca: bool):
    
    
    """
    Method plots improvement in % for all scaling and model combinations to find the best option
    """
    
    model = 'lasso'
    target = target
    df = pd.read_csv(f'output/{model}_{target}.csv')
    df.iloc[:,1:] = df.iloc[:,1:].astype(float)
    df.loc[:, 'best_param'] = df.iloc[:,1:-2].idxmax(axis = 1)
    df[['city', 'radius', 'density_type', 'socio_year', 'scaled']] = df['Unnamed: 0'].str.split('_', 4, True)

    df = df[(df.city.isin(['marseille', 'paris', 'lyon']))|((df.socio_year == str(year)))]


    df = df[(df.radius ==str(1000))&(df.density_type=='count')]
    df.loc[:, 'improvement_pct'] = 100 - ((df.test_mse / df.naive_mse)*100)
    df.loc[:, 'model'] = 'lasso'
    full = df[['city','model', 'scaled', 'radius', 'density_type', 'socio_year','naive_mse', 'test_mse', 'improvement_pct']].copy()

    if (ignore_pca != True):

        model = 'pca'
        target = target
        df = pd.read_csv(f'output/{model}_{target}.csv')
        df.iloc[:,1:] = df.iloc[:,1:].astype(float)
        df.loc[:, 'best_param'] = df.iloc[:,1:-2].idxmax(axis = 1)
        df[['city', 'radius', 'density_type', 'socio_year', 'scaled']] = df['Unnamed: 0'].str.split('_', 4, True)

        df = df[(df.city.isin(['marseille', 'paris', 'lyon']))|((df.socio_year == str(year)))]


        df = df[(df.radius ==str(1000))&(df.density_type=='count')]
        df.loc[:, 'improvement_pct'] = 100 - ((df.test_mse / df.naive_mse)*100)
        df.loc[:, 'model'] = 'pca'
        full = full.append(df[['city','model', 'scaled', 'radius', 'density_type', 'socio_year','naive_mse', 'test_mse', 'improvement_pct']])

    model = 'pca_lasso'
    target = target
    df = pd.read_csv(f'output/{model}_{target}.csv')
    df.iloc[:,1:] = df.iloc[:,1:].astype(float)
    df.loc[:, 'best_param'] = df.iloc[:,1:-2].idxmax(axis = 1)
    df[['city', 'radius', 'density_type', 'socio_year', 'scaled']] = df['Unnamed: 0'].str.split('_', 4, True)

    df = df[(df.city.isin(['marseille', 'paris', 'lyon']))|((df.socio_year == str(year)))]


    df = df[(df.radius ==str(1000))&(df.density_type=='count')]
    df.loc[:, 'improvement_pct'] = 100 - ((df.test_mse / df.naive_mse)*100)
    df.loc[:, 'model'] = 'pca_lasso'
    full = full.append(df[['city','model', 'scaled', 'radius', 'density_type', 'socio_year','naive_mse', 'test_mse', 'improvement_pct']])

    full.loc[:,'model_type'] = full.model+'_'+full.scaled

    ax = sns.barplot(data = full, x='city', y='improvement_pct', hue = 'model_type')



##### Utilities #############################################
def get_csv_as_gpd(file: str, city: str = 'all'):
    """
    method reads in csv file which hs geodata stored in epsg:4326/WKT84
    file: path to file as string
    city: name of desired city. 'all' returns all cities
    """

    # open file
    df = pd.read_csv(file)
    # transform geometry column
    df.geometry = df.geometry.apply(wkt.loads)
    # load as geopandas with epsg
    geo_df = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    # Filter if city is specified
    if (city != 'all'):
        geo_df = geo_df[geo_df.assigned_city == city]

    return geo_df
