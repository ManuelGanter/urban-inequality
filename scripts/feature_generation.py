# This script loads pois, shapes and city center data and calculates the following features:
# minimal distances to each POI category
# counts of each poi category in surrounding area for a given radius list for every building shape
# location score for every building shape and each poi category for a given radius list
# distance to city center
# results are stored in a folder called "output" as csv
# Additional requirements: "helper.py"


# Imports
from helper import get_csv_as_gpd, get_features

# Constants
EPSG_METER = 3035
SHAPES_PATH = 'data/augmented_building_shapes'
POI_PATH = 'data/pois'
CITY_CENTER_PATH = 'data/city_center'
OUTPUT_PATH = '../output'
RADIUS_LIST = [500, 1000, 2000]


city_center = get_csv_as_gpd(CITY_CENTER_PATH)

# load data for each city
print('marseille')
marseille_shapes = get_csv_as_gpd(SHAPES_PATH, 'marseille')
marseille_poi = get_csv_as_gpd(POI_PATH, 'marseille')
marseille_features = get_features(shapes=marseille_shapes, poi=marseille_poi, city_center=city_center, radius_list=RADIUS_LIST, city='marseille')
marseille_features.to_csv(f'{OUTPUT_PATH}/marseille_features_lin_kernel', index=False)


print('paris')
paris_shapes = get_csv_as_gpd(SHAPES_PATH, 'paris')
paris_poi = get_csv_as_gpd(POI_PATH, 'paris')
paris_features = get_features(shapes=paris_shapes, poi=paris_poi, city_center=city_center, radius_list=RADIUS_LIST, city='paris')
paris_features.to_csv(f'{OUTPUT_PATH}/paris_features_lin_kernel', index=False)



print('lyon')
lyon_shapes = get_csv_as_gpd(SHAPES_PATH, 'lyon')
lyon_poi = get_csv_as_gpd(POI_PATH, 'lyon')
lyon_features = get_features(shapes=lyon_shapes, poi=lyon_poi, city_center=city_center, radius_list=RADIUS_LIST, city='lyon')
lyon_features.to_csv(f'{OUTPUT_PATH}/lyon_features_lin_kernel', index=False)


print('hamburg')
hamburg_shapes = get_csv_as_gpd(SHAPES_PATH, 'hamburg')
hamburg_poi = get_csv_as_gpd(POI_PATH, 'hamburg')
hamburg_features = get_features(shapes=hamburg_shapes, poi=hamburg_poi, city_center=city_center, radius_list=RADIUS_LIST, city='hamburg')
hamburg_features.to_csv(f'{OUTPUT_PATH}/hamburg_features_lin_kernel', index=False)


print('bremen')
bremen_shapes = get_csv_as_gpd(SHAPES_PATH, 'bremen')
bremen_poi = get_csv_as_gpd(POI_PATH, 'bremen')
bremen_features = get_features(shapes=bremen_shapes, poi=bremen_poi, city_center=city_center, radius_list=RADIUS_LIST, city='bremen')
bremen_features.to_csv(f'{OUTPUT_PATH}/bremen_features_lin_kernel', index=False)


print('berlin')
bremen_shapes = get_csv_as_gpd(SHAPES_PATH, 'berlin')
bremen_poi = get_csv_as_gpd(POI_PATH, 'berlin')
bremen_features = get_features(shapes=bremen_shapes, poi=bremen_poi, city_center=city_center, radius_list=RADIUS_LIST, city='berlin')
bremen_features.to_csv(f'{OUTPUT_PATH}/berlin_features_lin_kernel', index=False)
print('all done')


