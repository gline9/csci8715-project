import pandas as pd
import geoplotlib
from geoplotlib.utils import DataAccessObject

upper_left_lat = 45.211201
upper_left_lon = -93.550766
lower_right_lat = 44.761985
lower_right_lon = -92.818414

ROUTES_DATA = [
    'mn_od_main_JT00_2017.csv',
    'mn_od_main_JT01_2017.csv',
    'mn_od_main_JT02_2017.csv',
    'mn_od_main_JT03_2017.csv',
    'mn_od_main_JT04_2015.csv',
    'mn_od_main_JT05_2015.csv'
]

block_to_lat_lon = pd.read_csv("C:/Users/gline/U of MN/2020 Spring/CSCI 8715/cbg_geographic_data.csv").set_index('census_block_group')
routes = pd.DataFrame()
for routes_data in ROUTES_DATA:
    routes = pd.concat([routes, pd.read_csv("C:/Users/gline/U of MN/2020 Spring/CSCI 8715/" + routes_data)[['w_geocode', 'h_geocode']]])

routes['w_geocode'] = routes['w_geocode'].map(lambda x: int(str(x)[0:12]))
routes['h_geocode'] = routes['h_geocode'].map(lambda x: int(str(x)[0:12]))

routes = pd.merge(left=routes, right=block_to_lat_lon[['latitude', 'longitude']].rename(columns={'latitude': 'work_lat', 'longitude': 'work_lon'}), left_on='w_geocode', right_on='census_block_group')
routes = pd.merge(left=routes, right=block_to_lat_lon[['latitude', 'longitude']].rename(columns={'latitude': 'home_lat', 'longitude': 'home_lon'}), left_on='h_geocode', right_on='census_block_group')
# work_locs = routes.rename(columns={"home_lat": "lat", "home_lon": "lon"}).sample(100000)
# geoplotlib.hist(DataAccessObject.from_dataframe(work_locs), cmap='summer', binsize=8)
# geoplotlib.show()

routes = routes[routes['work_lat'] < upper_left_lat]
routes = routes[routes['work_lon'] > upper_left_lon]
routes = routes[routes['home_lat'] < upper_left_lat]
routes = routes[routes['home_lon'] > upper_left_lon]
routes = routes[routes['work_lat'] > lower_right_lat]
routes = routes[routes['work_lon'] < lower_right_lon]
routes = routes[routes['home_lat'] > lower_right_lat]
routes = routes[routes['home_lon'] < lower_right_lon]

pd.to_pickle(routes, 'routes.pkl')
print(routes)

# print(routes)
# print(block_to_lat_lon)
