#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as ddf
from pandas import Series, DataFrame
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import seaborn as sn
import plotly.express as px


# ### Load temperature dataset

# In[3]:


ur_files = ddf.read_csv(r"US_counties_weather_2016.csv", dtype={'eightieth_percentile_income': 'float64','fips': 'object','fog': 'float64','hail': 'float64',
                   'median_household_income': 'float64','num_alcohol_impaired_driving_deaths': 'float64','num_chlamydia_cases': 'float64',
                   'num_deaths': 'float64','num_deaths_2': 'float64','num_deaths_3': 'float64','num_deaths_4': 'float64','num_deaths_5': 'float64',
                   'num_dentists': 'float64','num_driving_deaths': 'float64','num_drug_overdose_deaths': 'float64','num_firearm_fatalities': 'float64',
                   'num_hiv_cases': 'float64','num_households_with_severe_cost_burden': 'float64','num_injury_deaths': 'float64','num_mental_health_providers': 'float64',
                   'num_motor_vehicle_deaths': 'float64','num_primary_care_physicians': 'float64','num_workers_who_drive_alone': 'float64',
                   'per_capita_income': 'float64','percent_vaccinated': 'float64','percent_with_annual_mammogram': 'float64','preventable_hospitalization_rate': 'float64',
                   'rain': 'float64','snow': 'float64','station_id': 'object','thunder': 'float64','tornado': 'float64','twentieth_percentile_income': 'float64',
                   'deaths': 'float64','labor_force': 'float64','num_age_17_and_younger': 'float64','num_age_65_and_older': 'float64','num_american_indian_alaska_native': 'float64',
                   'num_asian': 'float64','num_associations': 'float64','num_below_poverty': 'float64','num_black': 'float64','num_disabled': 'float64','num_food_insecure': 'float64',
                   'num_hispanic': 'float64','num_homeowners': 'float64','num_households_CDC': 'float64','num_households_CHR': 'float64','num_households_with_no_vehicle': 'float64',
                   'num_housing_units': 'float64','num_institutionalized_in_group_quarters': 'float64','num_limited_english_abilities': 'float64','num_minorities': 'float64', 'num_mobile_homes': 'float64',
                   'num_multi_unit_housing': 'float64','num_native_hawaiian_other_pacific_islander': 'float64','num_no_highschool_diploma': 'float64','num_non_hispanic_white': 'float64',
                   'num_not_proficient_in_english': 'float64','num_overcrowding': 'float64','num_rural': 'float64','num_single_parent_households_CDC': 'float64','num_single_parent_households_CHR': 'float64',
                   'num_some_college': 'float64','num_unemployed_CDC': 'float64','num_unemployed_CHR': 'float64','num_uninsured': 'float64','num_uninsured_2': 'float64',
                   'num_uninsured_3': 'float64','population': 'float64','population_2': 'float64','total_population': 'float64', 'high_school_graduation_rate': 'float64',
                   'percent_age_65_and_older': 'float64', 'percent_mobile_homes': 'float64', 'closest_station_usaf_wban': 'object'})
df_svi = ur_files.compute()
df_svi.head()


# In[4]:


cols = df_svi.columns


# In[5]:


for col in cols:
    if 'income' in col:
        print(col)


# In[6]:


for col in cols:
    print(col)


# In[7]:


df_svi_2016 = df_svi[['state','county','fips','lat','lon','station_id','station_name','station_lat','station_lon','date','percent_below_poverty', 'per_capita_income','percent_unemployed_CDC', 
                      'percent_no_highschool_diploma', 'percent_age_65_and_older', 'percent_age_17_and_younger', 
                     'percent_disabled', 'percent_single_parent_households_CDC', 'percent_minorities', 'percent_limited_english_abilities', 
                      'percent_multi_unit_housing','percent_mobile_homes', 'percent_overcrowding', 'percent_no_vehicle', 
                      'percent_institutionalized_in_group_quarters', 'percentile_rank_below_poverty','percentile_rank_unemployed',
                      'percentile_rank_per_capita_income','percentile_rank_no_highschool_diploma','percentile_rank_socioeconomic_theme',
                      'percentile_rank_age_65_and_older','percentile_rank_age_17_and_younger','percentile_rank_disabled','percentile_rank_single_parent_households',
                      'percentile_rank_household_comp_disability_theme','percentile_rank_minorities','percentile_rank_limited_english_abilities',
                      'percentile_rank_minority_status_and_language_theme','percentile_rank_multi_unit_housing','percentile_rank_mobile_homes',
                      'percentile_rank_overcrowding','percentile_rank_no_vehicle','percentile_rank_institutionalized_in_group_quarters',
                      'percentile_rank_housing_and_transportation','percentile_rank_social_vulnerability']]


# In[8]:


df_svi_2016.head()


# In[9]:


df_svi_county = df_svi_2016.groupby(['state', 'county', 'fips']).mean()
df_svi_county = df_svi_county.reset_index()
df_svi_county.head()


# In[10]:


df_variable = df_svi_county[['state','county','fips','lat','lon','percent_below_poverty', 'per_capita_income','percent_unemployed_CDC', 
                      'percent_no_highschool_diploma', 'percent_age_65_and_older', 'percent_age_17_and_younger', 
                     'percent_disabled', 'percent_single_parent_households_CDC', 'percent_minorities', 'percent_limited_english_abilities', 
                      'percent_multi_unit_housing','percent_mobile_homes', 'percent_overcrowding', 'percent_no_vehicle', 
                      'percent_institutionalized_in_group_quarters']]


# In[11]:


def correlation_matrix(df):
    """
    A function to calculate and plot
    correlation matrix of a DataFrame.
    """
    # Create the matrix
    matrix = df.corr()
    
    # Create cmap
    cmap = sn.diverging_palette(250, 15, s=75, l=40,
                             n=9, center="light", as_cmap=True)
    # Create a mask
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    
    # Make figsize bigger
    fig, ax = plt.subplots(figsize=(16,12))
    
    # Plot the matrix
    _ = sn.heatmap(matrix, mask=mask, center=0, annot=True,
             fmt='.2f', square=True, cmap=cmap, ax=ax)


# In[12]:


df_variable.shape[0]


# In[13]:


correlation_matrix(df_variable)


# In[14]:


df_sd_16 = pd.read_csv(r'SuicideRate_Imputed_2016.csv', dtype={"fips": str})
df_sd_16 = df_sd_16[['county', 'fips', 'SuicideDeathRate']]
df_sd_16


# In[15]:


df_se_sd = df_variable.merge(df_sd_16, on = 'fips', how = 'left')
df_se_sd = df_se_sd.drop(['county_y'], axis = 1)
df_se_sd.head()


# In[16]:


df_se_sd = df_se_sd.rename(columns = {'county_x': 'county'})
df_se_sd.head()


# In[17]:


df_se_sd.to_csv('Social_Economic_Suicide_2016.csv')


# In[18]:


df_percentile = df_svi_county[['state','county','fips','lat','lon','percentile_rank_below_poverty','percentile_rank_unemployed',
                      'percentile_rank_per_capita_income','percentile_rank_no_highschool_diploma','percentile_rank_socioeconomic_theme',
                      'percentile_rank_age_65_and_older','percentile_rank_age_17_and_younger','percentile_rank_disabled','percentile_rank_single_parent_households',
                      'percentile_rank_household_comp_disability_theme','percentile_rank_minorities','percentile_rank_limited_english_abilities',
                      'percentile_rank_minority_status_and_language_theme','percentile_rank_multi_unit_housing','percentile_rank_mobile_homes',
                      'percentile_rank_overcrowding','percentile_rank_no_vehicle','percentile_rank_institutionalized_in_group_quarters',
                      'percentile_rank_housing_and_transportation','percentile_rank_social_vulnerability']]


# In[19]:


df_percentile


# In[20]:


df_sample = df_percentile[df_percentile['county'] == 'Adams']
df_sample


# In[21]:


df_sample2 = df_sample[df_sample['state'] == 'Washington']


# In[22]:


df_result = df_sample2[['percentile_rank_socioeconomic_theme', 'percentile_rank_household_comp_disability_theme', 'percentile_rank_minority_status_and_language_theme',
                        'percentile_rank_housing_and_transportation', 'percentile_rank_social_vulnerability']]
df_result


# ### Find the climate data part

# In[23]:


ur_files = ddf.read_csv('Climate_Data_Census_Tract_Level_2016.csv', dtype = {'year': str, 'station_id': str, 'STATEFP': str, 'COUNTYFP': str})
climate_2016 = ur_files.compute()
climate_2016 = climate_2016.loc[:, ~climate_2016.columns.str.contains('^Unnamed')]
climate_2016.head()


# In[24]:


climate_2016['fips'] = climate_2016['STATEFP'] + climate_2016['COUNTYFP']


# In[25]:


climate_county_2016 = climate_2016.groupby(['fips']).mean()
climate_county_2016 = climate_county_2016.reset_index()
climate_county_2016.head()


# In[26]:


climate_county_2016.columns


# In[27]:


climate_county_2016 = climate_county_2016[['fips', 'elevation', 'mean_temp',
       'dewpoint', 'sea_level_pressure', 'station_pressure', 'visibility',
       'wind_speed', 'precipitation', 'Relative_Humidity', 'Heat_Index', 'fog',
       'rain', 'snow', 'hail', 'thunder', 'tornado']]


# In[28]:


climate_county_2016.head()


# In[29]:


correlation_matrix(climate_county_2016)


# ## Load the Humidity Temperature Index Data

# In[30]:


ur_files = ddf.read_csv(r"imputed_humid_index_2016.csv", dtype = {'county': str, 'STATE': str, 'COUNTY': str})
df_humidity = ur_files.compute()
df_humidity = df_humidity.loc[:, ~df_humidity.columns.str.contains('^Unnamed')]
df_humidity.head()


# #### From the above graph, we can see that max temp/min temp/dewpoint are highly correlated with mean temp, max Wind speed and max wind gust are highly correlated with wind values, so we can drop max temp, min temp, max wind speed, and max wind gust.

# In[31]:


df_climate2 = climate_county_2016.drop(['dewpoint'], axis=1)
correlation_matrix(df_climate2)


# ### upload PM2.5 Data

# In[32]:


ur_files = ddf.read_csv(r"PM2.5.csv", dtype = {'CountyFIPS': 'object'})
df_pm = ur_files.compute()
df_pm = df_pm.loc[:, ~df_pm.columns.str.contains('^Unnamed')]
df_pm.head()


# In[33]:


df_pm_2016 = df_pm[df_pm['Year'] == 2016]
df_pm_2016 = df_pm_2016.rename(columns = {'State': 'state', 'CountyFIPS': 'fips', 'County': 'county'})
df_pm_2016 = df_pm_2016.groupby(['state', 'fips', 'county']).mean()
df_pm_2016 = df_pm_2016.reset_index()
df_pm_2016


# In[34]:


df_pm_2016 = df_pm_2016.rename(columns = {'Value': 'PM2.5'})
df_pm_2016


# ### load Ozone data

# In[35]:


ur_files = ddf.read_csv(r"Ozone.csv", dtype = {'CountyFIPS': 'object', 'Value': 'object'})
df_oz = ur_files.compute()
df_oz = df_oz.loc[:, ~df_oz.columns.str.contains('^Unnamed')]
df_oz.head()


# In[36]:


df_oz['Value'].unique()


# In[37]:


def changeValue(value):
    if value == 'No Data':
        value = None
    else:
        value = int(value)
    return value


# In[38]:


df_oz['Value'] = df_oz['Value'].apply(changeValue)
df_oz_2016 = df_oz[df_oz['Year'] == 2016]
df_oz_2016 = df_oz_2016.rename(columns = {'State': 'state', 'CountyFIPS': 'fips', 'County': 'county', 'Value': 'Ozone'})
df_oz_2016 = df_oz_2016.groupby(['state', 'fips', 'county']).mean()
df_oz_2016 = df_oz_2016.reset_index()
df_oz_2016


# #### Combine environmental factors

# In[39]:


df_air = df_pm_2016.merge(df_oz_2016, on = ['state','fips', 'Year', 'StateFIPS', 'county'], how = 'left')
df_air.head()


# In[40]:


df_air = df_air[['state', 'fips', 'county', 'PM2.5', 'Ozone']]
df_climate = df_climate2
df_env = df_climate.merge(df_air, on = ['fips'], how = 'left')
df_env


# In[41]:


df_humidity.columns


# In[42]:


df_humidity['county'].dtype


# In[43]:


def adjustCountyCode(county):
    return county.rjust(5, '0')


# In[44]:


df_humidity['county'] = df_humidity['county'].apply(adjustCountyCode)


# In[45]:


df_humidity.head()


# In[46]:


df_humidity = df_humidity.rename(columns = {'county': 'fips', 'closest_station_index': 'Humidity_Temp_Index'})


# In[47]:


df_humidity = df_humidity[['fips', 'Humidity_Temp_Index']]


# In[48]:


df_env = df_env.merge(df_humidity, on = 'fips', how = 'left')
df_env.head()


# ### checking missing climate data

# In[49]:


df_env['mean_temp'].isnull().sum()


# In[50]:


df_env['sea_level_pressure'].isnull().sum()


# In[51]:


df_env['station_pressure'].isnull().sum()


# In[52]:


df_env['visibility'].isnull().sum()


# In[53]:


df_env['wind_speed'].isnull().sum()


# In[54]:


df_env['precipitation'].isnull().sum()


# In[55]:


df_env['fog'].isnull().sum()


# In[56]:


df_env['rain'].isnull().sum()


# In[57]:


df_env['snow'].isnull().sum()


# In[58]:


df_env['hail'].isnull().sum()


# In[59]:


df_env['thunder'].isnull().sum()


# In[60]:


df_env['tornado'].isnull().sum()


# In[61]:


df_env['PM2.5'].isnull().sum()


# In[62]:


df_env['Ozone'].isnull().sum()


# In[63]:


df_env['Relative_Humidity'].isnull().sum()


# In[64]:


df_env['Heat_Index'].isnull().sum()


# In[65]:


df_env['Humidity_Temp_Index'].isnull().sum()


# In[66]:


df_env.columns


# In[67]:


# since almost half of the counties are missing see_level_pressure data, we choose to drop it off
df_env = df_env.drop(['sea_level_pressure'], axis = 1)


# In[68]:


df_sd_16 = pd.read_csv(r'SuicideRate_Imputed_2016.csv', dtype={"fips": str})
df_sd_16 = df_sd_16[['county', 'fips', 'SuicideDeathRate']]
df_sd_16


# In[69]:


df_env_sd = df_env.merge(df_sd_16, on = ['fips'], how = 'left')


# In[70]:


df_env_sd = df_env_sd.drop(['county_x'], axis = 1)  
df_env_sd = df_env_sd.rename(columns = {'county_y': 'county'})
df_env_sd


# In[71]:


df_env_sd = df_env_sd.drop(['state', 'county'], axis = 1)


# In[72]:


df_env_sd.head()


# In[73]:


df_env_sd = df_env_sd.dropna(how = 'any')


# In[74]:


correlation_matrix(df_env_sd)


# In[75]:


df_env_sd.to_csv('environment_suicide_2016.csv')


# In[76]:


weight = {'elevation': 0.36, 'mean_temp': -0.17, 'station_pressure': 0.20, 'visibility': 0.06, 'wind_speed': 0.12, 'precipitation': -0.12, 'Relative_Humidity': -0.19, 
          'Heat_Index': -0.18, 'fog': -0.09, 'rain': -0.13, 'snow': 0.05, 'hail': -0.04, 'tornado': -0.04, 'PM2.5': -0.29, 'Ozone': -0.12, 'Humidity_Temp_Index': -0.17}


# In[77]:


def findStateCode(code):
    return code[:2]


# In[78]:


df_env_sd['State Code'] = df_env_sd['fips'].apply(findStateCode)
df_env_sd.head()


# In[79]:


df_env_sd[['elevation', 'mean_temp', 'PM2.5', 'Humidity_Temp_Index', 'Relative_Humidity', 'SuicideDeathRate']].describe()


# In[80]:


df_env_sd = df_env_sd[df_env_sd['State Code'] == '08']
df_env_sd.head()


# In[81]:


df_env_sd[['elevation', 'mean_temp', 'PM2.5', 'Humidity_Temp_Index', 'Relative_Humidity','SuicideDeathRate']].describe()


# In[82]:


df_env['State Code'] = df_env['fips'].apply(findStateCode)
df_env= df_env[df_env['State Code'] == '08']
df_env.head()


# In[83]:


correlation_matrix(df_env_sd)


# In[84]:


weight = {'elevation': 0.61, 'mean_temp': -0.54, 'station_pressure': -0.59, 'visibility': -0.25, 'wind_speed': 0.05, 'precipitation': -0.05, 'Relative_Humidity': 0.01, 
          'Heat_Index': -0.53, 'fog': 0.19, 'rain': 0.10, 'snow': 0.50, 'hail': -0.17, 'thunder': 0.18, 'tornado': -0.01, 'PM2.5': -0.53, 'Ozone': -0.24, 'Humidity_Temp_Index': -0.43}


# In[85]:


df_env.head()


# In[86]:


variable = ['elevation', 'mean_temp', 'station_pressure', 'visibility', 'wind_speed', 'precipitation', 'Relative_Humidity', 'Heat_Index', 'fog',
            'rain', 'snow', 'hail', 'thunder', 'tornado', 'PM2.5', 'Ozone', 'Humidity_Temp_Index']
for col in variable:
    df_env[col] = df_env[col] * weight[col]


# In[87]:


df_env.head()


# ### find percentile for each variable

# In[88]:


df_env['elevation_percentile'] = (df_env['elevation'] - df_env['elevation'].min()) / (df_env['elevation'].max() - df_env['elevation'].min())
df_env['mean_temp_percentile'] = (df_env['mean_temp'] - df_env['mean_temp'].min()) / (df_env['mean_temp'].max() - df_env['mean_temp'].min())
df_env['station_pressure_percentile'] = (df_env['station_pressure'] - df_env['station_pressure'].min()) / (df_env['station_pressure'].max() - df_env['station_pressure'].min())
df_env['visibility_percentile'] = (df_env['visibility'] - df_env['visibility'].min()) / (df_env['visibility'].max() - df_env['visibility'].min())
df_env['wind_speed_percentile'] = (df_env['wind_speed'] - df_env['wind_speed'].min()) / (df_env['wind_speed'].max() - df_env['wind_speed'].min())
df_env['precipitation_percentile'] = (df_env['precipitation'] - df_env['precipitation'].min()) / (df_env['precipitation'].max() - df_env['precipitation'].min())
df_env['Relative_Humidity_percentile'] = (df_env['Relative_Humidity'] - df_env['Relative_Humidity'].min()) / ( df_env['Relative_Humidity'].max() -  df_env['Relative_Humidity'].min())
df_env['Heat_Index_percentile'] = (df_env['Heat_Index'] - df_env['Heat_Index'].min()) / ( df_env['Heat_Index'].max() -  df_env['Heat_Index'].min())
df_env['fog_percentile'] = (df_env['fog'] - df_env['fog'].min()) / (df_env['fog'].max() - df_env['fog'].min())
df_env['rain_percentile'] = (df_env['rain'] - df_env['rain'].min()) / (df_env['rain'].max() - df_env['rain'].min())
df_env['snow_percentile'] = (df_env['snow'] - df_env['snow'].min()) / (df_env['snow'].max() - df_env['snow'].min())
df_env['hail_percentile'] = (df_env['hail'] - df_env['hail'].min()) / (df_env['hail'].max() - df_env['hail'].min())
df_env['thunder_percentile'] = (df_env['thunder'] - df_env['thunder'].min()) / (df_env['thunder'].max() - df_env['thunder'].min())
df_env['tornado_percentile'] = (df_env['tornado'] - df_env['tornado'].min()) / (df_env['tornado'].max() - df_env['tornado'].min())
df_env['PM2.5_percentile'] = (df_env['PM2.5'] - df_env['PM2.5'].min()) / (df_env['PM2.5'].max() - df_env['PM2.5'].min())
df_env['Ozone_percentile'] = (df_env['Ozone'] - df_env['Ozone'].min()) / (df_env['Ozone'].max() - df_env['Ozone'].min())
df_env['Humidity_Temp_Index_percentile'] = (df_env['Humidity_Temp_Index'] - df_env['Humidity_Temp_Index'].min()) / (df_env['Humidity_Temp_Index'].max() - df_env['Humidity_Temp_Index'].min())


# #### give a score to climate factors and air quality factors

# In[90]:


df_env['ClimateScore'] = df_env['elevation_percentile'] + df_env['mean_temp_percentile'] + df_env['station_pressure_percentile'] + df_env['visibility_percentile'] + df_env['wind_speed_percentile'] + df_env['precipitation_percentile']+ df_env['Relative_Humidity_percentile']+ df_env['Heat_Index_percentile'] + df_env['fog_percentile'] + df_env['rain_percentile'] + df_env['snow_percentile'] + df_env['hail_percentile'] + df_env['thunder_percentile'] + df_env['tornado_percentile'] + df_env['Humidity_Temp_Index_percentile']  


# In[91]:


df_env['AirQualityScore'] = df_env['PM2.5_percentile'] + df_env['Ozone_percentile']


# #### give a percentile to each category

# In[92]:


df_env['ClimateScore_percentile'] = (df_env['ClimateScore'] - df_env['ClimateScore'].min()) / (df_env['ClimateScore'].max() - df_env['ClimateScore'].min())
df_env['AirQuality_percentile'] = (df_env['AirQualityScore'] - df_env['AirQualityScore'].min()) / (df_env['AirQualityScore'].max() - df_env['AirQualityScore'].min())


# #### sum the percentile of each category and assign the sum to be environment score

# In[93]:


df_env['EnvironmentScore'] = df_env['ClimateScore_percentile'] + df_env['AirQuality_percentile']


# #### MEDI is the percentile of environmental score

# In[94]:


df_env['MEDI'] = (df_env['EnvironmentScore'] - df_env['EnvironmentScore'].min()) / (df_env['EnvironmentScore'].max() - df_env['EnvironmentScore'].min())


# In[95]:


df_env.head()


# In[96]:


df_env['MEDI'].describe()


# In[97]:


df_env['county'] = df_env['county'] + ',' + df_env['state']


# In[98]:


df_env2 = df_env.dropna(how = 'any')
df_env2.shape[0]


# # Plot out MEDI

# In[99]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_env, geojson=counties, locations='fips', color='MEDI',
                           color_continuous_scale="rainbow",
                           range_color=(0, 0.7),
                           scope="usa",
                           hover_name="county",
                           labels={'MEDI':'MEDI_Colorado'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[100]:


df_env_sd.head()


# In[101]:


df_env_sd['SuicideDeathRate'].describe()


# In[102]:


df_env_sd['elevation'].describe()


# In[103]:


import pandas as pd
my_data = df_env_sd['SuicideDeathRate']
my_data.hist()


# In[104]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_env_sd, geojson=counties, locations='fips', color='SuicideDeathRate',
                           color_continuous_scale="rainbow",
                           range_color=(0, 35),
                           scope="usa",
                           labels={'SuicideDeathRate':'SuicideRate_Colorado'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[107]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_env_sd, geojson=counties, locations='fips', color='elevation',
                           color_continuous_scale="rainbow",
                           range_color=(0, 2500),
                           scope="usa",
                           labels={'elevation':'Elevation_Colorado'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[108]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_env_sd, geojson=counties, locations='fips', color='mean_temp',
                           color_continuous_scale="rainbow",
                           range_color=(20, 60),
                           scope="usa",
                           labels={'mean_temp':'Mean_Temp_Colorado'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[109]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_env_sd, geojson=counties, locations='fips', color='PM2.5',
                           color_continuous_scale="rainbow",
                           range_color=(0, 10),
                           scope="usa",
                           labels={'PM2.5':'PM2.5_Colorado'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[111]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_env_sd, geojson=counties, locations='fips', color='Relative_Humidity',
                           color_continuous_scale="rainbow",
                           range_color=(20, 60),
                           scope="usa",
                           labels={'Relative_Humidity':'Relative_Humidity_Colorado'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Combining SEDI and MEDI

# In[110]:


df_SEDI = pd.read_csv(r'SEDI_2016.csv',dtype={"fips": str})
df_SEDI


# In[111]:


df_SEDI = df_SEDI[['fips', 'SEDI']]
df_SEDI


# In[ ]:





# In[112]:


df_MEDI = df_env[['county', 'fips', 'MEDI']]
df_MEDI


# In[113]:


df_final_index = df_SEDI.merge(df_MEDI, on = 'fips', how = 'left')
df_final_index


# In[114]:


df_final_index['SumOfTwoIndex'] = df_final_index['SEDI'] + df_final_index['MEDI']


# In[115]:


df_final_index['MEDI + SEDI'] = (df_final_index['SumOfTwoIndex'] - df_final_index['SumOfTwoIndex'].min())/(df_final_index['SumOfTwoIndex'].max() - df_final_index['SumOfTwoIndex'].min())
df_final_index


# In[116]:


df_final_index['MEDI + SEDI'].describe()


# In[117]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_final_index, geojson=counties, locations='fips', color='MEDI + SEDI',
                           color_continuous_scale="rainbow",
                           range_color=(0.2, 0.8),
                           scope="usa",
                           hover_name="county",
                           labels={'MEDI + SEDI':'MEDI + SEDI'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# ### plot out the SVI 

# In[118]:


df_percentile.columns


# In[119]:


df_percentile['county'] = df_percentile['county'] + ',' + df_percentile['state']


# In[120]:


df_percentile = df_percentile.rename(columns = {'percentile_rank_socioeconomic_theme': 'socioeconomic_theme', 'percentile_rank_household_comp_disability_theme': 'household_comp_disability_theme',
                                               'percentile_rank_minority_status_and_language_theme': 'minority_status_and_language_theme', 'percentile_rank_housing_and_transportation': 'housing_and_transportation',
                                               'percentile_rank_social_vulnerability': 'SVI'})


# In[121]:


df_percentile.head()


# In[122]:


df_percentile.shape[0]


# In[123]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_percentile, geojson=counties, locations='fips', color='SVI',
                           color_continuous_scale="rainbow",
                           range_color=(0, 1),
                           scope="usa",
                           hover_name="county",
                           labels={'MEDI':'MEDI'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# #### Combine SVI and MEDI

# In[124]:


df_MEDI = df_env[['state', 'county', 'fips', 'MEDI']]
df_MEDI


# In[125]:


df_percentile


# In[126]:


df_final = df_percentile.merge(df_MEDI, on = ['state', 'county', 'fips'], how = 'left')


# In[127]:


df_final


# In[128]:


df_final['SVI + MEDI'] = df_final['SVI'] + df_final['MEDI']


# In[129]:


df_final['Updated Index'] = (df_final['SVI + MEDI'] - df_final['SVI + MEDI'].min()) / (df_final['SVI + MEDI'].max() - df_final['SVI + MEDI'].min())


# In[130]:


df_final2 = df_final.dropna(how = 'any')


# In[131]:


df_final2.shape[0]


# In[136]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_final, geojson=counties, locations='fips', color='Updated Index',
                           color_continuous_scale="rainbow",
                           range_color=(0.2, 0.8),
                           scope="usa",
                           hover_name="county",
                           labels={'Updated Index':'SVI + MEDI'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[137]:


df_final


# ### Evaluation and Comparison

# In[138]:


df_index = df_final[['state', 'county', 'fips', 'SVI', 'MEDI', 'Updated Index']]


# In[139]:


df_index2 = df_index.merge(df_final_index, on = ['fips', 'county', 'MEDI'], how = 'left')


# In[140]:


df_index2


# In[141]:


df_sd = pd.read_csv(r'SuicideRate_Imputed_2017.csv',dtype={"fips": str})


# In[142]:


df_sd = df_sd[['county', 'fips', 'SuicideDeathRate']]


# In[143]:


df_sd


# In[144]:


df_index3 = df_index2.merge(df_sd, on = ['fips'], how = 'left')
df_index3


# In[145]:


df_index3 = df_index3[['state', 'county_y', 'fips', 'SVI', 'MEDI', 'Updated Index', 'SuicideDeathRate','SEDI','MEDI + SEDI']]
df_index3 = df_index3.rename(columns = {'county_y': 'county'})
df_index3


# In[146]:


df_index3['Suicide_Rate_Percentile'] = (df_index3['SuicideDeathRate'] - df_index3['SuicideDeathRate'].min()) / (df_index3['SuicideDeathRate'].max() - df_index3['SuicideDeathRate'].min())


# In[147]:


df_index3


# In[161]:


df_index3 = df_index3[df_index3['state'] == 'Colorado']


# In[162]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_index3, geojson=counties, locations='fips', color='SuicideDeathRate',
                           color_continuous_scale="rainbow",
                           range_color=(2, 36),
                           scope="usa",
                           hover_name="county",
                           labels={'Updated Index':'Updated Index'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[164]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_index3, geojson=counties, locations='fips', color='SVI',
                           color_continuous_scale="rainbow",
                           range_color=(0.2, 0.8),
                           scope="usa",
                           hover_name="county",
                           labels={'SVI':'SVI'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# ### Use Mean absolute error/Mean squared error/Root-Mean-Square Error to check the model performance

# ####  SVI and Suicide Rate

# In[165]:


df_index4 = df_index3[['state', 'county', 'fips', 'SVI', 'MEDI', 'Updated Index', 'SuicideDeathRate', 'SEDI','MEDI + SEDI', 'Suicide_Rate_Percentile']]


# In[166]:


correlation_matrix(df_index4)


# In[167]:


df_index4 = df_index3.dropna(how = 'any')


# In[168]:


import seaborn as sn
import plotly.express as px
sn.lmplot('SVI', 'Suicide_Rate_Percentile', data = df_index4)


# In[169]:


sn.lmplot('MEDI', 'Suicide_Rate_Percentile', data = df_index4)


# In[170]:


sn.lmplot('Updated Index', 'Suicide_Rate_Percentile', data = df_index4)


# In[171]:


sn.lmplot('SEDI', 'Suicide_Rate_Percentile', data = df_index4)


# In[172]:


sn.lmplot('MEDI + SEDI', 'Suicide_Rate_Percentile', data = df_index4)


# In[173]:


import sklearn
from sklearn.linear_model import LinearRegression
# Create a LinearRegression Object
y = df_index4['Suicide_Rate_Percentile']
x_index = df_index4.drop(['county', 'state', 'fips', 'SuicideDeathRate','Suicide_Rate_Percentile'], axis =1)
lreg = LinearRegression()
lreg.fit(x_index,y)
# Set a DataFrame from the Features
coeff_index = DataFrame(x_index.columns)
coeff_index.columns = ['Features']

# Set a new column lining up the coefficients from the linear regression
coeff_index["Coefficient Estimate"] = pd.Series(lreg.coef_)

# Show
coeff_index


# In[174]:


y = df_index4['SVI'].values.reshape(-1, 1)
X = df_index4['SuicideDeathRate'].values.reshape(-1, 1)
from sklearn.model_selection import train_test_split
SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = SEED)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
intercept_SVI = regressor.intercept_
coefficient_SVI = regressor.coef_
y_pred = regressor.predict(X_test)
df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
df_preds.head()


# In[175]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_SVI = mean_absolute_error(y_test, y_pred)
mse_SVI = mean_squared_error(y_test, y_pred)
rmse_SVI = np.sqrt(mse_SVI)
print(f'Mean absolute error: {mae_SVI:.2f}')
print(f'Mean squared error: {mse_SVI:.2f}')
print(f'Root mean squared error: {rmse_SVI:.2f}')


# In[176]:


coefficient_SVI[0][0],intercept_SVI[0]


# #### MEDI with suicide rate

# In[177]:


y = df_index4['MEDI'].values.reshape(-1, 1)
X = df_index4['SuicideDeathRate'].values.reshape(-1, 1)
from sklearn.model_selection import train_test_split
SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = SEED)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
intercept_MEDI = regressor.intercept_
coefficient_MEDI = regressor.coef_
y_pred = regressor.predict(X_test)
df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
df_preds.head()


# In[178]:


coefficient_MEDI[0][0],intercept_MEDI[0]


# In[179]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_MEDI = mean_absolute_error(y_test, y_pred)
mse_MEDI = mean_squared_error(y_test, y_pred)
rmse_MEDI = np.sqrt(mse_MEDI)
print(f'Mean absolute error: {mae_MEDI:.2f}')
print(f'Mean squared error: {mse_MEDI:.2f}')
print(f'Root mean squared error: {rmse_MEDI:.2f}')


# #### Total Index with suicide rate

# In[180]:


y = df_index4['Updated Index'].values.reshape(-1, 1)
X = df_index4['SuicideDeathRate'].values.reshape(-1, 1)
from sklearn.model_selection import train_test_split
SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = SEED)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
intercept = regressor.intercept_
coefficient = regressor.coef_
y_pred = regressor.predict(X_test)
df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
df_preds.head()


# In[181]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_Total_Index = mean_absolute_error(y_test, y_pred)
mse_Total_Index = mean_squared_error(y_test, y_pred)
rmse_Total_Index = np.sqrt(mse_Total_Index)
print(f'Mean absolute error: {mae_Total_Index:.2f}')
print(f'Mean squared error: {mse_Total_Index:.2f}')
print(f'Root mean squared error: {rmse_Total_Index:.2f}')


# #### SEDI with suicide rate

# In[182]:


y = df_index4['SEDI'].values.reshape(-1, 1)
X = df_index4['SuicideDeathRate'].values.reshape(-1, 1)
from sklearn.model_selection import train_test_split
SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = SEED)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
intercept = regressor.intercept_
coefficient = regressor.coef_
y_pred = regressor.predict(X_test)
df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
df_preds.head()


# In[183]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_Total_Index = mean_absolute_error(y_test, y_pred)
mse_Total_Index = mean_squared_error(y_test, y_pred)
rmse_Total_Index = np.sqrt(mse_Total_Index)
print(f'Mean absolute error: {mae_Total_Index:.2f}')
print(f'Mean squared error: {mse_Total_Index:.2f}')
print(f'Root mean squared error: {rmse_Total_Index:.2f}')


# #### MEDI + SEDI with suicide rate

# In[184]:


y = df_index4['MEDI + SEDI'].values.reshape(-1, 1)
X = df_index4['SuicideDeathRate'].values.reshape(-1, 1)
from sklearn.model_selection import train_test_split
SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = SEED)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
intercept = regressor.intercept_
coefficient = regressor.coef_
y_pred = regressor.predict(X_test)
df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
df_preds.head()


# In[187]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_Total_Index = mean_absolute_error(y_test, y_pred)
mse_Total_Index = mean_squared_error(y_test, y_pred)
rmse_Total_Index = np.sqrt(mse_Total_Index)
print(f'Mean absolute error: {mae_Total_Index:.2f}')
print(f'Mean squared error: {mse_Total_Index:.2f}')
print(f'Root mean squared error: {rmse_Total_Index:.2f}')


# In[188]:


df_index3


# In[189]:


df_index5 = df_index3.drop(['SuicideDeathRate'], axis = 1)
df_index5


# In[190]:


df_sd_16


# In[191]:


df_index_current = df_index5.merge(df_sd_16, on = ['county', 'fips'], how = 'left')


# In[192]:


df_index_current


# In[193]:


df_index3['Year'] = '2016'


# In[162]:


df_index_current['Year'] = '2016'


# In[163]:


df_index3.to_csv('All_Index_Next_Year_2016.csv')


# In[164]:


df_index_current.to_csv('All_Index_Current_Year_2016.csv')

