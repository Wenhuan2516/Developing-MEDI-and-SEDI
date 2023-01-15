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

# In[4]:


ur_files = ddf.read_csv(r"Temperature/US_counties_weather_2016.csv", dtype={'eightieth_percentile_income': 'float64','fips': 'object','fog': 'float64','hail': 'float64',
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


# In[5]:


cols = df_svi.columns


# In[6]:


for col in cols:
    if 'income' in col:
        print(col)


# In[7]:


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


df_percentile = df_svi_county[['state','county','fips','lat','lon','percentile_rank_below_poverty','percentile_rank_unemployed',
                      'percentile_rank_per_capita_income','percentile_rank_no_highschool_diploma','percentile_rank_socioeconomic_theme',
                      'percentile_rank_age_65_and_older','percentile_rank_age_17_and_younger','percentile_rank_disabled','percentile_rank_single_parent_households',
                      'percentile_rank_household_comp_disability_theme','percentile_rank_minorities','percentile_rank_limited_english_abilities',
                      'percentile_rank_minority_status_and_language_theme','percentile_rank_multi_unit_housing','percentile_rank_mobile_homes',
                      'percentile_rank_overcrowding','percentile_rank_no_vehicle','percentile_rank_institutionalized_in_group_quarters',
                      'percentile_rank_housing_and_transportation','percentile_rank_social_vulnerability']]


# In[15]:


df_percentile


# In[16]:


df_sample = df_percentile[df_percentile['county'] == 'Adams']
df_sample


# In[17]:


df_sample2 = df_sample[df_sample['state'] == 'Washington']


# In[18]:


df_result = df_sample2[['percentile_rank_socioeconomic_theme', 'percentile_rank_household_comp_disability_theme', 'percentile_rank_minority_status_and_language_theme',
                        'percentile_rank_housing_and_transportation', 'percentile_rank_social_vulnerability']]
df_result


# ### Find the climate data part

# In[31]:


df_temp_2016 = df_svi[['state', 'county', 'fips','station_id', 'station_name', 'lat', 'lon', 'date', 'mean_temp', 
                        'min_temp', 'max_temp', 'dewpoint', 'sea_level_pressure', 'visibility', 'wind_speed', 'wind_gust', 'precipitation',
                       'precip_flag', 'fog', 'rain', 'snow', 'hail', 'thunder', 'tornado']]
df_temp_2016.head()


# In[32]:


df_temp_2016.shape


# In[33]:


df_temp_2016['dewpoint'].isnull().sum()


# In[34]:


57729/1114026


# In[35]:


1114026/366


# In[36]:


days = df_temp_2016['date'].unique().tolist()
len(days)


# In[37]:


df_temp_county = df_temp_2016.groupby(['state', 'county', 'fips']).mean()
df_temp_county = df_temp_county.reset_index()
df_temp_county.head()


# In[38]:


df_temp_county['dewpoint'].isnull().sum()


# In[39]:


df_temp_county.shape


# In[40]:


158/3142


# In[41]:


correlation_matrix(df_temp_county)


# ## Calculate relative humidity

# In[42]:


def Convertion_F_to_C(T):
    return (T - 32)*(5/9)


# In[43]:


import math


# In[44]:


def RH_Calculator(T, TD):
    T = Convertion_F_to_C(T)
    TD = Convertion_F_to_C(TD)
    return 100*(math.exp((17.625*TD)/(243.04+TD))/math.exp((17.625*T)/(243.04+T)))


# In[45]:


df_temp_2016['Relative_Humidity'] = df_temp_2016.apply(lambda x: RH_Calculator(x['mean_temp'], x['dewpoint']), axis=1)


# In[46]:


df_temp_2016


# ## Calculate the heat index

# In[47]:


abs(9-12)


# In[48]:


def heat_index_calculation(T, RH):
    if RH < 13 and T > 80 and T < 112:
        return ((13-RH)/4)* math.sqrt((17- abs(T-95))/17)
    elif RH > 85 and T > 80 and T < 87:
        return ((RH - 85)/10)*((87-T)/5)
    elif T < 80:
        return 0.5 * (T + 61.0 + ((T-68.0)*1.2) + (RH*0.094))
    else:
        return -42.379 + 2.04901523*T + 10.14333127*RH - .22475541*T*RH - .00683783*T*T - .05481717*RH*RH + .00122874*T*T*RH + .00085282*T*RH*RH - .00000199*T*T*RH*RH


# In[49]:


value = heat_index_calculation(46.8,72.42)
value


# In[50]:


df_temp_2016['Heat_Index'] = df_temp_2016.apply(lambda x: heat_index_calculation(x['mean_temp'], x['Relative_Humidity']), axis=1)


# In[51]:


df_temp_2016


# In[52]:


df_temp_2016.to_csv('daily_temp_2016.csv')


# In[41]:


ur_files = ddf.read_csv(r"Humidity_Temp_Index/imputed_humid_index_2016.csv", dtype = {'county': 'object'})
df_humidity = ur_files.compute()
df_humidity = df_humidity.loc[:, ~df_humidity.columns.str.contains('^Unnamed')]
df_humidity.head()


# #### From the above graph, we can see that max temp/min temp/dewpoint are highly correlated with mean temp, max Wind speed and max wind gust are highly correlated with wind values, so we can drop max temp, min temp, max wind speed, and max wind gust.

# In[22]:


df_temp_county3 = df_temp_county.drop(['min_temp', 'max_temp', 'wind_gust', 'dewpoint'], axis=1)
correlation_matrix(df_temp_county3)


# ### upload PM2.5 Data

# In[23]:


ur_files = ddf.read_csv(r"PM2.5.csv", dtype = {'CountyFIPS': 'object'})
df_pm = ur_files.compute()
df_pm = df_pm.loc[:, ~df_pm.columns.str.contains('^Unnamed')]
df_pm.head()


# In[24]:


df_pm_2016 = df_pm[df_pm['Year'] == 2016]
df_pm_2016 = df_pm_2016.rename(columns = {'State': 'state', 'CountyFIPS': 'fips', 'County': 'county'})
df_pm_2016 = df_pm_2016.groupby(['state', 'fips', 'county']).mean()
df_pm_2016 = df_pm_2016.reset_index()
df_pm_2016


# In[25]:


df_pm_2016 = df_pm_2016.rename(columns = {'Value': 'PM2.5'})
df_pm_2016


# ### load Ozone data

# In[26]:


ur_files = ddf.read_csv(r"Ozone.csv", dtype = {'CountyFIPS': 'object', 'Value': 'object'})
df_oz = ur_files.compute()
df_oz = df_oz.loc[:, ~df_oz.columns.str.contains('^Unnamed')]
df_oz.head()


# In[27]:


df_oz['Value'].unique()


# In[28]:


def changeValue(value):
    if value == 'No Data':
        value = None
    else:
        value = int(value)
    return value


# In[29]:


df_oz['Value'] = df_oz['Value'].apply(changeValue)
df_oz_2016 = df_oz[df_oz['Year'] == 2016]
df_oz_2016 = df_oz_2016.rename(columns = {'State': 'state', 'CountyFIPS': 'fips', 'County': 'county', 'Value': 'Ozone'})
df_oz_2016 = df_oz_2016.groupby(['state', 'fips', 'county']).mean()
df_oz_2016 = df_oz_2016.reset_index()
df_oz_2016


# #### Combine environmental factors

# In[30]:


df_air = df_pm_2016.merge(df_oz_2016, on = ['state','fips', 'Year', 'StateFIPS', 'county'], how = 'left')
df_air.head()


# In[31]:


df_air = df_air[['state', 'fips', 'county', 'PM2.5', 'Ozone']]
df_climate = df_temp_county3
df_env = df_climate.merge(df_air, on = ['state', 'fips', 'county'], how = 'left')
df_env


# ### checking missing climate data

# In[32]:


df_env['mean_temp'].isnull().sum()


# In[33]:


df_env['sea_level_pressure'].isnull().sum()


# In[34]:


df_env['visibility'].isnull().sum()


# In[35]:


df_env['wind_speed'].isnull().sum()


# In[36]:


df_env['precipitation'].isnull().sum()


# In[37]:


df_env['fog'].isnull().sum()


# In[38]:


df_env['rain'].isnull().sum()


# In[39]:


df_env['snow'].isnull().sum()


# In[40]:


df_env['hail'].isnull().sum()


# In[41]:


df_env['thunder'].isnull().sum()


# In[42]:


df_env['tornado'].isnull().sum()


# In[43]:


df_env['PM2.5'].isnull().sum()


# In[44]:


df_env['Ozone'].isnull().sum()


# ### Dropping variables when the variable missed too much data and assign the average mean value of adjacent counties to the missing county

# In[45]:


# since almost half of the counties are missing see_level_pressure data, we choose to drop it off
df_env = df_env.drop(['sea_level_pressure'], axis = 1)


# In[46]:


county_adjacency = pd.read_csv('county_adjacency.csv', dtype = {'fips': str})
county_adjacency = county_adjacency[['county', 'fips', 'Neighbors', 'Neighbor Code']]
county_adjacency


# In[47]:


df_env_new = df_env.merge(county_adjacency, on = 'fips', how = 'left')
df_env_new = df_env_new.drop(['county_x'], axis = 1)
df_env_new = df_env_new.rename(columns = {'county_y': 'county'})
df_env_new.head()


# ### fill up the visibility data

# In[48]:


list_visibility = df_env_new['visibility'].tolist()
list_fips = df_env_new['fips'].tolist()


# In[49]:


dict1 = {}
for i, j in zip(list_fips, list_visibility):
    dict1[i] = j


# In[50]:


def findMissingData(value, fips, neighbor):
    if pd.isna(value) == True:
        neighborList = neighbor.split(', ')
        rateSum = 0
        k = 0
        for code in neighborList:
            if code in dict1:
                if pd.isna(dict1[code]) == False:
                    k = k + 1
                    rateSum = rateSum + dict1[code]
        if k != 0: 
            mean_rate = rateSum/k
            return mean_rate
    else:
        return value


# In[51]:


df_env_new['visibility'] = df_env_new.apply(lambda x: findMissingData(x['visibility'], x['fips'], x['Neighbor Code']), axis = 1)


# In[52]:


county_adjacency.head()


# In[53]:


df_sd_16 = pd.read_csv(r'SuicideRate_Imputed_2016.csv', dtype={"fips": str})
df_sd_16 = df_sd_16[['county', 'fips', 'SuicideDeathRate']]
df_sd_16


# In[54]:


df_env_sd = df_env.merge(df_sd_16, on = ['fips'], how = 'left')


# In[55]:


df_env_sd = df_env_sd.drop(['county_x'], axis = 1)  
df_env_sd = df_env_sd.rename(columns = {'county_y': 'county'})
df_env_sd


# In[56]:


df_env_sd = df_env_sd.dropna(how = 'any')


# In[57]:


correlation_matrix(df_env_sd)


# In[58]:


weight = {'mean_temp': -0.16, 'visibility': 0.07, 'wind_speed': 0.11, 'precipitation': -0.11, 'fog': -0.09, 'rain': -0.12, 'snow': 0.05,
         'hail': -0.03, 'thunder': 0.01, 'tornado': -0.03, 'PM2.5': -0.29, 'Ozone': -0.12}


# In[59]:


variable = ['mean_temp', 'visibility', 'wind_speed', 'precipitation', 'fog',
            'rain', 'snow', 'hail', 'thunder', 'tornado', 'PM2.5', 'Ozone']
for col in variable:
    df_env[col] = df_env[col] * weight[col]


# ### find percentile for each variable

# In[60]:


df_env['mean_temp_percentile'] = (df_env['mean_temp'] - df_env['mean_temp'].min()) / (df_env['mean_temp'].max() - df_env['mean_temp'].min())
df_env['visibility_percentile'] = (df_env['visibility'] - df_env['visibility'].min()) / (df_env['visibility'].max() - df_env['visibility'].min())
df_env['wind_speed_percentile'] = (df_env['wind_speed'] - df_env['wind_speed'].min()) / (df_env['wind_speed'].max() - df_env['wind_speed'].min())
df_env['precipitation_percentile'] = (df_env['precipitation'] - df_env['precipitation'].min()) / (df_env['precipitation'].max() - df_env['precipitation'].min())
df_env['fog_percentile'] = (df_env['fog'] - df_env['fog'].min()) / (df_env['fog'].max() - df_env['fog'].min())
df_env['rain_percentile'] = (df_env['rain'] - df_env['rain'].min()) / (df_env['rain'].max() - df_env['rain'].min())
df_env['snow_percentile'] = (df_env['snow'] - df_env['snow'].min()) / (df_env['snow'].max() - df_env['snow'].min())
df_env['hail_percentile'] = (df_env['hail'] - df_env['hail'].min()) / (df_env['hail'].max() - df_env['hail'].min())
df_env['thunder_percentile'] = (df_env['thunder'] - df_env['thunder'].min()) / ( df_env['thunder'].max() -  df_env['thunder'].min())
df_env['tornado_percentile'] = (df_env['tornado'] - df_env['tornado'].min()) / (df_env['tornado'].max() - df_env['tornado'].min())
df_env['PM2.5_percentile'] = (df_env['PM2.5'] - df_env['PM2.5'].min()) / (df_env['PM2.5'].max() - df_env['PM2.5'].min())
df_env['Ozone_percentile'] = (df_env['Ozone'] - df_env['Ozone'].min()) / (df_env['Ozone'].max() - df_env['Ozone'].min())


# #### give a score to climate factors and air quality factors

# In[61]:


df_env['ClimateScore'] = df_env['mean_temp_percentile'] + df_env['visibility_percentile'] + df_env['wind_speed_percentile'] + df_env['precipitation_percentile']+ df_env['fog_percentile'] + df_env['rain_percentile'] + df_env['snow_percentile'] + df_env['hail_percentile'] + df_env['thunder_percentile'] + df_env['tornado_percentile']


# In[62]:


df_env['AirQualityScore'] = df_env['PM2.5_percentile'] + df_env['Ozone_percentile']


# #### give a percentile to each category

# In[63]:


df_env['ClimateScore_percentile'] = (df_env['ClimateScore'] - df_env['ClimateScore'].min()) / (df_env['ClimateScore'].max() - df_env['ClimateScore'].min())
df_env['AirQuality_percentile'] = (df_env['AirQualityScore'] - df_env['AirQualityScore'].min()) / (df_env['AirQualityScore'].max() - df_env['AirQualityScore'].min())


# #### sum the percentile of each category and assign the sum to be environment score

# In[64]:


df_env['EnvironmentScore'] = df_env['ClimateScore_percentile'] + df_env['AirQuality_percentile']


# #### MEDI is the percentile of environmental score

# In[65]:


df_env['MEDI'] = (df_env['EnvironmentScore'] - df_env['EnvironmentScore'].min()) / (df_env['EnvironmentScore'].max() - df_env['EnvironmentScore'].min())


# In[66]:


df_env.head()


# In[67]:


df_env['MEDI'].describe()


# In[68]:


df_env['county'] = df_env['county'] + ',' + df_env['state']


# In[69]:


df_env2 = df_env.dropna(how = 'any')
df_env2.shape[0]


# ### Plot out MEDI

# In[70]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_env, geojson=counties, locations='fips', color='MEDI',
                           color_continuous_scale="rainbow",
                           range_color=(0, 1),
                           scope="usa",
                           hover_name="county",
                           labels={'MEDI':'MEDI'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# ### Combining SEDI and MEDI

# In[71]:


df_SEDI = pd.read_csv(r'SEDI_2016.csv',dtype={"fips": str})
df_SEDI


# In[72]:


df_SEDI = df_SEDI[['fips', 'SEDI']]
df_SEDI


# In[73]:


df_MEDI = df_env[['county', 'fips', 'MEDI']]
df_MEDI


# In[74]:


df_final_index = df_SEDI.merge(df_MEDI, on = 'fips', how = 'left')
df_final_index


# In[75]:


df_final_index['SumOfTwoIndex'] = df_final_index['SEDI'] + df_final_index['MEDI']


# In[76]:


df_final_index['MEDI + SEDI'] = (df_final_index['SumOfTwoIndex'] - df_final_index['SumOfTwoIndex'].min())/(df_final_index['SumOfTwoIndex'].max() - df_final_index['SumOfTwoIndex'].min())
df_final_index


# In[77]:


df_final_index['MEDI + SEDI'].describe()


# In[78]:


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

# In[79]:


df_percentile.columns


# In[80]:


df_percentile['county'] = df_percentile['county'] + ',' + df_percentile['state']


# In[81]:


df_percentile = df_percentile.rename(columns = {'percentile_rank_socioeconomic_theme': 'socioeconomic_theme', 'percentile_rank_household_comp_disability_theme': 'household_comp_disability_theme',
                                               'percentile_rank_minority_status_and_language_theme': 'minority_status_and_language_theme', 'percentile_rank_housing_and_transportation': 'housing_and_transportation',
                                               'percentile_rank_social_vulnerability': 'SVI'})


# In[82]:


df_percentile.head()


# In[83]:


df_percentile.shape[0]


# In[84]:


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

# In[85]:


df_MEDI = df_env[['state', 'county', 'fips', 'MEDI']]
df_MEDI


# In[86]:


df_percentile


# In[87]:


df_final = df_percentile.merge(df_MEDI, on = ['state', 'county', 'fips'], how = 'left')


# In[88]:


df_final


# In[89]:


df_final['SVI + MEDI'] = df_final['SVI'] + df_final['MEDI']


# In[90]:


df_final['Updated Index'] = (df_final['SVI + MEDI'] - df_final['SVI + MEDI'].min()) / (df_final['SVI + MEDI'].max() - df_final['SVI + MEDI'].min())


# In[91]:


df_final2 = df_final.dropna(how = 'any')


# In[92]:


df_final2.shape[0]


# In[93]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_final, geojson=counties, locations='fips', color='Updated Index',
                           color_continuous_scale="rainbow",
                           range_color=(0, 1),
                           scope="usa",
                           hover_name="county",
                           labels={'Updated Index':'Updated Index'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[94]:


df_final


# ### Evaluation and Comparison

# In[95]:


df_index = df_final[['state', 'county', 'fips', 'SVI', 'MEDI', 'Updated Index']]


# In[96]:


df_index2 = df_index.merge(df_final_index, on = ['fips', 'county', 'MEDI'], how = 'left')


# In[97]:


df_index2


# In[98]:


df_sd = pd.read_csv(r'SuicideRate_Imputed_2017.csv',dtype={"fips": str})


# In[99]:


df_sd = df_sd[['county', 'fips', 'SuicideDeathRate']]


# In[100]:


df_sd


# In[128]:


df_index3 = df_index2.merge(df_sd, on = ['fips'], how = 'left')
df_index3


# In[129]:


df_index3 = df_index3[['state', 'county_y', 'fips', 'SVI', 'MEDI', 'Updated Index', 'SuicideDeathRate','SEDI','MEDI + SEDI']]
df_index3 = df_index3.rename(columns = {'county_y': 'county'})
df_index3


# In[130]:


df_index3['Suicide_Rate_Percentile'] = (df_index3['SuicideDeathRate'] - df_index3['SuicideDeathRate'].min()) / (df_index3['SuicideDeathRate'].max() - df_index3['SuicideDeathRate'].min())


# In[131]:


df_index3


# ### Use Mean absolute error/Mean squared error/Root-Mean-Square Error to check the model performance

# ####  SVI and Suicide Rate

# In[132]:


df_index4 = df_index3[['state', 'county', 'fips', 'SVI', 'MEDI', 'Updated Index', 'SuicideDeathRate', 'SEDI','MEDI + SEDI', 'Suicide_Rate_Percentile']]


# In[133]:


correlation_matrix(df_index4)


# In[134]:


df_index4 = df_index3.dropna(how = 'any')


# In[135]:


import seaborn as sn
import plotly.express as px
sn.lmplot('SVI', 'Suicide_Rate_Percentile', data = df_index4)


# In[136]:


sn.lmplot('MEDI', 'Suicide_Rate_Percentile', data = df_index4)


# In[137]:


sn.lmplot('Updated Index', 'Suicide_Rate_Percentile', data = df_index4)


# In[138]:


sn.lmplot('SEDI', 'Suicide_Rate_Percentile', data = df_index4)


# In[139]:


sn.lmplot('MEDI + SEDI', 'Suicide_Rate_Percentile', data = df_index4)


# In[140]:


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


# In[141]:


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


# In[142]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_SVI = mean_absolute_error(y_test, y_pred)
mse_SVI = mean_squared_error(y_test, y_pred)
rmse_SVI = np.sqrt(mse_SVI)
print(f'Mean absolute error: {mae_SVI:.2f}')
print(f'Mean squared error: {mse_SVI:.2f}')
print(f'Root mean squared error: {rmse_SVI:.2f}')


# In[143]:


coefficient_SVI[0][0],intercept_SVI[0]


# #### MEDI with suicide rate

# In[144]:


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


# In[145]:


coefficient_MEDI[0][0],intercept_MEDI[0]


# In[146]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_MEDI = mean_absolute_error(y_test, y_pred)
mse_MEDI = mean_squared_error(y_test, y_pred)
rmse_MEDI = np.sqrt(mse_MEDI)
print(f'Mean absolute error: {mae_MEDI:.2f}')
print(f'Mean squared error: {mse_MEDI:.2f}')
print(f'Root mean squared error: {rmse_MEDI:.2f}')


# #### Total Index with suicide rate

# In[147]:


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


# In[148]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_Total_Index = mean_absolute_error(y_test, y_pred)
mse_Total_Index = mean_squared_error(y_test, y_pred)
rmse_Total_Index = np.sqrt(mse_Total_Index)
print(f'Mean absolute error: {mae_Total_Index:.2f}')
print(f'Mean squared error: {mse_Total_Index:.2f}')
print(f'Root mean squared error: {rmse_Total_Index:.2f}')


# #### SEDI with suicide rate

# In[149]:


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


# In[150]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_Total_Index = mean_absolute_error(y_test, y_pred)
mse_Total_Index = mean_squared_error(y_test, y_pred)
rmse_Total_Index = np.sqrt(mse_Total_Index)
print(f'Mean absolute error: {mae_Total_Index:.2f}')
print(f'Mean squared error: {mse_Total_Index:.2f}')
print(f'Root mean squared error: {rmse_Total_Index:.2f}')


# #### MEDI + SEDI with suicide rate

# In[151]:


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


# In[152]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_Total_Index = mean_absolute_error(y_test, y_pred)
mse_Total_Index = mean_squared_error(y_test, y_pred)
rmse_Total_Index = np.sqrt(mse_Total_Index)
print(f'Mean absolute error: {mae_Total_Index:.2f}')
print(f'Mean squared error: {mse_Total_Index:.2f}')
print(f'Root mean squared error: {rmse_Total_Index:.2f}')


# In[156]:


df_index3


# In[157]:


df_index5 = df_index3.drop(['SuicideDeathRate'], axis = 1)
df_index5


# In[158]:


df_sd_16


# In[159]:


df_index_current = df_index5.merge(df_sd_16, on = ['county', 'fips'], how = 'left')


# In[160]:


df_index_current


# In[161]:


df_index3['Year'] = '2016'


# In[162]:


df_index_current['Year'] = '2016'


# In[163]:


df_index3.to_csv('All_Index_Next_Year_2016.csv')


# In[164]:


df_index_current.to_csv('All_Index_Current_Year_2016.csv')

