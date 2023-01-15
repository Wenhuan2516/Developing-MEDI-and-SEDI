#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as ddf
from pandas import Series, DataFrame
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import seaborn as sn
import plotly.express as px


# ### Load temperature dataset

# In[4]:


ur_files = ddf.read_csv(r"Temperature/US_counties_weather_2014.csv", dtype={'eightieth_percentile_income': 'float64','fips': 'object','fog': 'float64','hail': 'float64',
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


# In[8]:


df_svi_2014 = pd.read_csv(r'SVI_2014_US_COUNTY.csv', dtype = {'FIPS': str})
df_svi_2014


# In[9]:


df_svi_2014 = df_svi_2014[['COUNTY', 'ST_ABBR', 'FIPS', 'EP_POV', 'EP_UNEMP', 'EP_PCI', 'EP_NOHSDP', 'EP_AGE65', 'EP_AGE17', 
                           'EP_DISABL', 'EP_SNGPNT', 'EP_MINRTY', 'EP_LIMENG', 'EP_MUNIT', 'EP_MOBILE', 'EP_CROWD', 'EP_NOVEH', 
                           'EP_GROUPQ', 'RPL_THEMES']]


# In[11]:


df_svi_2014 = df_svi_2014.rename(columns = {'COUNTY': 'county', 'ST_ABBR': 'state', 'FIPS': 'fips', 'EP_POV': 'percent_below_poverty', 'EP_UNEMP': 'percent_unemployed_CDC', 'EP_PCI': 'per_capita_income', 
                                            'EP_NOHSDP': 'percent_no_highschool_diploma', 'EP_AGE65' : 'percent_age_65_and_older', 'EP_AGE17': 'percent_age_17_and_younger', 
                                            'EP_DISABL': 'percent_disabled', 'EP_SNGPNT': 'percent_single_parent_households_CDC', 'EP_MINRTY': 'percent_minorities', 'EP_LIMENG':'percent_limited_english_abilities' ,
                                            'EP_MUNIT': 'percent_multi_unit_housing', 'EP_MOBILE': 'percent_mobile_homes', 'EP_CROWD': 'percent_overcrowding', 'EP_NOVEH': 'percent_no_vehicle', 
                                            'EP_GROUPQ': 'percent_institutionalized_in_group_quarters', 'RPL_THEMES': 'SVI'})


# In[12]:


df_svi_2014


# In[13]:


df_svi_2014['county'] = df_svi_2014['county'] + ', ' + df_svi_2014['state']


# In[14]:


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


# In[15]:


correlation_matrix(df_svi_2014)


# ### Find the climate data part

# In[16]:


df_temp_2014 = df_svi[['state', 'county', 'fips','station_id', 'station_name', 'lat', 'lon', 'date', 'mean_temp', 
                        'min_temp', 'max_temp', 'dewpoint', 'sea_level_pressure', 'visibility', 'wind_speed', 'wind_gust', 'precipitation',
                       'precip_flag', 'fog', 'rain', 'snow', 'hail', 'thunder', 'tornado']]
df_temp_2014.head()


# In[17]:


df_temp_2014.shape


# In[18]:


days = df_temp_2014['date'].unique().tolist()
len(days)


# In[19]:


1125027/365


# In[20]:


df_temp_2014['dewpoint'].isnull().sum()


# In[21]:


80307/1125027


# In[22]:


df_temp_county = df_temp_2014.groupby(['state', 'county', 'fips']).mean()
df_temp_county = df_temp_county.reset_index()
df_temp_county.head()


# In[23]:


df_temp_county.shape


# In[24]:


df_temp_county['dewpoint'].isnull().sum()


# In[25]:


217/3124


# In[26]:


correlation_matrix(df_temp_county)


# ## Calculate Relative Humidity

# In[27]:


def Convertion_F_to_C(T):
    return (T - 32)*(5/9)


# In[28]:


import math


# In[29]:


def RH_Calculator(T, TD):
    T = Convertion_F_to_C(T)
    TD = Convertion_F_to_C(TD)
    return 100*(math.exp((17.625*TD)/(243.04+TD))/math.exp((17.625*T)/(243.04+T)))


# In[30]:


df_temp_2014['Relative_Humidity'] = df_temp_2014.apply(lambda x: RH_Calculator(x['mean_temp'], x['dewpoint']), axis=1)


# ## Calculate Heat Index

# In[31]:


def heat_index_calculation(T, RH):
    if RH < 13 and T > 80 and T < 112:
        return ((13-RH)/4)* math.sqrt((17- abs(T-95))/17)
    elif RH > 85 and T > 80 and T < 87:
        return ((RH - 85)/10)*((87-T)/5)
    elif T < 80:
        return 0.5 * (T + 61.0 + ((T-68.0)*1.2) + (RH*0.094))
    else:
        return -42.379 + 2.04901523*T + 10.14333127*RH - .22475541*T*RH - .00683783*T*T - .05481717*RH*RH + .00122874*T*T*RH + .00085282*T*RH*RH - .00000199*T*T*RH*RH


# In[32]:


df_temp_2014['Heat_Index'] = df_temp_2014.apply(lambda x: heat_index_calculation(x['mean_temp'], x['Relative_Humidity']), axis=1)


# In[33]:


df_temp_2014


# In[34]:


df_temp_2014.to_csv('daily_temp_2014.csv')


# #### From the above graph, we can see that max temp/min temp/dewpoint are highly correlated with mean temp, max Wind speed and max wind gust are highly correlated with wind values, so we can drop max temp, min temp, max wind speed, and max wind gust.

# In[35]:


df_temp_county3 = df_temp_county.drop(['min_temp', 'max_temp', 'wind_gust', 'dewpoint'], axis=1)
correlation_matrix(df_temp_county3)


# ### upload PM2.5 Data

# In[36]:


ur_files = ddf.read_csv(r"PM2.5_1014.csv", dtype = {'CountyFIPS': 'object'})
df_pm = ur_files.compute()
df_pm = df_pm.loc[:, ~df_pm.columns.str.contains('^Unnamed')]
df_pm.head()


# In[37]:


df_pm_2014 = df_pm[df_pm['Year'] == 2014]
df_pm_2014 = df_pm_2014.rename(columns = {'State': 'state', 'CountyFIPS': 'fips', 'County': 'county'})
df_pm_2014 = df_pm_2014.groupby(['state', 'fips', 'county']).mean()
df_pm_2014 = df_pm_2014.reset_index()
df_pm_2014


# In[38]:


df_pm_2014 = df_pm_2014.rename(columns = {'Value': 'PM2.5'})
df_pm_2014


# ### load Ozone data

# In[39]:


ur_files = ddf.read_csv(r"Ozone_1014.csv", dtype = {'CountyFIPS': 'object', 'Value': 'object'})
df_oz = ur_files.compute()
df_oz = df_oz.loc[:, ~df_oz.columns.str.contains('^Unnamed')]
df_oz.head()


# In[40]:


df_oz['Value'].unique()


# In[41]:


def changeValue(value):
    if value == 'No Data':
        value = None
    else:
        value = int(value)
    return value


# In[42]:


df_oz['Value'] = df_oz['Value'].apply(changeValue)
df_oz_2014 = df_oz[df_oz['Year'] == 2014]
df_oz_2014 = df_oz_2014.rename(columns = {'State': 'state', 'CountyFIPS': 'fips', 'County': 'county', 'Value': 'Ozone'})
df_oz_2014 = df_oz_2014.groupby(['state', 'fips', 'county']).mean()
df_oz_2014 = df_oz_2014.reset_index()
df_oz_2014


# #### Combine environmental factors

# In[43]:


df_air = df_pm_2014.merge(df_oz_2014, on = ['state','fips', 'Year', 'StateFIPS', 'county'], how = 'left')
df_air.head()


# In[44]:


df_air = df_air[['state', 'fips', 'county', 'PM2.5', 'Ozone']]
df_climate = df_temp_county3
df_env = df_climate.merge(df_air, on = ['state', 'fips', 'county'], how = 'left')
df_env


# ### checking missing climate data

# In[45]:


df_env['mean_temp'].isnull().sum()


# In[46]:


df_env['sea_level_pressure'].isnull().sum()


# In[47]:


df_env['visibility'].isnull().sum()


# In[48]:


df_env['wind_speed'].isnull().sum()


# In[49]:


df_env['precipitation'].isnull().sum()


# In[50]:


df_env['fog'].isnull().sum()


# In[51]:


df_env['rain'].isnull().sum()


# In[52]:


df_env['snow'].isnull().sum()


# In[53]:


df_env['hail'].isnull().sum()


# In[54]:


df_env['thunder'].isnull().sum()


# In[55]:


df_env['tornado'].isnull().sum()


# In[56]:


df_env['PM2.5'].isnull().sum()


# In[57]:


df_env['Ozone'].isnull().sum()


# ### Dropping variables when the variable missed too much data and assign the average mean value of adjacent counties to the missing county

# In[58]:


# since almost half of the counties are missing see_level_pressure data, we choose to drop it off
df_env = df_env.drop(['sea_level_pressure'], axis = 1)


# In[59]:


df_sd_14 = pd.read_csv(r'SuicideRate_Imputed/SuicideRate_Imputed_2014.csv', dtype={"fips": str})
df_sd_14 = df_sd_14[['county', 'fips', 'SuicideDeathRate']]
df_sd_14


# In[60]:


df_env_sd = df_env.merge(df_sd_14, on = ['fips'], how = 'left')


# In[61]:


df_env_sd = df_env_sd.drop(['county_x'], axis = 1)  
df_env_sd = df_env_sd.rename(columns = {'county_y': 'county'})
df_env_sd


# In[62]:


correlation_matrix(df_env_sd)


# In[63]:


weight = {'mean_temp': -0.13, 'visibility': 0.14, 'wind_speed': 0.03, 'precipitation': -0.17, 'fog': -0.07, 'rain': -0.11, 'snow': 0.03,
          'thunder': -0.06, 'tornado': -0.02, 'PM2.5': -0.27, 'Ozone': -0.09}


# In[64]:


variable = ['mean_temp', 'visibility', 'wind_speed', 'precipitation', 'fog',
            'rain', 'snow', 'thunder', 'tornado', 'PM2.5', 'Ozone']
for col in variable:
    df_env[col] = df_env[col] * weight[col]


# ### find percentile for each variable

# In[65]:


df_env['mean_temp_percentile'] = (df_env['mean_temp'] - df_env['mean_temp'].min()) / (df_env['mean_temp'].max() - df_env['mean_temp'].min())
df_env['visibility_percentile'] = (df_env['visibility'] - df_env['visibility'].min()) / (df_env['visibility'].max() - df_env['visibility'].min())
df_env['wind_speed_percentile'] = (df_env['wind_speed'] - df_env['wind_speed'].min()) / (df_env['wind_speed'].max() - df_env['wind_speed'].min())
df_env['precipitation_percentile'] = (df_env['precipitation'] - df_env['precipitation'].min()) / (df_env['precipitation'].max() - df_env['precipitation'].min())
df_env['fog_percentile'] = (df_env['fog'] - df_env['fog'].min()) / (df_env['fog'].max() - df_env['fog'].min())
df_env['rain_percentile'] = (df_env['rain'] - df_env['rain'].min()) / (df_env['rain'].max() - df_env['rain'].min())
df_env['snow_percentile'] = (df_env['snow'] - df_env['snow'].min()) / (df_env['snow'].max() - df_env['snow'].min())
df_env['thunder_percentile'] = (df_env['thunder'] - df_env['thunder'].min()) / ( df_env['thunder'].max() -  df_env['thunder'].min())
df_env['tornado_percentile'] = (df_env['tornado'] - df_env['tornado'].min()) / (df_env['tornado'].max() - df_env['tornado'].min())
df_env['PM2.5_percentile'] = (df_env['PM2.5'] - df_env['PM2.5'].min()) / (df_env['PM2.5'].max() - df_env['PM2.5'].min())
df_env['Ozone_percentile'] = (df_env['Ozone'] - df_env['Ozone'].min()) / (df_env['Ozone'].max() - df_env['Ozone'].min())


# #### give a score to climate factors and air quality factors

# In[66]:


df_env['ClimateScore'] = df_env['mean_temp_percentile'] + df_env['visibility_percentile'] + df_env['wind_speed_percentile'] + df_env['precipitation_percentile']+ df_env['fog_percentile'] + df_env['rain_percentile'] + df_env['snow_percentile'] + df_env['thunder_percentile'] + df_env['tornado_percentile']


# In[67]:


df_env['AirQualityScore'] = df_env['PM2.5_percentile'] + df_env['Ozone_percentile']


# #### give a percentile to each category

# In[68]:


df_env['ClimateScore_percentile'] = (df_env['ClimateScore'] - df_env['ClimateScore'].min()) / (df_env['ClimateScore'].max() - df_env['ClimateScore'].min())
df_env['AirQuality_percentile'] = (df_env['AirQualityScore'] - df_env['AirQualityScore'].min()) / (df_env['AirQualityScore'].max() - df_env['AirQualityScore'].min())


# #### sum the percentile of each category and assign the sum to be environment score

# In[69]:


df_env['EnvironmentScore'] = df_env['ClimateScore_percentile'] + df_env['AirQuality_percentile']


# #### MEDI is the percentile of environmental score

# In[70]:


df_env['MEDI'] = (df_env['EnvironmentScore'] - df_env['EnvironmentScore'].min()) / (df_env['EnvironmentScore'].max() - df_env['EnvironmentScore'].min())


# In[71]:


df_env.head()


# In[72]:


df_env['MEDI'].describe()


# In[73]:


df_env['county'] = df_env['county'] + ',' + df_env['state']


# In[74]:


df_env2 = df_env.dropna(how = 'any')
df_env2.shape[0]


# ### Plot out MEDI

# In[75]:


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

# In[91]:


df_svi_2014


# In[92]:


df_SEDI = df_svi_2014.merge(df_sd_14, on = ['fips'], how = 'left')


# In[93]:


correlation_matrix(df_SEDI)


# In[94]:


df_SEDI.to_csv('Social_Economic_Suicide_2014.csv')


# In[95]:


weight = {'percent_below_poverty': 0.03, 'per_capita_income': -0.03, 'percent_unemployed_CDC': -0.15, 'percent_no_highschool_diploma': 0.02, 
         'percent_age_65_and_older': 0.24, 'percent_age_17_and_younger': -0.13, 'percent_disabled': 0.26, 'percent_single_parent_households_CDC': -0.19,
         'percent_minorities': -0.17, 'percent_limited_english_abilities': -0.12, 'percent_multi_unit_housing': -0.25, 'percent_mobile_homes':0.20, 
         'percent_overcrowding': 0.001, 'percent_no_vehicle': -0.09, 'percent_institutionalized_in_group_quarters': 0.04}


# In[96]:


cols = df_SEDI.columns
cols


# In[97]:


df_SEDI = df_SEDI.drop(['county_y'], axis = 1)
df_SEDI = df_SEDI.rename(columns = {'county_x': 'county'})
df_SEDI


# In[98]:


cols = df_SEDI.columns


# In[99]:


cols


# In[100]:


cols_list = cols[3:-2]
cols_list


# In[101]:


for col in cols_list:
    df_SEDI[col] = df_SEDI[col] * weight[col]


# In[102]:


col_percentile = []
for col in cols_list:
    col_per = col + '_percentile'
    col_percentile.append(col_per)
    
col_percentile


# In[103]:


for col_per, col in zip(col_percentile, cols_list):
    df_SEDI[col_per] = (df_SEDI[col] - df_SEDI[col].min())/(df_SEDI[col].max() - df_SEDI[col].min())


# In[104]:


df_SEDI.columns


# In[105]:


df_SEDI['Socialeconomic_Theme'] = df_SEDI['percent_below_poverty_percentile'] + df_SEDI['per_capita_income_percentile'] + df_SEDI['percent_unemployed_CDC_percentile'] + df_SEDI['percent_no_highschool_diploma_percentile']


# In[106]:


df_SEDI['Household_Theme'] = df_SEDI['percent_age_65_and_older_percentile'] + df_SEDI['percent_age_17_and_younger_percentile'] + df_SEDI['percent_disabled_percentile'] + df_SEDI['percent_single_parent_households_CDC_percentile']
df_SEDI['Minority_Theme'] = df_SEDI['percent_minorities_percentile'] + df_SEDI['percent_limited_english_abilities_percentile']
df_SEDI['HousingType_Theme'] = df_SEDI['percent_multi_unit_housing_percentile'] + df_SEDI['percent_mobile_homes_percentile'] + df_SEDI['percent_overcrowding_percentile'] + df_SEDI['percent_no_vehicle_percentile']


# In[107]:


df_SEDI['Socialeconomic_Theme_percentile'] = (df_SEDI['Socialeconomic_Theme'] - df_SEDI['Socialeconomic_Theme'].min())/(df_SEDI['Socialeconomic_Theme'].max() - df_SEDI['Socialeconomic_Theme'].min())
df_SEDI['Household_Theme_percentile'] = (df_SEDI['Household_Theme'] - df_SEDI['Household_Theme'].min())/(df_SEDI['Household_Theme'].max() - df_SEDI['Household_Theme'].min())
df_SEDI['Minority_Theme_percentile'] = (df_SEDI['Minority_Theme'] - df_SEDI['Minority_Theme'].min())/(df_SEDI['Minority_Theme'].max() - df_SEDI['Minority_Theme'].min())
df_SEDI['HousingType_Theme_percentile'] = (df_SEDI['HousingType_Theme'] - df_SEDI['HousingType_Theme'].min())/(df_SEDI['HousingType_Theme'].max() - df_SEDI['HousingType_Theme'].min())
df_SEDI['SEI_Score'] = df_SEDI['Socialeconomic_Theme_percentile'] + df_SEDI['Household_Theme_percentile'] + df_SEDI['Minority_Theme_percentile'] + df_SEDI['HousingType_Theme_percentile']
df_SEDI['SEDI'] = (df_SEDI['SEI_Score'] - df_SEDI['SEI_Score'].min())/(df_SEDI['SEI_Score'].max() - df_SEDI['SEI_Score'].min())
df_SEDI


# In[111]:


df_SEDI.to_csv('SEDI_2014.csv')


# In[112]:


df_SEDI['SEDI'].describe()


# In[113]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_SEDI, geojson=counties, locations='fips', color='SEDI',
                           color_continuous_scale="rainbow",
                           range_color=(0.25, 0.75),
                           scope="usa",
                           hover_name="county",
                           labels={'SEI':'SEI'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[114]:


df_svi_2014


# In[115]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_svi_2014, geojson=counties, locations='fips', color='SVI',
                           color_continuous_scale="rainbow",
                           range_color=(0, 1),
                           scope="usa",
                           hover_name="county",
                           labels={'SVI':'SVI'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[116]:


df_SEDI = df_SEDI[['fips', 'SEDI']]
df_SEDI


# In[117]:


df_MEDI = df_env[['county', 'fips', 'MEDI']]
df_MEDI


# In[118]:


df_final_index = df_SEDI.merge(df_MEDI, on = 'fips', how = 'left')
df_final_index


# In[119]:


df_final_index['SumOfTwoIndex'] = df_final_index['SEDI'] + df_final_index['MEDI']


# In[120]:


df_final_index['MEDI + SEDI'] = (df_final_index['SumOfTwoIndex'] - df_final_index['SumOfTwoIndex'].min())/(df_final_index['SumOfTwoIndex'].max() - df_final_index['SumOfTwoIndex'].min())
df_final_index


# In[121]:


df_final_index['MEDI + SEDI'].describe()


# In[122]:


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


# #### Combine SVI and MEDI

# In[123]:


df_MEDI = df_env[['state', 'county', 'fips', 'MEDI']]
df_MEDI


# In[124]:


df_svi_2014


# In[125]:


df_final = df_svi_2014.merge(df_MEDI, on = ['fips'], how = 'left')


# In[126]:


df_final


# In[127]:


df_final = df_final.drop(['county_y', 'state_x'], axis = 1)
df_final = df_final.rename(columns = {'county_x': 'county', 'state_y': 'state'})
df_final


# In[128]:


df_final['SVI + MEDI'] = df_final['SVI'] + df_final['MEDI']


# In[129]:


df_final['Updated Index'] = (df_final['SVI + MEDI'] - df_final['SVI + MEDI'].min()) / (df_final['SVI + MEDI'].max() - df_final['SVI + MEDI'].min())


# In[130]:


df_final2 = df_final.dropna(how = 'any')


# In[131]:


df_final2.shape[0]


# In[132]:


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


# In[133]:


df_final


# ### Evaluation and Comparison

# In[134]:


df_index = df_final[['state', 'county', 'fips', 'SVI', 'MEDI', 'Updated Index']]


# In[135]:


df_index2 = df_index.merge(df_final_index, on = ['fips', 'MEDI'], how = 'left')


# In[136]:


df_index2


# In[137]:


df_index2 = df_index2.drop(['county_y'], axis = 1)
df_index2 = df_index2.rename(columns = {'county_x': 'county'})
df_index2


# In[138]:


df_sd = pd.read_csv(r'SuicideRate_Imputed/SuicideRate_Imputed_2015.csv',dtype={"fips": str})


# In[139]:


df_sd = df_sd[['county', 'fips', 'SuicideDeathRate']]


# In[140]:


df_sd


# In[141]:


df_index3 = df_index2.merge(df_sd, on = ['fips'], how = 'left')
df_index3


# In[142]:


df_index3 = df_index3[['state', 'county_y', 'fips', 'SVI', 'MEDI', 'Updated Index', 'SuicideDeathRate','SEDI','MEDI + SEDI']]
df_index3 = df_index3.rename(columns = {'county_y': 'county'})
df_index3


# In[143]:


df_index3['Suicide_Rate_Percentile'] = (df_index3['SuicideDeathRate'] - df_index3['SuicideDeathRate'].min()) / (df_index3['SuicideDeathRate'].max() - df_index3['SuicideDeathRate'].min())


# In[144]:


df_index3


# ### Use Mean absolute error/Mean squared error/Root-Mean-Square Error to check the model performance

# ####  SVI and Suicide Rate

# In[145]:


df_index4 = df_index3[['state', 'county', 'fips', 'SVI', 'MEDI', 'Updated Index', 'SuicideDeathRate', 'SEDI','MEDI + SEDI', 'Suicide_Rate_Percentile']]


# In[146]:


correlation_matrix(df_index4)


# In[147]:


df_index4 = df_index3.dropna(how = 'any')


# In[148]:


import seaborn as sn
import plotly.express as px
sn.lmplot('SVI', 'Suicide_Rate_Percentile', data = df_index4)


# In[149]:


sn.lmplot('MEDI', 'Suicide_Rate_Percentile', data = df_index4)


# In[150]:


sn.lmplot('Updated Index', 'Suicide_Rate_Percentile', data = df_index4)


# In[151]:


sn.lmplot('SEDI', 'Suicide_Rate_Percentile', data = df_index4)


# In[152]:


sn.lmplot('MEDI + SEDI', 'Suicide_Rate_Percentile', data = df_index4)


# In[153]:


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


# In[154]:


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


# In[155]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_SVI = mean_absolute_error(y_test, y_pred)
mse_SVI = mean_squared_error(y_test, y_pred)
rmse_SVI = np.sqrt(mse_SVI)
print(f'Mean absolute error: {mae_SVI:.2f}')
print(f'Mean squared error: {mse_SVI:.2f}')
print(f'Root mean squared error: {rmse_SVI:.2f}')


# In[156]:


coefficient_SVI[0][0],intercept_SVI[0]


# #### MEDI with suicide rate

# In[157]:


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


# In[158]:


coefficient_MEDI[0][0],intercept_MEDI[0]


# In[159]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_MEDI = mean_absolute_error(y_test, y_pred)
mse_MEDI = mean_squared_error(y_test, y_pred)
rmse_MEDI = np.sqrt(mse_MEDI)
print(f'Mean absolute error: {mae_MEDI:.2f}')
print(f'Mean squared error: {mse_MEDI:.2f}')
print(f'Root mean squared error: {rmse_MEDI:.2f}')


# #### Total Index with suicide rate

# In[160]:


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


# In[161]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_Total_Index = mean_absolute_error(y_test, y_pred)
mse_Total_Index = mean_squared_error(y_test, y_pred)
rmse_Total_Index = np.sqrt(mse_Total_Index)
print(f'Mean absolute error: {mae_Total_Index:.2f}')
print(f'Mean squared error: {mse_Total_Index:.2f}')
print(f'Root mean squared error: {rmse_Total_Index:.2f}')


# #### SEDI with suicide rate

# In[162]:


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


# In[163]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_Total_Index = mean_absolute_error(y_test, y_pred)
mse_Total_Index = mean_squared_error(y_test, y_pred)
rmse_Total_Index = np.sqrt(mse_Total_Index)
print(f'Mean absolute error: {mae_Total_Index:.2f}')
print(f'Mean squared error: {mse_Total_Index:.2f}')
print(f'Root mean squared error: {rmse_Total_Index:.2f}')


# #### MEDI + SEDI with suicide rate

# In[164]:


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


# In[165]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_Total_Index = mean_absolute_error(y_test, y_pred)
mse_Total_Index = mean_squared_error(y_test, y_pred)
rmse_Total_Index = np.sqrt(mse_Total_Index)
print(f'Mean absolute error: {mae_Total_Index:.2f}')
print(f'Mean squared error: {mse_Total_Index:.2f}')
print(f'Root mean squared error: {rmse_Total_Index:.2f}')


# In[166]:


df_index3


# In[167]:


df_index5 = df_index3.drop(['SuicideDeathRate'], axis = 1)
df_index5


# In[168]:


df_sd_14


# In[169]:


df_index_current = df_index5.merge(df_sd_14, on = ['county', 'fips'], how = 'left')


# In[170]:


df_index_current


# In[171]:


df_index3['Year'] = '2014'


# In[172]:


df_index_current['Year'] = '2014'


# In[173]:


df_index3.to_csv('All_Index_Next_Year_2014.csv')


# In[174]:


df_index_current.to_csv('All_Index_Current_Year_2014.csv')


# In[ ]:




