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


ur_files = ddf.read_csv('RuralUrban.csv', dtype = {'FIPS code': str, '1990-based code': 'object', 'County 2012 pop': 'object'})
df_urban = ur_files.compute()
df_urban = df_urban.loc[:, ~df_urban.columns.str.contains('^Unnamed')]
df_urban.head()


# In[11]:


df_urban['2013 code'].dtype


# In[12]:


def defineRuralUrban(value):
    if value >= 1 and value <= 4:
        return 'Urban'
    else:
        return 'Rural'


# In[13]:


df_urban['RuralOrUrban'] = df_urban['2013 code'].apply(defineRuralUrban)
df_urban


# In[14]:


def fixFips(code):
    return code.rjust(5, '0')


# In[15]:


code = '1001'
code1 = fixFips(code)
code1


# In[16]:


df_urban['FIPS code'] = df_urban['FIPS code'].apply(fixFips)
df_urban = df_urban.rename( columns = {'FIPS code': 'fips'})
df_urban


# In[17]:


df_variable = df_svi_county[['state','county','fips','lat','lon','percent_below_poverty', 'per_capita_income','percent_unemployed_CDC', 
                      'percent_no_highschool_diploma', 'percent_age_65_and_older', 'percent_age_17_and_younger', 
                     'percent_disabled', 'percent_single_parent_households_CDC', 'percent_minorities', 'percent_limited_english_abilities', 
                      'percent_multi_unit_housing','percent_mobile_homes', 'percent_overcrowding', 'percent_no_vehicle', 
                      'percent_institutionalized_in_group_quarters']]


# In[18]:


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


# In[19]:


df_variable.shape[0]


# In[20]:


correlation_matrix(df_variable)


# In[21]:


df_percentile = df_svi_county[['state','county','fips','lat','lon','percentile_rank_below_poverty','percentile_rank_unemployed',
                      'percentile_rank_per_capita_income','percentile_rank_no_highschool_diploma','percentile_rank_socioeconomic_theme',
                      'percentile_rank_age_65_and_older','percentile_rank_age_17_and_younger','percentile_rank_disabled','percentile_rank_single_parent_households',
                      'percentile_rank_household_comp_disability_theme','percentile_rank_minorities','percentile_rank_limited_english_abilities',
                      'percentile_rank_minority_status_and_language_theme','percentile_rank_multi_unit_housing','percentile_rank_mobile_homes',
                      'percentile_rank_overcrowding','percentile_rank_no_vehicle','percentile_rank_institutionalized_in_group_quarters',
                      'percentile_rank_housing_and_transportation','percentile_rank_social_vulnerability']]


# In[22]:


df_percentile


# In[23]:


df_sample = df_percentile[df_percentile['county'] == 'Adams']
df_sample


# In[24]:


df_sample2 = df_sample[df_sample['state'] == 'Washington']


# In[25]:


df_result = df_sample2[['percentile_rank_socioeconomic_theme', 'percentile_rank_household_comp_disability_theme', 'percentile_rank_minority_status_and_language_theme',
                        'percentile_rank_housing_and_transportation', 'percentile_rank_social_vulnerability']]
df_result


# ### Find the climate data part

# In[26]:


df_temp_2016 = df_svi[['state', 'county', 'fips','station_id', 'station_name', 'date', 'mean_temp', 
                        'min_temp', 'max_temp', 'dewpoint', 'sea_level_pressure', 'visibility', 'wind_speed', 'wind_gust', 'precipitation',
                       'precip_flag', 'fog', 'rain', 'snow', 'hail', 'thunder', 'tornado']]
df_temp_2016.head()


# In[27]:


df_temp_county = df_temp_2016.groupby(['state', 'county', 'fips']).mean()
df_temp_county = df_temp_county.reset_index()
df_temp_county.head()


# In[28]:


correlation_matrix(df_temp_county)


# #### From the above graph, we can see that max temp/min temp/dewpoint are highly correlated with mean temp, max Wind speed and max wind gust are highly correlated with wind values, so we can drop max temp, min temp, max wind speed, and max wind gust.

# In[29]:


df_temp_county3 = df_temp_county.drop(['min_temp', 'max_temp', 'wind_gust', 'dewpoint'], axis=1)
correlation_matrix(df_temp_county3)


# ### upload PM2.5 Data

# In[30]:


ur_files = ddf.read_csv(r"PM2.5.csv", dtype = {'CountyFIPS': 'object'})
df_pm = ur_files.compute()
df_pm = df_pm.loc[:, ~df_pm.columns.str.contains('^Unnamed')]
df_pm.head()


# In[31]:


df_pm_2016 = df_pm[df_pm['Year'] == 2016]
df_pm_2016 = df_pm_2016.rename(columns = {'State': 'state', 'CountyFIPS': 'fips', 'County': 'county'})
df_pm_2016 = df_pm_2016.groupby(['state', 'fips', 'county']).mean()
df_pm_2016 = df_pm_2016.reset_index()
df_pm_2016


# In[32]:


df_pm_2016 = df_pm_2016.rename(columns = {'Value': 'PM2.5'})
df_pm_2016


# ### load Ozone data

# In[33]:


ur_files = ddf.read_csv(r"Ozone.csv", dtype = {'CountyFIPS': 'object', 'Value': 'object'})
df_oz = ur_files.compute()
df_oz = df_oz.loc[:, ~df_oz.columns.str.contains('^Unnamed')]
df_oz.head()


# In[34]:


df_oz['Value'].unique()


# In[35]:


def changeValue(value):
    if value == 'No Data':
        value = None
    else:
        value = int(value)
    return value


# In[36]:


df_oz['Value'] = df_oz['Value'].apply(changeValue)
df_oz_2016 = df_oz[df_oz['Year'] == 2016]
df_oz_2016 = df_oz_2016.rename(columns = {'State': 'state', 'CountyFIPS': 'fips', 'County': 'county', 'Value': 'Ozone'})
df_oz_2016 = df_oz_2016.groupby(['state', 'fips', 'county']).mean()
df_oz_2016 = df_oz_2016.reset_index()
df_oz_2016


# ### Upload drought data

# In[37]:


ur_files = ddf.read_csv(r"Drought.csv", dtype = {'CountyFIPS': 'object'})
df_drought = ur_files.compute()
df_drought = df_drought.loc[:, ~df_drought.columns.str.contains('^Unnamed')]
df_drought.head()


# In[38]:


df_drought['Value'].dtype


# In[39]:


str1 = '26.00%'
print(float(str1[:-1]))


# In[40]:


def convertValue(value):
    value2 = value[:-1]
    return float(value2)


# In[41]:


df_drought['Value'] = df_drought['Value'].apply(convertValue)


# In[42]:


df_drought['Value'].dtype


# In[43]:


df_drought = df_drought.rename(columns = {'CountyFIPS': 'fips', 'Value': 'Drought'})


# In[44]:


df_drought['Cumulative Drought Severity'].unique()


# In[45]:


df_drought = df_drought[df_drought['Cumulative Drought Severity'] == 'Severe drought or greater']
df_drought


# In[46]:


df_drought['Year'].dtype


# In[47]:


df_drought = df_drought[df_drought['Year'] == 2016]
df_drought


# In[48]:


df_drought = df_drought[['fips', 'Drought']]
df_drought.head()


# In[49]:


df_climate = df_temp_county3.merge(df_drought, on = 'fips', how = 'left')
df_climate.head()


# ### upload humidity_temp_index

# In[50]:


ur_files = ddf.read_csv(r"Humidity_Temp_Index/imputed_humid_index_2016.csv", dtype = {'county': 'object'})
df_humidity = ur_files.compute()
df_humidity = df_humidity.loc[:, ~df_humidity.columns.str.contains('^Unnamed')]
df_humidity.head()


# In[51]:


df_humidity['county'].dtype


# In[52]:


def convertCode(value):
    return value.rjust(5, '0')


# In[53]:


print(convertCode('1001'))


# In[54]:


df_humidity['county'] = df_humidity['county'].apply(convertCode)


# In[55]:


df_humidity = df_humidity[['county', 'closest_station_index']]
df_humidity = df_humidity.rename(columns = {'county': 'fips', 'closest_station_index': 'humidity_temp_index'})
df_humidity


# In[56]:


df_climate = df_climate.merge(df_humidity, on = 'fips', how = 'left')
df_climate


# ### Combine environmental factors

# In[57]:


df_air = df_pm_2016.merge(df_oz_2016, on = ['state','fips', 'Year', 'StateFIPS', 'county'], how = 'left')
df_air.head()


# In[58]:


df_air = df_air[['state', 'fips', 'county', 'PM2.5', 'Ozone']]
df_env = df_climate.merge(df_air, on = ['state', 'fips', 'county'], how = 'left')
df_env


# ### checking missing climate data

# In[59]:


df_env['mean_temp'].isnull().sum()


# In[60]:


df_env['sea_level_pressure'].isnull().sum()


# In[61]:


df_env['visibility'].isnull().sum()


# In[62]:


df_env['wind_speed'].isnull().sum()


# In[63]:


df_env['precipitation'].isnull().sum()


# In[64]:


df_env['fog'].isnull().sum()


# In[65]:


df_env['rain'].isnull().sum()


# In[66]:


df_env['snow'].isnull().sum()


# In[67]:


df_env['hail'].isnull().sum()


# In[68]:


df_env['thunder'].isnull().sum()


# In[69]:


df_env['tornado'].isnull().sum()


# In[70]:


df_env['PM2.5'].isnull().sum()


# In[71]:


df_env['Ozone'].isnull().sum()


# In[72]:


df_env['Drought'].isnull().sum()


# In[73]:


df_env['humidity_temp_index'].isnull().sum()


# ### Dropping variables when the variable missed too much data and assign the average mean value of adjacent counties to the missing county

# In[74]:


# since almost half of the counties are missing see_level_pressure data, we choose to drop it off
df_env = df_env.drop(['sea_level_pressure'], axis = 1)


# In[75]:


df_env_urban = df_env.merge(df_urban[['fips', 'RuralOrUrban']], on = 'fips', how = 'left')
df_env_urban


# In[76]:


df_sd_16 = pd.read_csv(r'SuicideRate_Imputed/SuicideRate_Imputed_2016.csv', dtype={"fips": str})
df_sd_16 = df_sd_16[['county', 'fips', 'SuicideDeathRate']]
df_sd_16


# In[77]:


df_env_sd = df_env_urban.merge(df_sd_16, on = ['fips'], how = 'left')


# In[78]:


df_env_sd = df_env_sd.drop(['county_x'], axis = 1)  
df_env_sd = df_env_sd.rename(columns = {'county_y': 'county'})
df_env_sd


# In[79]:


correlation_matrix(df_env_sd)


# In[80]:


weight = {'mean_temp': -0.20, 'visibility': 0.06, 'wind_speed': 0.12, 'precipitation': -0.10, 'fog': -0.09, 'rain': -0.11, 'snow': 0.06,
         'hail': -0.04, 'thunder': 0.02, 'tornado': -0.03, 'Drought': 0.10, 'humidity_temp_index': -0.21, 'PM2.5': -0.29, 'Ozone': -0.12}


# In[81]:


variable = ['mean_temp', 'visibility', 'wind_speed', 'precipitation', 'fog',
            'rain', 'snow', 'hail', 'thunder', 'tornado','Drought', 'humidity_temp_index', 'PM2.5', 'Ozone']
for col in variable:
    df_env_urban[col] = df_env_urban[col] * weight[col]


# ### find percentile for each variable

# In[82]:


df_env_urban['mean_temp_percentile'] = (df_env_urban['mean_temp'] - df_env_urban['mean_temp'].min()) / (df_env_urban['mean_temp'].max() - df_env_urban['mean_temp'].min())
df_env_urban['visibility_percentile'] = (df_env_urban['visibility'] - df_env_urban['visibility'].min()) / (df_env_urban['visibility'].max() - df_env_urban['visibility'].min())
df_env_urban['wind_speed_percentile'] = (df_env_urban['wind_speed'] - df_env_urban['wind_speed'].min()) / (df_env_urban['wind_speed'].max() - df_env_urban['wind_speed'].min())
df_env_urban['precipitation_percentile'] = (df_env_urban['precipitation'] - df_env_urban['precipitation'].min()) / (df_env_urban['precipitation'].max() - df_env_urban['precipitation'].min())
df_env_urban['fog_percentile'] = (df_env_urban['fog'] - df_env_urban['fog'].min()) / (df_env_urban['fog'].max() - df_env_urban['fog'].min())
df_env_urban['rain_percentile'] = (df_env_urban['rain'] - df_env_urban['rain'].min()) / (df_env_urban['rain'].max() - df_env_urban['rain'].min())
df_env_urban['snow_percentile'] = (df_env_urban['snow'] - df_env_urban['snow'].min()) / (df_env_urban['snow'].max() - df_env_urban['snow'].min())
df_env_urban['hail_percentile'] = (df_env_urban['hail'] - df_env_urban['hail'].min()) / (df_env_urban['hail'].max() - df_env_urban['hail'].min())
df_env_urban['thunder_percentile'] = (df_env_urban['thunder'] - df_env_urban['thunder'].min()) / ( df_env_urban['thunder'].max() -  df_env_urban['thunder'].min())
df_env_urban['tornado_percentile'] = (df_env_urban['tornado'] - df_env_urban['tornado'].min()) / (df_env_urban['tornado'].max() - df_env_urban['tornado'].min())
df_env_urban['Drought_percentile'] = (df_env_urban['Drought'] - df_env_urban['Drought'].min()) / (df_env_urban['Drought'].max() - df_env_urban['Drought'].min())
df_env_urban['humidity_temp_index_percentile'] = (df_env_urban['humidity_temp_index'] - df_env_urban['humidity_temp_index'].min()) / (df_env_urban['humidity_temp_index'].max() - df_env_urban['humidity_temp_index'].min())
df_env_urban['PM2.5_percentile'] = (df_env_urban['PM2.5'] - df_env_urban['PM2.5'].min()) / (df_env_urban['PM2.5'].max() - df_env_urban['PM2.5'].min())
df_env_urban['Ozone_percentile'] = (df_env_urban['Ozone'] - df_env_urban['Ozone'].min()) / (df_env_urban['Ozone'].max() - df_env_urban['Ozone'].min())


# In[85]:


df_env_urban['ClimateScore'] = df_env_urban['mean_temp_percentile'] + df_env_urban['visibility_percentile'] + df_env_urban['wind_speed_percentile'] + df_env_urban['precipitation_percentile']+ df_env_urban['fog_percentile'] + df_env_urban['rain_percentile'] + df_env_urban['snow_percentile'] + df_env_urban['hail_percentile'] + df_env_urban['thunder_percentile'] + df_env_urban['tornado_percentile'] + df_env_urban['Drought_percentile'] + df_env_urban['humidity_temp_index_percentile']  


# In[86]:


df_env_urban['AirQualityScore'] = df_env_urban['PM2.5_percentile'] + df_env_urban['Ozone_percentile']


# #### give a percentile to each category

# In[87]:


df_env_urban['ClimateScore_percentile'] = (df_env_urban['ClimateScore'] - df_env_urban['ClimateScore'].min()) / (df_env_urban['ClimateScore'].max() - df_env_urban['ClimateScore'].min())
df_env_urban['AirQuality_percentile'] = (df_env_urban['AirQualityScore'] - df_env_urban['AirQualityScore'].min()) / (df_env_urban['AirQualityScore'].max() - df_env_urban['AirQualityScore'].min())


# #### sum the percentile of each category and assign the sum to be environment score

# In[88]:


df_env_urban['EnvironmentScore'] = df_env_urban['ClimateScore_percentile'] + df_env_urban['AirQuality_percentile']


# #### MEDI is the percentile of environmental score

# In[89]:


df_env_urban['MEDI'] = (df_env_urban['EnvironmentScore'] - df_env_urban['EnvironmentScore'].min()) / (df_env_urban['EnvironmentScore'].max() - df_env_urban['EnvironmentScore'].min())


# In[90]:


df_env_urban.head()


# In[91]:


df_env_urban.columns


# In[92]:


df_env_urban['MEDI'].describe()


# In[93]:


df_env_urban_original = df_env_urban[df_env_urban['RuralOrUrban'] == 'Urban']
df_env_rural_original = df_env_urban[df_env_urban['RuralOrUrban'] == 'Rural']


# In[280]:


df_env_urban_original.shape[0]


# In[281]:


df_env_rural_original.shape[0]


# In[94]:


df_env_urban_original = df_env_urban_original.rename(columns = {'MEDI': 'MEDI_Urban_Overall'})
df_env_rural_original = df_env_rural_original.rename(columns = {'MEDI': 'MEDI_Rural_Overall'})
df_env_rural_original


# ### Divide to two parts: Urban and Rural

# In[95]:


df_env_sd


# In[96]:


df_env_sd_urban = df_env_sd[df_env_sd['RuralOrUrban'] == 'Urban']
df_env_sd_rural = df_env_sd[df_env_sd['RuralOrUrban'] == 'Rural']


# In[97]:


correlation_matrix(df_env_sd_urban)


# In[98]:


weight_urban = {'mean_temp': -0.09, 'visibility': 0.04, 'wind_speed': 0, 'precipitation': -0.09, 'fog': 0.02, 'rain': -0.04, 'snow': 0.02,
         'hail': 0.03, 'thunder': 0.09, 'tornado': -0.02, 'Drought': 0.07, 'humidity_temp_index': -0.07, 'PM2.5': -0.16, 'Ozone': -0.16}


# In[99]:


variable = ['mean_temp', 'visibility', 'precipitation', 'fog',
            'rain', 'snow', 'hail', 'thunder', 'tornado', 'Drought', 'humidity_temp_index', 'PM2.5', 'Ozone']
for col in variable:
    df_env_sd_urban[col] = df_env_sd_urban[col] * weight_urban[col]


# In[100]:


df_env_sd_urban['mean_temp_percentile'] = (df_env_sd_urban['mean_temp'] - df_env_sd_urban['mean_temp'].min()) / (df_env_sd_urban['mean_temp'].max() - df_env_sd_urban['mean_temp'].min())
df_env_sd_urban['visibility_percentile'] = (df_env_sd_urban['visibility'] - df_env_sd_urban['visibility'].min()) / (df_env_sd_urban['visibility'].max() - df_env_sd_urban['visibility'].min())
df_env_sd_urban['precipitation_percentile'] = (df_env_sd_urban['precipitation'] - df_env_sd_urban['precipitation'].min()) / (df_env_sd_urban['precipitation'].max() - df_env_sd_urban['precipitation'].min())
df_env_sd_urban['fog_percentile'] = (df_env_sd_urban['fog'] - df_env_sd_urban['fog'].min()) / (df_env_sd_urban['fog'].max() - df_env_sd_urban['fog'].min())
df_env_sd_urban['rain_percentile'] = (df_env_sd_urban['rain'] - df_env_sd_urban['rain'].min()) / (df_env_sd_urban['rain'].max() - df_env_sd_urban['rain'].min())
df_env_sd_urban['snow_percentile'] = (df_env_sd_urban['snow'] - df_env_sd_urban['snow'].min()) / (df_env_sd_urban['snow'].max() - df_env_sd_urban['snow'].min())
df_env_sd_urban['hail_percentile'] = (df_env_sd_urban['hail'] - df_env_sd_urban['hail'].min()) / (df_env_sd_urban['hail'].max() - df_env_sd_urban['hail'].min())
df_env_sd_urban['thunder_percentile'] = (df_env_sd_urban['thunder'] - df_env_sd_urban['thunder'].min()) / ( df_env_sd_urban['thunder'].max() -  df_env_sd_urban['thunder'].min())
df_env_sd_urban['tornado_percentile'] = (df_env_sd_urban['tornado'] - df_env_sd_urban['tornado'].min()) / (df_env_sd_urban['tornado'].max() - df_env_sd_urban['tornado'].min())
df_env_sd_urban['Drought_percentile'] = (df_env_sd_urban['Drought'] - df_env_sd_urban['Drought'].min()) / (df_env_sd_urban['Drought'].max() - df_env_sd_urban['Drought'].min())
df_env_sd_urban['humidity_temp_index_percentile'] = (df_env_sd_urban['humidity_temp_index'] - df_env_sd_urban['humidity_temp_index'].min()) / (df_env_sd_urban['humidity_temp_index'].max() - df_env_sd_urban['humidity_temp_index'].min())
df_env_sd_urban['PM2.5_percentile'] = (df_env_sd_urban['PM2.5'] - df_env_sd_urban['PM2.5'].min()) / (df_env_sd_urban['PM2.5'].max() - df_env_sd_urban['PM2.5'].min())
df_env_sd_urban['Ozone_percentile'] = (df_env_sd_urban['Ozone'] - df_env_sd_urban['Ozone'].min()) / (df_env_sd_urban['Ozone'].max() - df_env_sd_urban['Ozone'].min())


# In[101]:


df_env_sd_urban


# In[102]:


df_env_sd_urban['ClimateScore'] = df_env_sd_urban['mean_temp_percentile'] + df_env_sd_urban['visibility_percentile'] + df_env_sd_urban['precipitation_percentile']+ df_env_sd_urban['fog_percentile'] + df_env_sd_urban['rain_percentile'] + df_env_sd_urban['snow_percentile'] + df_env_sd_urban['hail_percentile'] + df_env_sd_urban['thunder_percentile'] + df_env_sd_urban['tornado_percentile'] + df_env_sd_urban['Drought_percentile'] + df_env_sd_urban['humidity_temp_index_percentile']
df_env_sd_urban['AirQualityScore'] = df_env_sd_urban['PM2.5_percentile'] + df_env_sd_urban['Ozone_percentile']
df_env_sd_urban['ClimateScore_percentile'] = (df_env_sd_urban['ClimateScore'] - df_env_sd_urban['ClimateScore'].min()) / (df_env_sd_urban['ClimateScore'].max() - df_env_sd_urban['ClimateScore'].min())
df_env_sd_urban['AirQuality_percentile'] = (df_env_sd_urban['AirQualityScore'] - df_env_sd_urban['AirQualityScore'].min()) / (df_env_sd_urban['AirQualityScore'].max() - df_env_sd_urban['AirQualityScore'].min())
df_env_sd_urban['EnvironmentScore'] = df_env_sd_urban['ClimateScore_percentile'] + df_env_sd_urban['AirQuality_percentile']
df_env_sd_urban['MEDI_Urban'] = (df_env_sd_urban['EnvironmentScore'] - df_env_sd_urban['EnvironmentScore'].min()) / (df_env_sd_urban['EnvironmentScore'].max() - df_env_sd_urban['EnvironmentScore'].min())


# In[103]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_env_sd_urban, geojson=counties, locations='fips', color='MEDI_Urban',
                           color_continuous_scale="rainbow",
                           range_color=(0, 1),
                           scope="usa",
                           hover_name="county",
                           labels={'MEDI':'MEDI'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[104]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_env_urban_original, geojson=counties, locations='fips', color='MEDI_Urban_Overall',
                           color_continuous_scale="rainbow",
                           range_color=(0, 1),
                           scope="usa",
                           hover_name="county",
                           labels={'MEDI':'MEDI'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# #### Deal with rural areas

# In[105]:


correlation_matrix(df_env_sd_rural)


# In[106]:


weight_rural = {'mean_temp': -0.19, 'visibility': 0.08, 'wind_speed': 0.12, 'precipitation': -0.05, 'fog': -0.10, 'rain': -0.09, 'snow': 0.04,
         'hail': -0.02, 'thunder': -0.03, 'tornado': -0.02, 'Drought': 0.13, 'humidity_temp_index': -0.19, 'PM2.5': -0.27, 'Ozone': -0.06}


# In[107]:


variable = ['mean_temp', 'visibility', 'wind_speed','precipitation', 'fog',
            'rain', 'snow', 'hail', 'thunder', 'tornado', 'Drought', 'humidity_temp_index', 'PM2.5', 'Ozone']
for col in variable:
    df_env_sd_rural[col] = df_env_sd_rural[col] * weight_rural[col]


# In[108]:


df_env_sd_rural['mean_temp_percentile'] = (df_env_sd_rural['mean_temp'] - df_env_sd_rural['mean_temp'].min()) / (df_env_sd_rural['mean_temp'].max() - df_env_sd_rural['mean_temp'].min())
df_env_sd_rural['visibility_percentile'] = (df_env_sd_rural['visibility'] - df_env_sd_rural['visibility'].min()) / (df_env_sd_rural['visibility'].max() - df_env_sd_rural['visibility'].min())
df_env_sd_rural['wind_speed_percentile'] = (df_env_sd_rural['wind_speed'] - df_env_sd_rural['wind_speed'].min()) / (df_env_sd_rural['wind_speed'].max() - df_env_sd_rural['wind_speed'].min())
df_env_sd_rural['precipitation_percentile'] = (df_env_sd_rural['precipitation'] - df_env_sd_rural['precipitation'].min()) / (df_env_sd_rural['precipitation'].max() - df_env_sd_rural['precipitation'].min())
df_env_sd_rural['fog_percentile'] = (df_env_sd_rural['fog'] - df_env_sd_rural['fog'].min()) / (df_env_sd_rural['fog'].max() - df_env_sd_rural['fog'].min())
df_env_sd_rural['rain_percentile'] = (df_env_sd_rural['rain'] - df_env_sd_rural['rain'].min()) / (df_env_sd_rural['rain'].max() - df_env_sd_rural['rain'].min())
df_env_sd_rural['snow_percentile'] = (df_env_sd_rural['snow'] - df_env_sd_rural['snow'].min()) / (df_env_sd_rural['snow'].max() - df_env_sd_rural['snow'].min())
df_env_sd_rural['hail_percentile'] = (df_env_sd_rural['hail'] - df_env_sd_rural['hail'].min()) / (df_env_sd_rural['hail'].max() - df_env_sd_rural['hail'].min())
df_env_sd_rural['thunder_percentile'] = (df_env_sd_rural['thunder'] - df_env_sd_rural['thunder'].min()) / ( df_env_sd_rural['thunder'].max() -  df_env_sd_rural['thunder'].min())
df_env_sd_rural['tornado_percentile'] = (df_env_sd_rural['tornado'] - df_env_sd_rural['tornado'].min()) / (df_env_sd_rural['tornado'].max() - df_env_sd_rural['tornado'].min())
df_env_sd_rural['Drought_percentile'] = (df_env_sd_rural['Drought'] - df_env_sd_rural['Drought'].min()) / (df_env_sd_rural['Drought'].max() - df_env_sd_rural['Drought'].min())
df_env_sd_rural['humidity_temp_index_percentile'] = (df_env_sd_rural['humidity_temp_index'] - df_env_sd_rural['humidity_temp_index'].min()) / (df_env_sd_rural['humidity_temp_index'].max() - df_env_sd_rural['humidity_temp_index'].min())
df_env_sd_rural['PM2.5_percentile'] = (df_env_sd_rural['PM2.5'] - df_env_sd_rural['PM2.5'].min()) / (df_env_sd_rural['PM2.5'].max() - df_env_sd_rural['PM2.5'].min())
df_env_sd_rural['Ozone_percentile'] = (df_env_sd_rural['Ozone'] - df_env_sd_rural['Ozone'].min()) / (df_env_sd_rural['Ozone'].max() - df_env_sd_rural['Ozone'].min())


# In[109]:


df_env_sd_rural['ClimateScore'] = df_env_sd_rural['mean_temp_percentile'] + df_env_sd_rural['visibility_percentile'] + df_env_sd_rural['wind_speed_percentile'] + df_env_sd_rural['precipitation_percentile']+ df_env_sd_rural['fog_percentile'] + df_env_sd_rural['rain_percentile'] + df_env_sd_rural['snow_percentile'] + df_env_sd_rural['hail_percentile'] + df_env_sd_rural['thunder_percentile'] + df_env_sd_rural['tornado_percentile']
df_env_sd_rural['AirQualityScore'] = df_env_sd_rural['PM2.5_percentile'] + df_env_sd_rural['Ozone_percentile']
df_env_sd_rural['ClimateScore_percentile'] = (df_env_sd_rural['ClimateScore'] - df_env_sd_rural['ClimateScore'].min()) / (df_env_sd_rural['ClimateScore'].max() - df_env_sd_rural['ClimateScore'].min())
df_env_sd_rural['AirQuality_percentile'] = (df_env_sd_rural['AirQualityScore'] - df_env_sd_rural['AirQualityScore'].min()) / (df_env_sd_rural['AirQualityScore'].max() - df_env_sd_rural['AirQualityScore'].min())
df_env_sd_rural['EnvironmentScore'] = df_env_sd_rural['ClimateScore_percentile'] + df_env_sd_rural['AirQuality_percentile']
df_env_sd_rural['MEDI_Rural'] = (df_env_sd_rural['EnvironmentScore'] - df_env_sd_rural['EnvironmentScore'].min()) / (df_env_sd_rural['EnvironmentScore'].max() - df_env_sd_rural['EnvironmentScore'].min())


# In[110]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_env_sd_rural, geojson=counties, locations='fips', color='MEDI_Rural',
                           color_continuous_scale="rainbow",
                           range_color=(0, 1),
                           scope="usa",
                           hover_name="county",
                           labels={'MEDI':'MEDI'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[111]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_env_rural_original, geojson=counties, locations='fips', color='MEDI_Rural_Overall',
                           color_continuous_scale="rainbow",
                           range_color=(0, 1),
                           scope="usa",
                           hover_name="county",
                           labels={'MEDI':'MEDI'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# ### Plot out MEDI

# In[112]:


df_env_urban = df_env_urban.rename(columns = {'MEDI': 'MEDI_Overall'})


# In[113]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_env_urban, geojson=counties, locations='fips', color='MEDI_Overall',
                           color_continuous_scale="rainbow",
                           range_color=(0, 1),
                           scope="usa",
                           hover_name="county",
                           labels={'MEDI':'MEDI'}
                          )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[114]:


df_env_sd_urban = df_env_sd_urban.rename(columns = {'MEDI_Urban': 'MEDI'})
df_env_sd_rural = df_env_sd_rural.rename(columns = {'MEDI_Rural': 'MEDI'})


# In[115]:


df_urban_rural = pd.concat([df_env_sd_urban, df_env_sd_rural], axis = 0 )


# In[116]:


df_urban_rural


# In[117]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_urban_rural, geojson=counties, locations='fips', color='MEDI',
                           color_continuous_scale="rainbow",
                           range_color=(0, 1),
                           scope="usa",
                           hover_name="county",
                           labels={'MEDI':'MEDI'}
                          )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[118]:


df_urban_rural.columns


# ### upload suicide rate 2017

# In[119]:


df_sd = pd.read_csv(r'SuicideRate_Imputed/SuicideRate_Imputed_2017.csv',dtype={"fips": str})


# In[120]:


df_urban_rural = df_urban_rural.drop(['SuicideDeathRate'], axis = 1)
df_urban_rural.columns


# In[121]:


df_sd = df_sd[['county', 'fips', 'SuicideDeathRate']]
df_sd


# In[122]:


df_MEDI_combined = df_urban_rural.merge(df_sd, on = ['fips'], how = 'left')


# In[123]:


df_MEDI_combined


# In[124]:


df_MEDI2 = df_MEDI_combined.dropna(how = 'any')


# In[125]:


y = df_MEDI2['MEDI'].values.reshape(-1, 1)
X = df_MEDI2['SuicideDeathRate'].values.reshape(-1, 1)
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


# In[126]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_MEDI = mean_absolute_error(y_test, y_pred)
mse_MEDI = mean_squared_error(y_test, y_pred)
rmse_MEDI = np.sqrt(mse_MEDI)
print(f'Mean absolute error: {mae_MEDI:.2f}')
print(f'Mean squared error: {mse_MEDI:.2f}')
print(f'Root mean squared error: {rmse_MEDI:.2f}')


# ### check RMSE for MEDI before dividing to rural and urban

# In[127]:


df_env_urban


# In[128]:


df_env_urban.columns


# In[129]:


df_MEDI_before = df_env_urban.merge(df_sd, on = ['fips'], how = 'left')
df_MEDI_before


# In[130]:


df_MEDI3 = df_MEDI_before.dropna(how = 'any')


# In[131]:


y = df_MEDI3['MEDI_Overall'].values.reshape(-1, 1)
X = df_MEDI3['SuicideDeathRate'].values.reshape(-1, 1)
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


# In[132]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_MEDI = mean_absolute_error(y_test, y_pred)
mse_MEDI = mean_squared_error(y_test, y_pred)
rmse_MEDI = np.sqrt(mse_MEDI)
print(f'Mean absolute error: {mae_MEDI:.2f}')
print(f'Mean squared error: {mse_MEDI:.2f}')
print(f'Root mean squared error: {rmse_MEDI:.2f}')


# ### check RMSE for urban counties

# In[133]:


df_env_sd_urban.columns


# In[134]:


df_env_sd_urban = df_env_sd_urban.drop(['SuicideDeathRate'], axis = 1)
df_MEDI_urban = df_env_sd_urban.merge(df_sd, on = ['fips'], how = 'left')
df_MEDI4 = df_MEDI_urban.dropna(how = 'any')


# In[135]:


y = df_MEDI4['MEDI'].values.reshape(-1, 1)
X = df_MEDI4['SuicideDeathRate'].values.reshape(-1, 1)
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


# In[136]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_MEDI = mean_absolute_error(y_test, y_pred)
mse_MEDI = mean_squared_error(y_test, y_pred)
rmse_MEDI = np.sqrt(mse_MEDI)
print(f'Mean absolute error: {mae_MEDI:.2f}')
print(f'Mean squared error: {mse_MEDI:.2f}')
print(f'Root mean squared error: {rmse_MEDI:.2f}')


# ### check RMSE for rural areas

# In[137]:


df_env_sd_rural.columns


# In[138]:


df_env_sd_rural = df_env_sd_rural.drop(['SuicideDeathRate'], axis = 1)
df_MEDI_rural = df_env_sd_rural.merge(df_sd, on = ['fips'], how = 'left')
df_MEDI5 = df_MEDI_rural.dropna(how = 'any')


# In[139]:


y = df_MEDI5['MEDI'].values.reshape(-1, 1)
X = df_MEDI5['SuicideDeathRate'].values.reshape(-1, 1)
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


# In[140]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_MEDI = mean_absolute_error(y_test, y_pred)
mse_MEDI = mean_squared_error(y_test, y_pred)
rmse_MEDI = np.sqrt(mse_MEDI)
print(f'Mean absolute error: {mae_MEDI:.2f}')
print(f'Mean squared error: {mse_MEDI:.2f}')
print(f'Root mean squared error: {rmse_MEDI:.2f}')


# ### plot out the SVI 

# In[141]:


df_percentile.columns


# In[142]:


df_percentile['county'] = df_percentile['county'] + ',' + df_percentile['state']


# In[143]:


df_percentile = df_percentile.rename(columns = {'percentile_rank_socioeconomic_theme': 'socioeconomic_theme', 'percentile_rank_household_comp_disability_theme': 'household_comp_disability_theme',
                                               'percentile_rank_minority_status_and_language_theme': 'minority_status_and_language_theme', 'percentile_rank_housing_and_transportation': 'housing_and_transportation',
                                               'percentile_rank_social_vulnerability': 'SVI'})


# In[144]:


df_percentile.head()


# In[145]:


df_urban


# In[146]:


df_percentile.shape[0]


# In[147]:


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


# In[148]:


df_SVI = df_percentile[['fips', 'county', 'SVI']]


# In[149]:


df_SVI = df_SVI.merge(df_urban, on = 'fips', how = 'left')


# In[150]:


df_SVI


# In[151]:


df_SVI_urban = df_SVI[df_SVI['RuralOrUrban'] == 'Urban']
df_SVI_rural = df_SVI[df_SVI['RuralOrUrban'] == 'Rural']


# ### Combine SVI and MEDI in urban counties

# In[152]:


df_SVI_urban = df_SVI_urban[['fips', 'SVI']]
df_SVI_rural = df_SVI_rural[['fips', 'SVI']]


# In[153]:


df_env_sd_urban


# In[154]:


df_SVI_MEDI_urban = df_env_sd_urban.merge(df_SVI_urban, on = 'fips', how = 'left')
df_SVI_MEDI_urban.head()


# In[155]:


df_SVI_MEDI_urban['totalIndex'] = df_SVI_MEDI_urban['SVI'] + df_SVI_MEDI_urban['MEDI']


# In[156]:


df_SVI_MEDI_urban['SVI+MEDI_Urban'] = (df_SVI_MEDI_urban['totalIndex'] - df_SVI_MEDI_urban['totalIndex'].min()) / (df_SVI_MEDI_urban['totalIndex'].max() - df_SVI_MEDI_urban['totalIndex'].min())


# In[157]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_SVI_MEDI_urban, geojson=counties, locations='fips', color='SVI+MEDI_Urban',
                           color_continuous_scale="rainbow",
                           range_color=(0, 1),
                           scope="usa",
                           hover_name="county",
                           labels={'SVI_MEDI_Urban':'SVI+MEDI_Urban'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[158]:


df_SVI_urban = df_SVI_urban.rename(columns = {'SVI': 'SVI_Urban'})


# In[159]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_SVI_urban, geojson=counties, locations='fips', color='SVI_Urban',
                           color_continuous_scale="rainbow",
                           range_color=(0, 1),
                           scope="usa",
                           labels={'SVI_Urban':'SVI_Urban'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# #### Combine SVI and MEDI in rural counties

# In[160]:


df_SVI_MEDI_rural = df_env_sd_rural.merge(df_SVI_rural, on = 'fips', how = 'left')
df_SVI_MEDI_rural.head()


# In[161]:


df_SVI_MEDI_rural['totalIndex'] = df_SVI_MEDI_rural['SVI'] + df_SVI_MEDI_rural['MEDI']


# In[162]:


df_SVI_MEDI_rural['SVI+MEDI_Rural'] = (df_SVI_MEDI_rural['totalIndex'] - df_SVI_MEDI_rural['totalIndex'].min()) / (df_SVI_MEDI_rural['totalIndex'].max() - df_SVI_MEDI_rural['totalIndex'].min())


# In[163]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_SVI_MEDI_rural, geojson=counties, locations='fips', color='SVI+MEDI_Rural',
                           color_continuous_scale="rainbow",
                           range_color=(0, 1),
                           scope="usa",
                           hover_name="county",
                           labels={'SVI_MEDI_Rural':'SVI+MEDI_Rural'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[164]:


df_SVI_rural = df_SVI_rural.rename(columns = {'SVI': 'SVI_Rural'})


# In[165]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_SVI_rural, geojson=counties, locations='fips', color='SVI_Rural',
                           color_continuous_scale="rainbow",
                           range_color=(0, 1),
                           scope="usa",
                           labels={'SVI_Rural':'SVI_Rural'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# ### check RMSE for SVI before dividing to rural and urban

# In[166]:


df_sd


# In[167]:


df_SVI_sd = df_SVI.merge(df_sd, on = 'fips', how = 'left')
df_SVI_sd


# In[168]:


df_SVI_sd = df_SVI_sd.drop(['county_x'], axis = 1)
df_SVI_sd = df_SVI_sd.rename(columns = {'county_y': 'county'})
df_SVI_sd


# In[169]:


df_SVI_sd = df_SVI_sd.dropna(how = 'any')


# In[170]:


y = df_SVI_sd['SVI'].values.reshape(-1, 1)
X = df_SVI_sd['SuicideDeathRate'].values.reshape(-1, 1)
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


# In[171]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_MEDI = mean_absolute_error(y_test, y_pred)
mse_MEDI = mean_squared_error(y_test, y_pred)
rmse_MEDI = np.sqrt(mse_MEDI)
print(f'Mean absolute error: {mae_MEDI:.2f}')
print(f'Mean squared error: {mse_MEDI:.2f}')
print(f'Root mean squared error: {rmse_MEDI:.2f}')


# In[172]:


df_SVI_MEDI_sd_urban = df_SVI_MEDI_urban.merge(df_sd, on = 'fips', how = 'left')
df_SVI_MEDI_sd_urban


# In[173]:


df_SVI_MEDI_sd_urban = df_SVI_MEDI_sd_urban.dropna(how = 'any')


# In[174]:


y = df_SVI_MEDI_sd_urban['SVI+MEDI_Urban'].values.reshape(-1, 1)
X = df_SVI_MEDI_sd_urban['SuicideDeathRate'].values.reshape(-1, 1)
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


# In[175]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_MEDI = mean_absolute_error(y_test, y_pred)
mse_MEDI = mean_squared_error(y_test, y_pred)
rmse_MEDI = np.sqrt(mse_MEDI)
print(f'Mean absolute error: {mae_MEDI:.2f}')
print(f'Mean squared error: {mse_MEDI:.2f}')
print(f'Root mean squared error: {rmse_MEDI:.2f}')


# In[176]:


df_SVI_MEDI_sd_rural = df_SVI_MEDI_rural.merge(df_sd, on = 'fips', how = 'left')
df_SVI_MEDI_sd_rural


# In[177]:


df_SVI_MEDI_sd_rural = df_SVI_MEDI_sd_rural.dropna(how = 'any')


# In[178]:


y = df_SVI_MEDI_sd_rural['SVI+MEDI_Rural'].values.reshape(-1, 1)
X = df_SVI_MEDI_sd_rural['SuicideDeathRate'].values.reshape(-1, 1)
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


# In[179]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_MEDI = mean_absolute_error(y_test, y_pred)
mse_MEDI = mean_squared_error(y_test, y_pred)
rmse_MEDI = np.sqrt(mse_MEDI)
print(f'Mean absolute error: {mae_MEDI:.2f}')
print(f'Mean squared error: {mse_MEDI:.2f}')
print(f'Root mean squared error: {rmse_MEDI:.2f}')


# In[180]:


df_SVI_MEDI_sd_rural = df_SVI_MEDI_sd_rural.rename(columns = {'SVI+MEDI_Rural': 'SVI+MEDI'})
df_SVI_MEDI_sd_urban = df_SVI_MEDI_sd_urban.rename(columns = {'SVI+MEDI_Urban': 'SVI+MEDI'})


# In[181]:


df_SVI_MEDI = pd.concat([df_SVI_MEDI_sd_urban, df_SVI_MEDI_sd_rural], axis = 0)


# In[184]:


df_SVI_MEDI


# In[185]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_SVI_MEDI, geojson=counties, locations='fips', color='SVI+MEDI',
                           color_continuous_scale="rainbow",
                           range_color=(0, 1),
                           scope="usa",
                           labels={'SVI+MEDI':'SVI+MEDI'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# ### Create SEDI 

# In[186]:


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


# In[187]:


df_svi_county = df_svi_2016.groupby(['state', 'county', 'fips']).mean()
df_svi_county = df_svi_county.reset_index()
df_svi_county.head()


# In[188]:


df_variable = df_svi_county[['state','county','fips','percent_below_poverty', 'per_capita_income','percent_unemployed_CDC', 
                      'percent_no_highschool_diploma', 'percent_age_65_and_older', 'percent_age_17_and_younger', 
                     'percent_disabled', 'percent_single_parent_households_CDC', 'percent_minorities', 'percent_limited_english_abilities', 
                      'percent_multi_unit_housing','percent_mobile_homes', 'percent_overcrowding', 'percent_no_vehicle', 
                      'percent_institutionalized_in_group_quarters']]


# In[189]:


df_sd_16 = pd.read_csv(r'SuicideRate_Imputed/SuicideRate_Imputed_2016.csv',dtype={"fips": str})
df_sd_16 = df_sd_16[['county', 'fips', 'SuicideDeathRate']]
df_sd_16


# In[190]:


df_SEDI = df_variable.merge(df_sd_16, on = 'fips', how = 'left')


# In[191]:


df_SEDI = df_SEDI.drop(['county_x'], axis = 1)
df_SEDI = df_SEDI.rename(columns = {'county_y': 'county'})


# In[192]:


correlation_matrix(df_SEDI)


# In[193]:


weight = {'percent_below_poverty': 0.06, 'per_capita_income': -0.11, 'percent_unemployed_CDC': -0.06, 'percent_no_highschool_diploma': -0.04, 
         'percent_age_65_and_older': 0.17, 'percent_age_17_and_younger': -0.01, 'percent_disabled': 0.12, 'percent_single_parent_households_CDC': -0.10,
         'percent_minorities': -0.11, 'percent_limited_english_abilities': -0.13, 'percent_multi_unit_housing': -0.18, 'percent_mobile_homes':0.10, 
         'percent_overcrowding': 0.14}


# In[194]:


cols = df_SEDI.columns


# In[195]:


cols_list = cols[2:-4]
cols_list


# In[196]:


for col in cols_list:
    df_SEDI[col] = df_SEDI[col] * weight[col]


# In[197]:


col_percentile = []
for col in cols_list:
    col_per = col + '_percentile'
    col_percentile.append(col_per)
    
col_percentile


# In[198]:


for col_per, col in zip(col_percentile, cols_list):
    df_SEDI[col_per] = (df_SEDI[col] - df_SEDI[col].min())/(df_SEDI[col].max() - df_SEDI[col].min())


# In[199]:


df_SEI = df_SEDI


# In[200]:


df_SEI['Socialeconomic_Theme'] = df_SEI['percent_below_poverty_percentile'] + df_SEI['per_capita_income_percentile'] + df_SEI['percent_unemployed_CDC_percentile'] + df_SEI['percent_no_highschool_diploma_percentile']
df_SEI['Household_Theme'] = df_SEI['percent_age_65_and_older_percentile'] + df_SEI['percent_age_17_and_younger_percentile'] + df_SEI['percent_disabled_percentile'] + df_SEI['percent_single_parent_households_CDC_percentile']
df_SEI['Minority_Theme'] = df_SEI['percent_minorities_percentile'] + df_SEI['percent_limited_english_abilities_percentile']
df_SEI['HousingType_Theme'] = df_SEI['percent_multi_unit_housing_percentile'] + df_SEI['percent_mobile_homes_percentile'] + df_SEI['percent_overcrowding_percentile']


# In[201]:


df_SEI['Socialeconomic_Theme_percentile'] = (df_SEI['Socialeconomic_Theme'] - df_SEI['Socialeconomic_Theme'].min())/(df_SEI['Socialeconomic_Theme'].max() - df_SEI['Socialeconomic_Theme'].min())
df_SEI['Household_Theme_percentile'] = (df_SEI['Household_Theme'] - df_SEI['Household_Theme'].min())/(df_SEI['Household_Theme'].max() - df_SEI['Household_Theme'].min())
df_SEI['Minority_Theme_percentile'] = (df_SEI['Minority_Theme'] - df_SEI['Minority_Theme'].min())/(df_SEI['Minority_Theme'].max() - df_SEI['Minority_Theme'].min())
df_SEI['HousingType_Theme_percentile'] = (df_SEI['HousingType_Theme'] - df_SEI['HousingType_Theme'].min())/(df_SEI['HousingType_Theme'].max() - df_SEI['HousingType_Theme'].min())


# In[202]:


df_SEI['SEI_Score'] = df_SEI['Socialeconomic_Theme_percentile'] + df_SEI['Household_Theme_percentile'] + df_SEI['Minority_Theme_percentile'] + df_SEI['HousingType_Theme_percentile']
df_SEI['SEDI'] = (df_SEI['SEI_Score'] - df_SEI['SEI_Score'].min())/(df_SEI['SEI_Score'].max() - df_SEI['SEI_Score'].min())


# In[203]:


df_SEI_overall = df_SEI.rename(columns = {'SEDI': 'SEDI_Overall'})


# In[204]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_SEI_overall, geojson=counties, locations='fips', color='SEDI_Overall',
                           color_continuous_scale="rainbow",
                           range_color=(0.25, 0.75),
                           scope="usa",
                           hover_name="county",
                           labels={'SEDI_Overall':'SEDI_Overall'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[243]:


df_SEI_overall = df_SEI_overall.drop(['SuicideDeathRate'], axis = 1)


# In[244]:


df_SEDI_sd = df_SEI_overall.merge(df_sd, on = 'fips', how = 'left')
df_SEDI_sd.head()


# In[245]:


df_SEDI_sd = df_SEDI_sd.dropna(how = 'any')


# In[246]:


y = df_SEDI_sd['SEDI_Overall'].values.reshape(-1, 1)
X = df_SEDI_sd['SuicideDeathRate'].values.reshape(-1, 1)
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


# In[247]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_MEDI = mean_absolute_error(y_test, y_pred)
mse_MEDI = mean_squared_error(y_test, y_pred)
rmse_MEDI = np.sqrt(mse_MEDI)
print(f'Mean absolute error: {mae_MEDI:.2f}')
print(f'Mean squared error: {mse_MEDI:.2f}')
print(f'Root mean squared error: {rmse_MEDI:.2f}')


# ### Diving to rual and urban two part

# In[205]:


df_variable


# In[206]:


df_urban_new = df_urban[['fips', 'RuralOrUrban']]


# In[207]:


df_variable_urban = df_variable.merge(df_urban_new, on = 'fips', how = 'left')


# In[208]:


df_variable_urban


# In[209]:


df_SEDI_Urban = df_variable_urban[df_variable_urban['RuralOrUrban'] == 'Urban']
df_SEDI_Rural = df_variable_urban[df_variable_urban['RuralOrUrban'] == 'Rural']


# ### create SEDI in urban counties

# In[256]:


df_SEDI_urban = df_SEDI_Urban.merge(df_sd_16, on = 'fips', how = 'left')
df_SEDI_urban = df_SEDI_urban.drop(['county_x'], axis = 1)
df_SEDI_urban = df_SEDI_urban.rename(columns = {'county_y': 'county'})


# In[257]:


correlation_matrix(df_SEDI_urban)


# In[258]:


weight_urban = {'percent_below_poverty': 0.04, 'per_capita_income': -0.25, 'percent_unemployed_CDC': -0.03, 'percent_no_highschool_diploma': -0.01, 
         'percent_age_65_and_older': 0.30, 'percent_age_17_and_younger': -0.12, 'percent_disabled': 0.38, 'percent_single_parent_households_CDC': -0.18,
         'percent_minorities': -0.37, 'percent_limited_english_abilities': -0.32, 'percent_multi_unit_housing': -0.33, 'percent_mobile_homes':0.27, 
         'percent_overcrowding': -0.19, 'percent_no_vehicle': -0.25, 'percent_institutionalized_in_group_quarters': -0.03}


# In[259]:


cols = df_SEDI_urban.columns
cols


# In[260]:


cols_list = cols[2:-3]
cols_list


# In[261]:


for col in cols_list:
    df_SEDI_urban[col] = df_SEDI_urban[col] * weight_urban[col]


# In[262]:


col_percentile = []
for col in cols_list:
    col_per = col + '_percentile'
    col_percentile.append(col_per)
    
col_percentile


# In[263]:


for col_per, col in zip(col_percentile, cols_list):
    df_SEDI_urban[col_per] = (df_SEDI_urban[col] - df_SEDI_urban[col].min())/(df_SEDI_urban[col].max() - df_SEDI_urban[col].min())


# In[264]:


df_SEDI_urban['Socialeconomic_Theme'] = df_SEDI_urban['percent_below_poverty_percentile'] + df_SEDI_urban['per_capita_income_percentile'] + df_SEDI_urban['percent_unemployed_CDC_percentile'] + df_SEDI_urban['percent_no_highschool_diploma_percentile']
df_SEDI_urban['Household_Theme'] = df_SEDI_urban['percent_age_65_and_older_percentile'] + df_SEDI_urban['percent_age_17_and_younger_percentile'] + df_SEDI_urban['percent_disabled_percentile'] + df_SEDI_urban['percent_single_parent_households_CDC_percentile']
df_SEDI_urban['Minority_Theme'] = df_SEDI_urban['percent_minorities_percentile'] + df_SEDI_urban['percent_limited_english_abilities_percentile']
df_SEDI_urban['HousingType_Theme'] = df_SEDI_urban['percent_multi_unit_housing_percentile'] + df_SEDI_urban['percent_mobile_homes_percentile'] + df_SEDI_urban['percent_overcrowding_percentile'] + df_SEDI_urban['percent_no_vehicle_percentile'] + df_SEDI_urban['percent_institutionalized_in_group_quarters_percentile']


# In[265]:


df_SEDI_urban['Socialeconomic_Theme_percentile'] = (df_SEDI_urban['Socialeconomic_Theme'] - df_SEDI_urban['Socialeconomic_Theme'].min())/(df_SEDI_urban['Socialeconomic_Theme'].max() - df_SEDI_urban['Socialeconomic_Theme'].min())
df_SEDI_urban['Household_Theme_percentile'] = (df_SEDI_urban['Household_Theme'] - df_SEDI_urban['Household_Theme'].min())/(df_SEDI_urban['Household_Theme'].max() - df_SEDI_urban['Household_Theme'].min())
df_SEDI_urban['Minority_Theme_percentile'] = (df_SEDI_urban['Minority_Theme'] - df_SEDI_urban['Minority_Theme'].min())/(df_SEDI_urban['Minority_Theme'].max() - df_SEDI_urban['Minority_Theme'].min())
df_SEDI_urban['HousingType_Theme_percentile'] = (df_SEDI_urban['HousingType_Theme'] - df_SEDI_urban['HousingType_Theme'].min())/(df_SEDI_urban['HousingType_Theme'].max() - df_SEDI_urban['HousingType_Theme'].min())


# In[266]:


df_SEDI_urban['SEI_Score'] = df_SEDI_urban['Socialeconomic_Theme_percentile'] + df_SEDI_urban['Household_Theme_percentile'] + df_SEDI_urban['Minority_Theme_percentile'] + df_SEDI_urban['HousingType_Theme_percentile']
df_SEDI_urban['SEDI_Urban'] = (df_SEDI_urban['SEI_Score'] - df_SEDI_urban['SEI_Score'].min())/(df_SEDI_urban['SEI_Score'].max() - df_SEDI_urban['SEI_Score'].min())


# In[267]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_SEDI_urban, geojson=counties, locations='fips', color='SEDI_Urban',
                           color_continuous_scale="rainbow",
                           range_color=(0.25, 0.75),
                           scope="usa",
                           hover_name="county",
                           labels={'SEDI_Urban':'SEDI_Urban'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[268]:


df_SEDI_urban = df_SEDI_urban.drop(['SuicideDeathRate'], axis = 1)


# In[269]:


df_SEDI_urban = df_SEDI_urban.merge(df_sd, on = 'fips', how = 'left')
df_SEDI_urban.head()


# In[270]:


df_SEDI_urban = df_SEDI_urban.dropna(how = 'any')


# In[272]:


y = df_SEDI_urban['SEDI_Urban'].values.reshape(-1, 1)
X = df_SEDI_urban['SuicideDeathRate'].values.reshape(-1, 1)
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


# In[273]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_MEDI = mean_absolute_error(y_test, y_pred)
mse_MEDI = mean_squared_error(y_test, y_pred)
rmse_MEDI = np.sqrt(mse_MEDI)
print(f'Mean absolute error: {mae_MEDI:.2f}')
print(f'Mean squared error: {mse_MEDI:.2f}')
print(f'Root mean squared error: {rmse_MEDI:.2f}')


# ### create SEDI in rural counties

# In[223]:


df_SEDI_rural = df_SEDI_Rural.merge(df_sd_16, on = 'fips', how = 'left')
df_SEDI_rural = df_SEDI_rural.drop(['county_x'], axis = 1)
df_SEDI_rural = df_SEDI_rural.rename(columns = {'county_y': 'county'})


# In[224]:


correlation_matrix(df_SEDI_rural)


# In[225]:


weight_rural = {'percent_below_poverty': -0.01, 'per_capita_income': -0.05, 'percent_unemployed_CDC': -0.07, 'percent_no_highschool_diploma': -0.12, 
         'percent_age_65_and_older': 0.04, 'percent_age_17_and_younger': 0.04, 'percent_disabled': -0.03, 'percent_single_parent_households_CDC': -0.05,
         'percent_minorities': -0.02, 'percent_limited_english_abilities': -0.06, 'percent_multi_unit_housing': 0, 'percent_mobile_homes': -0.02, 
         'percent_overcrowding': 0.21, 'percent_no_vehicle': 0.06, 'percent_institutionalized_in_group_quarters': -0.03}


# In[226]:


cols = df_SEDI_rural.columns


# In[227]:


cols = cols.tolist()


# In[228]:


cols.pop(12)


# In[229]:


cols


# In[230]:


cols_list = cols[2:-3]
cols_list


# In[231]:


for col in cols_list:
    df_SEDI_rural[col] = df_SEDI_rural[col] * weight_rural[col]


# In[232]:


col_percentile = []
for col in cols_list:
    col_per = col + '_percentile'
    col_percentile.append(col_per)
    
col_percentile


# In[233]:


for col_per, col in zip(col_percentile, cols_list):
    df_SEDI_rural[col_per] = (df_SEDI_rural[col] - df_SEDI_rural[col].min())/(df_SEDI_rural[col].max() - df_SEDI_rural[col].min())


# In[234]:


df_SEDI_rural['Socialeconomic_Theme'] = df_SEDI_rural['percent_below_poverty_percentile'] + df_SEDI_rural['per_capita_income_percentile'] + df_SEDI_rural['percent_unemployed_CDC_percentile'] + df_SEDI_rural['percent_no_highschool_diploma_percentile']
df_SEDI_rural['Household_Theme'] = df_SEDI_rural['percent_age_65_and_older_percentile'] + df_SEDI_rural['percent_age_17_and_younger_percentile'] + df_SEDI_rural['percent_disabled_percentile'] + df_SEDI_rural['percent_single_parent_households_CDC_percentile']
df_SEDI_rural['Minority_Theme'] = df_SEDI_rural['percent_minorities_percentile'] + df_SEDI_rural['percent_limited_english_abilities_percentile']
df_SEDI_rural['HousingType_Theme'] = df_SEDI_rural['percent_mobile_homes_percentile'] + df_SEDI_rural['percent_overcrowding_percentile'] + df_SEDI_rural['percent_no_vehicle_percentile'] + df_SEDI_rural['percent_institutionalized_in_group_quarters_percentile']


# In[235]:


df_SEDI_rural['Socialeconomic_Theme_percentile'] = (df_SEDI_rural['Socialeconomic_Theme'] - df_SEDI_rural['Socialeconomic_Theme'].min())/(df_SEDI_rural['Socialeconomic_Theme'].max() - df_SEDI_rural['Socialeconomic_Theme'].min())
df_SEDI_rural['Household_Theme_percentile'] = (df_SEDI_rural['Household_Theme'] - df_SEDI_rural['Household_Theme'].min())/(df_SEDI_rural['Household_Theme'].max() - df_SEDI_rural['Household_Theme'].min())
df_SEDI_rural['Minority_Theme_percentile'] = (df_SEDI_rural['Minority_Theme'] - df_SEDI_rural['Minority_Theme'].min())/(df_SEDI_rural['Minority_Theme'].max() - df_SEDI_rural['Minority_Theme'].min())
df_SEDI_rural['HousingType_Theme_percentile'] = (df_SEDI_rural['HousingType_Theme'] - df_SEDI_rural['HousingType_Theme'].min())/(df_SEDI_rural['HousingType_Theme'].max() - df_SEDI_rural['HousingType_Theme'].min())


# In[236]:


df_SEDI_rural['SEI_Score'] = df_SEDI_rural['Socialeconomic_Theme_percentile'] + df_SEDI_rural['Household_Theme_percentile'] + df_SEDI_rural['Minority_Theme_percentile'] + df_SEDI_rural['HousingType_Theme_percentile']
df_SEDI_rural['SEDI_Rural'] = (df_SEDI_rural['SEI_Score'] - df_SEDI_rural['SEI_Score'].min())/(df_SEDI_rural['SEI_Score'].max() - df_SEDI_rural['SEI_Score'].min())


# In[237]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(df_SEDI_rural, geojson=counties, locations='fips', color='SEDI_Rural',
                           color_continuous_scale="rainbow",
                           range_color=(0.25, 0.75),
                           scope="usa",
                           hover_name="county",
                           labels={'SEDI_Urban':'SEDI_Rural'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[274]:


df_SEDI_rural = df_SEDI_rural.drop(['SuicideDeathRate'], axis = 1)


# In[276]:


df_SEDI_rural = df_SEDI_rural.merge(df_sd, on = 'fips', how = 'left')
df_SEDI_rural.head()


# In[277]:


df_SEDI_rural = df_SEDI_rural.dropna(how = 'any')


# In[278]:


y = df_SEDI_rural['SEDI_Rural'].values.reshape(-1, 1)
X = df_SEDI_rural['SuicideDeathRate'].values.reshape(-1, 1)
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


# In[279]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_MEDI = mean_absolute_error(y_test, y_pred)
mse_MEDI = mean_squared_error(y_test, y_pred)
rmse_MEDI = np.sqrt(mse_MEDI)
print(f'Mean absolute error: {mae_MEDI:.2f}')
print(f'Mean squared error: {mse_MEDI:.2f}')
print(f'Root mean squared error: {rmse_MEDI:.2f}')


# ### Evaluation and Comparison
