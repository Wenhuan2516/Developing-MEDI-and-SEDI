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


ur_files = ddf.read_csv(r"Temperature/US_counties_weather_2018.csv", dtype={'eightieth_percentile_income': 'float64','fips': 'object','fog': 'float64','hail': 'float64',
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
df_weather = ur_files.compute()
df_weather.head()


# In[4]:


cols = df_weather.columns


# In[5]:


for col in cols:
    if 'income' in col:
        print(col)


# In[6]:


for col in cols:
    print(col)


# In[7]:


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


# ### Find the climate data part

# In[8]:


df_temp_2018 = df_weather[['state', 'county', 'fips','station_id', 'station_name', 'date', 'mean_temp', 
                        'min_temp', 'max_temp', 'dewpoint', 'sea_level_pressure', 'visibility', 'wind_speed', 'wind_gust', 'precipitation',
                       'precip_flag', 'fog', 'rain', 'snow', 'hail', 'thunder', 'tornado']]
df_temp_2018.head()


# In[9]:


df_temp_county = df_temp_2018.groupby(['state', 'county', 'fips']).mean()
df_temp_county = df_temp_county.reset_index()
df_temp_county.head()


# In[10]:


correlation_matrix(df_temp_county)


# #### From the above graph, we can see that max temp/min temp/dewpoint are highly correlated with mean temp, max Wind speed and max wind gust are highly correlated with wind values, so we can drop max temp, min temp, max wind speed, and max wind gust.

# In[11]:


df_temp_county3 = df_temp_county.drop(['min_temp', 'max_temp', 'wind_gust', 'dewpoint'], axis=1)
correlation_matrix(df_temp_county3)


# ### upload PM2.5 Data

# In[12]:


ur_files = ddf.read_csv(r"PM2.5.csv", dtype = {'CountyFIPS': 'object'})
df_pm = ur_files.compute()
df_pm = df_pm.loc[:, ~df_pm.columns.str.contains('^Unnamed')]
df_pm.head()


# In[13]:


df_pm_2018 = df_pm[df_pm['Year'] == 2018]
df_pm_2018 = df_pm_2018.rename(columns = {'State': 'state', 'CountyFIPS': 'fips', 'County': 'county'})
df_pm_2018 = df_pm_2018.groupby(['state', 'fips', 'county']).mean()
df_pm_2018 = df_pm_2018.reset_index()
df_pm_2018


# In[14]:


df_pm_2018 = df_pm_2018.rename(columns = {'Value': 'PM2.5'})
df_pm_2018


# ### load Ozone data

# In[15]:


ur_files = ddf.read_csv(r"Ozone.csv", dtype = {'CountyFIPS': 'object', 'Value': 'object'})
df_oz = ur_files.compute()
df_oz = df_oz.loc[:, ~df_oz.columns.str.contains('^Unnamed')]
df_oz.head()


# In[16]:


df_oz['Value'].unique()


# In[17]:


def changeValue(value):
    if value == 'No Data':
        value = None
    else:
        value = int(value)
    return value


# In[18]:


df_oz['Value'] = df_oz['Value'].apply(changeValue)
df_oz_2018 = df_oz[df_oz['Year'] == 2018]
df_oz_2018 = df_oz_2018.rename(columns = {'State': 'state', 'CountyFIPS': 'fips', 'County': 'county', 'Value': 'Ozone'})
df_oz_2018 = df_oz_2018.groupby(['state', 'fips', 'county']).mean()
df_oz_2018 = df_oz_2018.reset_index()
df_oz_2018


# #### Combine environmental factors

# In[19]:


df_air = df_pm_2018.merge(df_oz_2018, on = ['state','fips', 'Year', 'StateFIPS', 'county'], how = 'left')
df_air.head()


# In[20]:


df_air = df_air[['state', 'fips', 'county', 'PM2.5', 'Ozone']]
df_climate = df_temp_county3
df_env = df_climate.merge(df_air, on = ['state', 'fips', 'county'], how = 'left')
df_env


# ### checking missing climate data

# In[21]:


df_env['mean_temp'].isnull().sum()


# In[22]:


df_env['sea_level_pressure'].isnull().sum()


# In[23]:


df_env['visibility'].isnull().sum()


# In[24]:


df_env['wind_speed'].isnull().sum()


# In[25]:


df_env['precipitation'].isnull().sum()


# In[26]:


df_env['fog'].isnull().sum()


# In[27]:


df_env['rain'].isnull().sum()


# In[28]:


df_env['snow'].isnull().sum()


# In[29]:


df_env['hail'].isnull().sum()


# In[30]:


df_env['thunder'].isnull().sum()


# In[31]:


df_env['tornado'].isnull().sum()


# In[32]:


df_env['PM2.5'].isnull().sum()


# In[33]:


df_env['Ozone'].isnull().sum()


# ### Dropping variables when the variable missed too much data and assign the average mean value of adjacent counties to the missing county

# In[34]:


# since almost half of the counties are missing see_level_pressure data, we choose to drop it off
df_env = df_env.drop(['sea_level_pressure'], axis = 1)


# In[35]:


df_env.to_csv('EnvironmentalData_2018.csv')


# In[36]:


df_sd_18 = pd.read_csv(r'Suicide_new/Suicide_18.txt', sep='\t',dtype={"County Code": str})
df_sd_18 = df_sd_18.drop("Notes",1)


# In[ ]:





# In[37]:


df_sd_18 = df_sd_18.dropna(how = 'any')
df_sd_18['Deaths']=df_sd_18['Deaths'].astype(int)
df_sd_18['Population']=df_sd_18['Population'].astype(int)
df_sd_18['SuicideDeathRate'] = (df_sd_18['Deaths'] / df_sd_18['Population'])*100000
df_sd_18


# In[38]:


df_sd_18 = df_sd_18.rename(columns = {'County Code': 'fips', 'County': 'county'})
df_sd_18 = df_sd_18[['county', 'fips', 'SuicideDeathRate']]
df_sd_18


# In[39]:


df_env_sd = df_env.merge(df_sd_18, on = ['fips'], how = 'left')


# In[40]:


df_env_sd = df_env_sd.drop(['county_x'], axis = 1)  
df_env_sd = df_env_sd.rename(columns = {'county_y': 'county'})
df_env_sd


# In[41]:


correlation_matrix(df_env_sd)


# In[42]:


weight = {'mean_temp': -0.11, 'visibility': 0.12, 'wind_speed': -0.06, 'precipitation': -0.17, 'fog': -0.03, 'rain': -0.09, 'snow': 0.06,
         'hail': -0.01, 'thunder': -0.06, 'tornado': -0.01, 'PM2.5': -0.19, 'Ozone': -0.13}


# In[43]:


variable = ['mean_temp', 'visibility', 'wind_speed', 'precipitation', 'fog',
            'rain', 'snow', 'hail', 'tornado', 'thunder','PM2.5', 'Ozone']
for col in variable:
    df_env[col] = df_env[col] * weight[col]


# ### find percentile for each variable

# In[44]:


df_env['mean_temp_percentile'] = (df_env['mean_temp'] - df_env['mean_temp'].min()) / (df_env['mean_temp'].max() - df_env['mean_temp'].min())
df_env['visibility_percentile'] = (df_env['visibility'] - df_env['visibility'].min()) / (df_env['visibility'].max() - df_env['visibility'].min())
df_env['wind_speed_percentile'] = (df_env['wind_speed'] - df_env['wind_speed'].min()) / (df_env['wind_speed'].max() - df_env['wind_speed'].min())
df_env['precipitation_percentile'] = (df_env['precipitation'] - df_env['precipitation'].min()) / (df_env['precipitation'].max() - df_env['precipitation'].min())
df_env['fog_percentile'] = (df_env['fog'] - df_env['fog'].min()) / (df_env['fog'].max() - df_env['fog'].min())
df_env['rain_percentile'] = (df_env['rain'] - df_env['rain'].min()) / (df_env['rain'].max() - df_env['rain'].min())
df_env['snow_percentile'] = (df_env['snow'] - df_env['snow'].min()) / (df_env['snow'].max() - df_env['snow'].min())
df_env['hail_percentile'] = (df_env['hail'] - df_env['hail'].min()) / (df_env['hail'].max() - df_env['hail'].min())
df_env['tornado_percentile'] = (df_env['tornado'] - df_env['tornado'].min()) / (df_env['tornado'].max() - df_env['tornado'].min())
df_env['thunder_percentile'] = (df_env['thunder'] - df_env['thunder'].min()) / ( df_env['thunder'].max() -  df_env['thunder'].min())
df_env['PM2.5_percentile'] = (df_env['PM2.5'] - df_env['PM2.5'].min()) / (df_env['PM2.5'].max() - df_env['PM2.5'].min())
df_env['Ozone_percentile'] = (df_env['Ozone'] - df_env['Ozone'].min()) / (df_env['Ozone'].max() - df_env['Ozone'].min())


# #### give a score to climate factors and air quality factors

# In[45]:


df_env['ClimateScore'] = df_env['mean_temp_percentile'] + df_env['visibility_percentile'] + df_env['wind_speed_percentile'] + df_env['precipitation_percentile']+ df_env['fog_percentile'] + df_env['rain_percentile'] + df_env['snow_percentile'] + df_env['tornado_percentile'] + df_env['thunder_percentile']


# In[46]:


df_env['AirQualityScore'] = df_env['PM2.5_percentile'] + df_env['Ozone_percentile']


# #### give a percentile to each category

# In[47]:


df_env['ClimateScore_percentile'] = (df_env['ClimateScore'] - df_env['ClimateScore'].min()) / (df_env['ClimateScore'].max() - df_env['ClimateScore'].min())
df_env['AirQuality_percentile'] = (df_env['AirQualityScore'] - df_env['AirQualityScore'].min()) / (df_env['AirQualityScore'].max() - df_env['AirQualityScore'].min())


# #### sum the percentile of each category and assign the sum to be environment score

# In[48]:


df_env['EnvironmentScore'] = df_env['ClimateScore_percentile'] + df_env['AirQuality_percentile']


# #### MEDI is the percentile of environmental score

# In[49]:


df_env['MEDI'] = (df_env['EnvironmentScore'] - df_env['EnvironmentScore'].min()) / (df_env['EnvironmentScore'].max() - df_env['EnvironmentScore'].min())


# In[50]:


df_env.head()


# In[51]:


df_env.shape[0]


# In[52]:


df_env['MEDI'].describe()


# In[53]:


df_env['county'] = df_env['county'] + ',' + df_env['state']


# In[54]:


df_env2 = df_env.dropna(how = 'any')
df_env2.shape[0]


# ### Plot out MEDI

# In[55]:


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


# ### Evaluation and Comparison

# In[63]:


df_index = df_env[['state', 'county', 'fips', 'MEDI']]


# In[64]:


df_sd = pd.read_csv(r'Suicide_new/Suicide_19.txt', sep='\t',dtype={"County Code": str})
df_sd = df_sd.drop("Notes",1)


# In[65]:


df_sd = df_sd.dropna(how = 'any')


# In[66]:


df_sd['Deaths']=df_sd['Deaths'].astype(int)
df_sd['Population']=df_sd['Population'].astype(int)
df_sd['SuicideDeathRate'] = (df_sd['Deaths'] / df_sd['Population'])*100000
df_sd


# In[67]:


df_sd = df_sd.rename(columns = {'County Code': 'fips', 'County': 'county'})


# In[68]:


df_sd


# In[69]:


df_sd = df_sd[['county', 'fips', 'SuicideDeathRate']]


# In[70]:


df_index = df_index.merge(df_sd, on = ['fips'], how = 'left')
df_index = df_index.dropna(how = 'any')
df_index


# In[71]:


df_index = df_index[['state', 'county_y', 'fips', 'MEDI', 'SuicideDeathRate']]
df_index = df_index.rename(columns = {'county_y': 'county'})
df_index


# In[72]:


df_index['Suicide_Rate_Percentile'] = (df_index['SuicideDeathRate'] - df_index['SuicideDeathRate'].min()) / (df_index['SuicideDeathRate'].max() - df_index['SuicideDeathRate'].min())


# In[73]:


df_index


# ### Use Mean absolute error/Mean squared error/Root-Mean-Square Error to check the model performance

# ####  SVI and Suicide Rate

# In[74]:


sn.lmplot('MEDI', 'SuicideDeathRate', data = df_index)


# In[75]:


import sklearn
from sklearn.linear_model import LinearRegression
# Create a LinearRegression Object
y = df_index['SuicideDeathRate']
x_index = df_index.drop(['county', 'state', 'fips', 'SuicideDeathRate', 'Suicide_Rate_Percentile'], axis =1)
lreg = LinearRegression()
lreg.fit(x_index,y)
# Set a DataFrame from the Features
coeff_index = DataFrame(x_index.columns)
coeff_index.columns = ['Features']

# Set a new column lining up the coefficients from the linear regression
coeff_index["Coefficient Estimate"] = pd.Series(lreg.coef_)

# Show
coeff_index


# #### MEDI with suicide rate

# In[76]:


y = df_index['MEDI'].values.reshape(-1, 1)
X = df_index['SuicideDeathRate'].values.reshape(-1, 1)
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


# In[77]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae_MEDI = mean_absolute_error(y_test, y_pred)
mse_MEDI = mean_squared_error(y_test, y_pred)
rmse_MEDI = np.sqrt(mse_MEDI)
print(f'Mean absolute error: {mae_MEDI:.2f}')
print(f'Mean squared error: {mse_MEDI:.2f}')
print(f'Root mean squared error: {rmse_MEDI:.2f}')


# In[ ]:




