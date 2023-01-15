# Developing-MEDI-and-SEDI
## 12 environmental factors are combined to create a Multi-Dimensional Environmental Deprivation Index(MEDI)  and 15 social economic factors are combined to create a Social Economic Deprivation Index(SEDI) to indicate the potential negative impact on suicide rates caused by environmental factors and social economic factors.
### Climate Factors includes mean temperature, visibility, precipitation and so on. The code to collect climate data can be found here:  [Climate Factors](https://github.com/Wenhuan2516/Climate-Data-Scraping)
### Social economic factors includes poverty, unemployment, disabled, and so on. The code to collect social economic data from Census API can be found here: [Social Economic Factors](https://github.com/Wenhuan2516/Census-API-data-collection)
### Since only about 1000 counties have the suicide death records from CDC, I imputed the suicide rates by assigning the average suicide rates from the abjecent counties. The code to impute suicide rates can be found here: [Suicide Rate Imputation](https://github.com/Wenhuan2516/SuicideRatesDataImputation)
### The form to compare coefficients and Root Mean Square Error were here: [Coefficients and RMSE comparison](https://docs.google.com/spreadsheets/d/1bYvy_ZorYkxhBcsd2F1QwCtBal5hvAIoIU-v7us_fYo/edit#gid=565528874)

### The interactive visualized results were displayed by a Dash App. The code can be found here: [MEDI and SEDI Dash Application](https://github.com/Wenhuan2516/MEDI-and-SEDI-Dash-Application)
### This is the poster
<img src="https://github.com/Wenhuan2516/MEDI-and-SEDI-Dash-Application/blob/main/medi.png" alt="image" title="medi">
### Here I am using results from 2016 to show you how MEDI and SEDI were developed.


