# Developing-MEDI-and-SEDI
## 12 environmental factors are combined to create a Multi-Dimensional Environmental Deprivation Index(MEDI)  and 15 social economic factors are combined to create a Social Economic Deprivation Index(SEDI) to indicate the potential negative impact on suicide rates caused by environmental factors and social economic factors.
### Climate Factors includes mean temperature, visibility, precipitation and so on. The code to collect climate data can be found here:  [Climate Factors](https://github.com/Wenhuan2516/Climate-Data-Scraping)
### Social economic factors includes poverty, unemployment, disabled, and so on. The code to collect social economic data from Census API can be found here: [Social Economic Factors](https://github.com/Wenhuan2516/Census-API-data-collection)
### Since only about 1000 counties have the suicide death records from CDC, I imputed the suicide rates by assigning the average suicide rates from the abjecent counties. The code to impute suicide rates can be found here: [Suicide Rate Imputation](https://github.com/Wenhuan2516/SuicideRatesDataImputation)
### The form to compare coefficients and Root Mean Square Error were here: [Coefficients and RMSE comparison](https://docs.google.com/spreadsheets/d/1bYvy_ZorYkxhBcsd2F1QwCtBal5hvAIoIU-v7us_fYo/edit#gid=565528874)

### The interactive visualized results were displayed by a Dash App. The code can be found here: [MEDI and SEDI Dash Application](https://github.com/Wenhuan2516/MEDI-and-SEDI-Dash-Application)
### This is the poster. The video I recorded to present my poster can be found here: ['Poster Presentation Video'](https://www.youtube.com/watch?v=OYm15fynyu8&feature=youtu.be)

<img src="https://github.com/Wenhuan2516/Developing-MEDI-and-SEDI/blob/main/poster-full.png" alt="image" title="medi">

### Here I am using results from 2016 to show you how MEDI and SEDI were developed.

#### Step 1: Using pairwise correlation to find the coeffecients between each climate factor or social economic factor and suicide rate in current year
<img src="https://github.com/Wenhuan2516/Developing-MEDI-and-SEDI/blob/main/pairwise.png" alt="image" title="medi">
<img src="https://github.com/Wenhuan2516/Developing-MEDI-and-SEDI/blob/main/pairwise-2.png" alt="image" title="medi">

#### Step 2: Use the last line of the pairwise correlation results to be the weight of each factor 
<img src="https://github.com/Wenhuan2516/Developing-MEDI-and-SEDI/blob/main/weight.png" alt="image" title="medi">

#### Step 3: Multiple the weight and the data in each category to get the a score; use percentile function to find a sub-index between 0 and 1
<img src="https://github.com/Wenhuan2516/Developing-MEDI-and-SEDI/blob/main/percentile.png" alt="image" title="medi">

#### Step 4: Use percentile function again to get a percentile for the sum of sub-indexes - This will be the final MEDI or SEDI
## MEDI
<img src="https://github.com/Wenhuan2516/Developing-MEDI-and-SEDI/blob/main/MEDI-2016.png" alt="image" title="medi">

## SEDI
<img src="https://github.com/Wenhuan2516/Developing-MEDI-and-SEDI/blob/main/SEDI-2016.png" alt="image" title="medi">

## SVI
<img src="https://github.com/Wenhuan2516/Developing-MEDI-and-SEDI/blob/main/SVI-2016.png" alt="image" title="medi">

## MEDI + SVI
<img src="https://github.com/Wenhuan2516/Developing-MEDI-and-SEDI/blob/main/updated-index.png" alt="image" title="medi">

## MEDI + SEDI
<img src="https://github.com/Wenhuan2516/Developing-MEDI-and-SEDI/blob/main/medi+sedi-2016.png" alt="image" title="medi">

#### Step 5: Use Root Mean Square Error to evaluation the result
<img src="https://github.com/Wenhuan2516/Developing-MEDI-and-SEDI/blob/main/evaluation.png" alt="image" title="medi">

## Some states are chosen to compare MEDI. The full comparison PPT can be found here: [MEDI Comparison in some states](https://docs.google.com/presentation/d/1fLIHfY_L9HCYICEGvxbB-1jxjH0KwQO_KRyAm56dc6w/edit?usp=sharing)

<img src="https://github.com/Wenhuan2516/Developing-MEDI-and-SEDI/blob/main/colorado.png" alt="image" title="medi">
<img src="https://github.com/Wenhuan2516/Developing-MEDI-and-SEDI/blob/main/comparison1.png" alt="image" title="medi">
<img src="https://github.com/Wenhuan2516/Developing-MEDI-and-SEDI/blob/main/comparison2.png" alt="image" title="medi">
<img src="https://github.com/Wenhuan2516/Developing-MEDI-and-SEDI/blob/main/comparison3.png" alt="image" title="medi">
<img src="https://github.com/Wenhuan2516/Developing-MEDI-and-SEDI/blob/main/comparison4.png" alt="image" title="medi">
<img src="https://github.com/Wenhuan2516/Developing-MEDI-and-SEDI/blob/main/comparison5.png" alt="image" title="medi">
