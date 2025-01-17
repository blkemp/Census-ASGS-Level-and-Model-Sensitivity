# Census-ASGS-Level-and-Model-Sensitivity
An analysis of machine learning sensitivity to different levels of regional granularity in training.

The goal of this project is to compare the efficacy of machine learning models based on different levels of geographic aggregation, based on the [Australian Statistical Geography Standard](https://www.abs.gov.au/websitedbs/D3310114.nsf/home/Australian+Statistical+Geography+Standard+(ASGS)) framework. the key questions are:   

- Can we find a model for predicting with an R2 value > 0.7? (Based on [prior work](https://github.com/blkemp/ABS-Region-Data/tree/BKSubmission), I suspect the answer is yes).
- Using this model, what is the impact on accuracy for feeding in data that is consolidated at a different level (e.g. neighborhood vs city vs county)?
- How do models trained at differing levels of granularity compare in both baseline accuracy and generalisability? I.e. Are models trained with the most fine grained data more accurate, or are they prone to overfitting?

As a starting point I will be utilising the [Australian Bureau of Statistics 2016 Census Datapacks](https://datapacks.censusdata.abs.gov.au/datapacks/) and attempting to predict "working from home" behaviours by region. Why this particular response vector? a) It just seems interesting and b) I suspect that demographic information available within the census itself (gender, age, profession and industry) will all be strongly related to both individuals' propensity to undertake working from home and their ability to do so with support from employers.

# Metric for evaluation
It may be expected that because we are comparing model efficacy it would be worthwhile exploring RMSE as noted [here](https://medium.com/usf-msds/choosing-the-right-metric-for-machine-learning-models-part-1-a99d7d7414e4). Given the increasing variance in the predicted outcome at different SA levels, by failing to take the population variance into account RMSE will tend to inflate the error rate of the models trained on more disaggregated SA levels. Accordingly, I have been focused on R2 score as a metric (due to my desire to explore the explanatory power of models). 

I considered using Adjusted R squared, but the high dimensionality simply made the results irrelevant (Adjusted $R^2$ is supposed to always be lower than $R^2$, and this was not the case where n samples < k parameters as often occurred at the SA3 level). Additionally, since the number of features is the same between models, the only driver of differences between Adjusted $R^2$ and regular $R^2$ is the sample size. Given the entire point of this analysis is to determine the power of having more training samples, using a metric which seeks to mitigate this as a factor seemed disingenuous.

# Summary of findings
## Comparisons of model efficacy
By applying machine learning techniques on a custom list of identified tables within the dataset, I find a fairly robust Random Forest model (R2 score of 0.7) to predict the "Worked from Home participation rate" in a given population at the SA3 level, which is the most aggregated of the SA levels used in this project. Models trained at the more granular SA2 and SA1 levels varied in performance, with SA2 trained models typically performing better than SA3 models based on r2 scores, while SA1 data performs signficantly worse than models trained at either of the other area levels. By my reckoning, this performance is attributable to two factors:   
- SA2 level data has significantly more points to train on, which allows a better tuning of features to generate an effective model.
- [The intentional addition of "noise" by the ABS](https://www.abs.gov.au/ausstats/abs@.nsf/Latestproducts/1160.0Main%20Features6Aug%202017?opendocument&tabname=Summary&prodno=1160.0&issue=Aug%202017&num=&view=) to its datasets in an effort to protect the privacy of citizens and sensitive information becomes overwhelming at the SA1 level, making prediction extremely difficult.

It is not transparent to me how much power each of the two have, and it would be interesting to see how performance varies when using different supervised learning techniques such as boosting (based on current success with Random Forests) or SVM (based on the high dimensionality but "low" number of data points). Additionally, there seemed to be consistent improvement across models after initial training by reducing the number of input features from ~500 to only the top 100. I have not explored this improvement trend further but it could be worth investigation in the future.

While there was some variation in the top features (by feature importance) in models trained at different levels, in general the themes remained consistent. Further numerical analysis could be undertaken to quantify these variances.

Because there was some variation in model performance attributable to the random seed given in the test train split, I simulated a number of model train:test evaluations using iteratively generated seeds in order to remove doubts I had regarding potential noise in the models. Using n of 200 to compare SA2 and SA3 models (given a rough estimate of ~2% expected variance), I was able to complete a t-test to definitively demonstrate that models trained and evaluated on SA2 level data outperformed those given similar treatment using SA3 level aggregations (p<0.01). This method was also applied to the difference between SA2 and SA1 trained models (p<0.001).

## Insights from the data
The key finding is that age and occupation play a sizable role as a predictor of the propensity to work from home. Increasing the proportion of the population who are managers over the age of 55 has notable impacts on the predicted "worked from home participation rate", and males are more likely to do so than females in this age group. Age >55 as a key feature within the dataset was visible in all the models generated, regardless of the SA level. 

Interestingly, there is very little impact on the R^2 of models trained when *excluding* age as a categorical variable, with Occupation playing a clear proxy. The inclusion of sex has a minor positive effect, and splitting datasets between sexes had minimal impact of model performance. That said, the simulated change to the Worked From Home Participation rate of increasing the share of 65-74 year old managers was dramatically higher in the male population than in females (or from any other feature) which is an interesting outcome. Why is this demographic so likely to work from home? I suspect the simulated outcomes may be something to do with the share of managers that this population represents, but need to investigate further.

The effect highlighted by occupation seems to be attributable to the accessibility of the option to work from home (as compared to factory line workers for example). Whether the sex and age effects are an outcome of preferences, accessibility or other factors is not yet determined, but it is interesting nonetheless.

A more accessible write-up can be found in my Medium blog post published [here](https://medium.com/@brian.l.kemp/is-more-always-better-lessons-in-machine-learning-from-aggregated-census-data-83abf9644725).

# System Requirements:
All exploration done on this dataset is currently being undertaking using Jupyter Notebooks using Python 3.7 with the below modules (list will surely expand over time):
* NumPy
* Pandas
* MatPlotLib
* SKLearn
* os
* TextWrap
* pickle
* re
* operator
* scipy
* tqdm
* statsmodels
* importlib (optional - used in development when refining custom .py functions)

# Files
* Anything under the /Data/ directory - ABS Census DataPack files by Statistical Area, with an additional custom csv file within the "Metadata" subdirectory compiled as part of analysis to assit with data pipelining.
* Anything under the /Saved data/ directory - pickled models and saved .csv pandas tables generated in progressing through analysis. Note due to the size of the SA1 tables, these have not been uploaded to github, although the model itself has.
* Census ASGS Level and Model Sensitivity.ipynb - the main Jupyter notebook used in analysing the data. 
* Census data navigation workings.ipynb - a preliminary Jupyter notebook used in building the ETL pipeline for use in the analysis.
* au_census_analysis_functions.py - custom python file containing ETL pipeline, model training and model assessment functions used in the analysis.

# Copyright
Note all data sourced from the ABS is subject to copyright under Creative Commons Attribution 4.0 International, as outlined [here](https://www.abs.gov.au/copyright).