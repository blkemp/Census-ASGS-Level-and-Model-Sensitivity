# Census-ASGS-Level-and-Model-Sensitivity
An analysis of machine learning sensitivity to different levels of regional granularity in training.

The goal of this project is to compare the efficacy of machine learning models based on different levels of geographic aggregation, based on the [Australian Statistical Geography Standard](https://www.abs.gov.au/websitedbs/D3310114.nsf/home/Australian+Statistical+Geography+Standard+(ASGS)) framework. the key questions are:   

- Can we find a model for predicting with an R2 value > 0.7? (Based on [prior work](https://github.com/blkemp/ABS-Region-Data/tree/BKSubmission), I suspect the answer is yes).
- Using this model, what is the impact on accuracy for feeding in data that is consolidated at a different level (e.g. neighborhood vs city vs county)?
- How do models trained at differing levels of granularity compare in both baseline accuracy and generalisability? I.e. Are models trained with the most fine grained data more accurate, or are they prone to overfitting?

As a starting point I will be utilising the [Australian Bureau of Statistics 2016 Census Datapacks](https://datapacks.censusdata.abs.gov.au/datapacks/) and attempting to predict "working from home" behaviours by region. Why this particular response vector? a) It just seems interesting and b) I suspect that demographic information available within the census itself (gender, age, profession and industry) will all be strongly related to both individuals' propensity to undertake working from home and their ability to do so with support from employers.

# Summary of findings
By applying machine learning techniques on a custom list of identified tables within the dataset, I find a fairly robust Random Forest model (R2 score of 0.7) to predict the "Worked from Home participation rate" in a given population at the SA3 level, which is the most aggregated of the SA levels used in this project. Models trained at the more granular SA2 and SA1 levels performed increasingly poorly, with r2 scores of 0.46 and 0.38 respectively. By my reckoning, this decrease in performance is attributable to two factors:   
- Increasing variance in the predicted feature, an effect of the law of large numbers where smaller populations will less robustly conform to the overall population average. This is visible in the statistical variance at an SA3 level being ~10% of that seen at the SA2 level.
- [The intentional addition of "noise" by the ABS](https://www.abs.gov.au/ausstats/abs@.nsf/Latestproducts/1160.0Main%20Features6Aug%202017?opendocument&tabname=Summary&prodno=1160.0&issue=Aug%202017&num=&view=) to its datasets in an effort to protect the privacy of citizens and sensitive information.

It is not transparent to me how much power each of the two have, but both make sense.

In terms of analysing the data itself and what the model can tell us about relationships between working from home and other demographic information, the key finding is that age and occupation play a sizable role as a predictor of the propensity to work from home. Increasing the proportion of the population who are managers over the age of 55 has notable impacts on the predicted "worked from home participation rate", and males are more likely to do so than females in this age group. Age >55 as a key feature within the dataset was visible in all the models generated, regardless of the SA level. Whether this is an effect of preferences, accessibility or other factors is not yet determined, but it is interesting nonetheless.

A more accessible write-up can be found in my Medium blog post published [here](linktocome).

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
* importlib (optional - used in development when refining custom .py functions)

# Files
* Anything under the /Data/ directory - ABS Census DataPack files by Statistical Area, with an additional custom csv file within the "Metadata" subdirectory compiled as part of analysis to assit with data pipelining.
* Anything under the /Saved data/ directory - pickled models and saved .csv pandas tables generated in progressing through analysis. Note due to the size of the SA1 tables, these have not been uploaded to github, although the model itself has.
* Census ASGS Level and Model Sensitivity.ipynb - the main Jupyter notebook used in analysing the data. 
* Census data navigation workings.ipynb - a preliminary Jupyter notebook used in building the ETL pipeline for use in the analysis.
* au_census_analysis_functions.py - custom python file containing ETL pipeline, model training and model assessment functions used in the analysis.

# Copyright
Note all data sourced from the ABS is subject to copyright under Creative Commons Attribution 4.0 International, as outlined [here](https://www.abs.gov.au/copyright).
