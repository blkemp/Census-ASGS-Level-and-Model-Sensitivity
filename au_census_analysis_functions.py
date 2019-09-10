# Import statements
# Declare Imports
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import operator
from textwrap import wrap

# Set a variable for current notebook's path for various loading/saving mechanisms
nb_path = os.getcwd()

'''Data import functions'''

def load_census_csv(table_list, statistical_area_code='SA3'):
    '''
    Navigates the file structure to import the relevant files for specified data tables at a defined statistical area level
    
    INPUTS
    table_list: LIST of STRING objects - the ABS Census Datapack table to draw information from (G01-G59)
    statistical_area_code: STRING - the ABS statistical area level of detail required (SA1-SA3)
    
    OUTPUTS
    A pandas dataframe
    '''
    statistical_area_code = statistical_area_code.upper()
    
    df_csv_load = pd.DataFrame()
    for index, table in enumerate(table_list):
        
        if index==0:
            df_csv_load = pd.read_csv('{}\Data\{}\AUST\\2016Census_{}_AUS_{}.csv'.format(nb_path,
                                                                                statistical_area_code,
                                                                                table,
                                                                                statistical_area_code
                                                                               ),
                                       engine='python')
        else:
            temp_df = pd.read_csv('{}\Data\{}\AUST\\2016Census_{}_AUS_{}.csv'.format(nb_path,
                                                                                statistical_area_code,
                                                                                table,
                                                                                statistical_area_code
                                                                               ),
                                       engine='python')
            merge_col = df_csv_load.columns[0]
            df_csv_load = pd.merge(df_csv_load, temp_df, on=merge_col)
    
    return df_csv_load



def refine_measure_name(table_namer, string_item, category_item, category_list):
    '''Simple function for generating measure names based on custom metadata information on ABS measures'''
    position_list = []
    for i, j in enumerate(category_item.split("|")):
        if j in category_list:
            position_list.append(i)
    return table_namer + '|' + '_'.join([string_item.split("|")[i] for i in position_list])


def load_table_refined(table_ref, category_list, statistical_area_code='SA3'):
    '''
    Function for loading ABS census data tables, and refining/aggregating by a set of defined categories
    (e.g. age, sex, occupation, English proficiency, etc.) where available.
    
    INPUTS
    table_ref: STRING - the ABS Census Datapack table to draw information from (G01-G59)
    category_list: LIST of STRING objects - Cetegorical informatio to slice/aggregate information from (e.g. Age)
    statistical_area_code: STRING - the ABS statistical area level of detail required (SA1-SA3)
    '''
    df_meta = pd.read_csv('{}\Data\Metadata\Metadata_2016_refined.csv'.format(os.getcwd()))
    
    # slice meta based on table
    meta_df_select = df_meta[df_meta['Profile table'].str[:(len(table_ref)+1)] == table_ref].copy()
    
    # for category in filter_cats, slice based on category >0
    for cat in category_list:
        # First, check if there *are* any instances of the given category
        try:
            if meta_df_select[cat].sum() > 0:
                # If so, apply the filter
                meta_df_select = meta_df_select[meta_df_select[cat]>0]
            else:
                pass # If not, don't apply (otherwise you will end up with no selections)
        except:
            pass
        
    # select rows with lowest value in "Number of Classes Excl Total" field
    min_fields = meta_df_select['Number of Classes Excl Total'].min()
    meta_df_select = meta_df_select[meta_df_select['Number of Classes Excl Total'] == min_fields]
    
    # Select the table file(s) to import
    import_table_list = meta_df_select['DataPack file'].unique()
    
    # Import the SA data tables
    df_data = load_census_csv(import_table_list, statistical_area_code.upper())
    
    # Select only columns included in the meta-sliced table above
    df_data.set_index(df_data.columns[0], inplace=True)
    refined_columns = meta_df_select.Short.tolist()
    df_data = df_data[refined_columns]
    
    # aggregate data by:
    # transposing the dataframe
    df_data_t = df_data.T.reset_index()
    df_data_t.rename(columns={ df_data_t.columns[0]: 'Short' }, inplace = True)
    # merging with the refined meta_df to give table name, "Measures" and "Categories" fields
    meta_merge_ref = meta_df_select[['Short','Table name','Measures','Categories']]
    df_data_t = df_data_t.merge(meta_merge_ref, on='Short')
    
    # from the "Categories" field, you should be able to split an individual entry by the "|" character
    # to give the index of the measure you are interested in grouping by
    # create a new column based on splitting the "Measure" field and selecting the value of this index/indices
    # Merge above with the table name to form "[Table_Name]|[groupby_value]" to have a good naming convention
    # eg "Method_of_Travel_to_Work_by_Sex|Three_methods_Females"
    df_data_t['Test_name'] = df_data_t.apply(lambda x: refine_measure_name(x['Table name'], 
                                                                           x['Measures'], 
                                                                           x['Categories'], 
                                                                           category_list), axis=1)
    
    # then groupby this new column 
    # then transpose again and either create the base data_df for future merges or merge with the already existing data_df
    df_data_t = df_data_t.drop(['Short','Table name','Measures','Categories'], axis=1)
    df_data_t = df_data_t.groupby(['Test_name']).sum()
    
    return df_data_t.T


def load_tables_specify_cats(table_list, category_list, statistical_area_code='SA3'):
    '''
    Function for loading ABS census data tables, and refining/aggregating by a set of defined categories
    (e.g. age, sex, occupation, English proficiency, etc.) where available.
    
    INPUTS
    table_list: LIST of STRING objects - list of the ABS Census Datapack tables to draw information from (G01-G59)
    category_list: LIST of STRING objects - Cetegorical information to slice/aggregate information from (e.g. Age)
    statistical_area_code: STRING - the ABS statistical area level of detail required (SA1-SA3)
    
    OUTPUTS
    A pandas dataframe
    '''
    for index, table in enumerate(table_list):
        if index==0:
            df = load_table_refined(table, category_list, statistical_area_code)
            df.reset_index(inplace=True)
        else:
            temp_df = load_table_refined(table, category_list, statistical_area_code)
            temp_df.reset_index(inplace=True)
            merge_col = df.columns[0]
            df = pd.merge(df, temp_df, on=merge_col)
    
    df.set_index(df.columns[0], inplace=True)
    
    return df


def sort_series_abs(S):
    '''Takes a pandas Series object and returns the series sorted by absolute value'''
    temp_df = pd.DataFrame(S)
    temp_df['abs'] = temp_df.iloc[:,0].abs()
    temp_df.sort_values('abs', ascending = False, inplace = True)
    return temp_df.iloc[:,0]


'''Plotting functions'''

def feature_plot_h(model, X_train, n_features):
    '''
    Takes a trained model and outputs a horizontal bar chart showing the "importance" of the
    most impactful n features.
    
    INPUTS
    model = Trained model in sklearn with  variable ".feature_importances_". Trained supervised learning model.
    X_train = Pandas Dataframe object. Feature set the training was completed using.
    n_features = Int. Top n features you would like to plot.
    '''
    importances = model.feature_importances_
    # Identify the n most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:n_features]]
    values = importances[indices][:n_features]
    
    columns = [ '\n'.join(wrap(c, 30)).replace("_", " ") for c in columns ]
    
    # Create the plot
    fig = plt.figure(figsize = (9,n_features))
    plt.title("Normalized Weights for {} Most Predictive Features".format(n_features), fontsize = 16)
    plt.barh(np.arange(n_features), values, height = 0.4, align="center", color = '#00A000', 
          label = "Feature Weight")
    plt.barh(np.arange(n_features) - 0.3, np.cumsum(values), height = 0.2, align = "center", color = '#00A0A0', 
          label = "Cumulative Feature Weight")
    plt.yticks(np.arange(n_features), columns)
    plt.xlabel("Weight", fontsize = 12)
    
    plt.legend(loc = 'upper right')
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()  

    
def feature_impact_plot(model, X_train, n_features, y_label):
    '''
    Takes a trained model and training dataset and synthesises the impacts of the top n features
    to show their relationship to the response vector (i.e. how a change in the feature changes
    the prediction). Returns n plots showing the variance for min, max, median, 1Q and 3Q.
    
    INPUTS
    model = Trained model in sklearn with  variable ".feature_importances_". Trained supervised learning model.
    X_train = Pandas Dataframe object. Feature set the training was completed using.
    n_features = Int. Top n features you would like to plot.
    y_label = String. Description of response variable for axis labelling.
    '''
    # Display the n most important features
    indices = np.argsort(model.feature_importances_)[::-1]
    columns = X_train.columns.values[indices[:n_features]]
    
    sim_var = [[]]
    
    for col in columns:
        base_pred = model.predict(X_train)
        #add percentiles of base predictions to a df for use in reporting
        base_percentiles = [np.percentile(base_pred, pc) for pc in range(0,101,25)]

        # Create new predictions based on tweaking the parameter
        # copy X, resetting values to align to the base information through different iterations
        df_copy = X_train.copy()

        for val in np.arange(-X_train[col].std(), X_train[col].std(), X_train[col].std()/50):
            df_copy[col] = X_train[col] + val
            # Add new predictions based on changed database
            predictions = model.predict(df_copy)
            
            # Add percentiles of these predictions to a df for use in reporting
            percentiles = [np.percentile(predictions, pc) for pc in range(0,101,25)]
            
            # Add variances between percentiles of these predictions and the base prediction to a df for use in reporting
            percentiles = list(map(operator.sub, percentiles, base_percentiles))
            percentiles = list(map(operator.truediv, percentiles, base_percentiles))
            sim_var.append([val, col] + percentiles)

    # Create a dataframe based off the arrays created above
    df_predictions = pd.DataFrame(sim_var,columns = ['Value','Feature']+[0,25,50,75,100])
    
    # Create a subplot object based on the number of features
    num_cols = 2
    subplot_rows = int(n_features/num_cols) + int(n_features%num_cols)
    fig, axs = plt.subplots(nrows = subplot_rows, ncols = num_cols, sharey = True, figsize=(15,5*subplot_rows))

    nlines = 1

    # Plot the feature variance impacts
    for i in range(axs.shape[0]*axs.shape[1]):
        if i < len(columns):
            # Cycle through each plot object in the axs array and plot the appropriate lines
            ax_row = int(i/num_cols)
            ax_column = int(i%num_cols)
            
            axs[ax_row, ax_column].plot(df_predictions[df_predictions['Feature'] == columns[i]]['Value'],
                     df_predictions[df_predictions['Feature'] == columns[i]][50])
            
            axs[ax_row, ax_column].set_title("\n".join(wrap(columns[i], int(100/num_cols))))
            
            # Create spacing between charts if chart titles happen to be really long.
            nlines = max(nlines, axs[ax_row, ax_column].get_title().count('\n'))

            axs[ax_row, ax_column].set_xlabel('Simulated +/- change to feature'.format(y_label))
            
            # Format the y-axis as %
            if ax_column == 0:
                vals = axs[ax_row, ax_column].get_yticks()
                axs[ax_row, ax_column].set_yticklabels(['{:,.2%}'.format(x) for x in vals])
                axs[ax_row, ax_column].set_ylabel('% change to {}'.format(y_label))
        
        # If there is a "spare" plot, hide the axis so it simply shows ans an empty space
        else:
            axs[int(i/num_cols),int(i%num_cols)].axis('off')
    
    # Apply spacing between subplots in case of very big headers
    fig.subplots_adjust(hspace=0.5*nlines)
    
    # Return the plot
    plt.tight_layout()    
    plt.show()