import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import *
import csv
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.compat import lzip


nRowsRead = None
df1 = pd.read_csv('C:/Users/saura/Downloads/data.csv', delimiter=',', nrows = nRowsRead)
df2 =pd.read_csv('C:/Users/saura/Downloads/data.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'Life Expectancy Data.csv'
df2.dataframeName = 'All years data'
nRow, nCol = df1.shape
#print(f'There are {nRow} rows and {nCol} columns')
#print(df1.Country.duplicated())
df = df1.drop_duplicates(subset = ['Country']) #Removing categorical entry

# Conversion of the categorical varibale Status into numerical value by assigning
# values to developing/developed statuses
lst_status = []
for i in df.Status:
    if(i == 'Developing'):
        lst_status.append(0.5)
    elif (i == 'Developed'):
        lst_status.append(1.0)
df.drop(columns=['Status'])
df['Status_metric'] = lst_status
df.rename({'Country':'C', 'Year':'Y', 'Status':'S', 'Life expectancy ':'D_le', 'Adult Mortality':'D_am',
       'infant deaths':'D_infd', 'Alcohol':'S_alc', 'percentage expenditure':'E_pe', 'Hepatitis B':'D_hb',
       'Measles ':'D_mea', ' BMI ':'D_bmi', 'under-five deaths ':'D_u5d', 'Polio':'D_pol', 'Total expenditure':'E_te',
       'Diphtheria ':'D_dip', ' HIV/AIDS':'D_hiv', 'GDP':'E_gdp', 'Population':'E_pop',
       ' thinness  1-19 years':'D_t1-19', ' thinness 5-9 years':'D_t5-9',
       'Income composition of resources':'E_icr', 'Schooling':'S_sch', 'Status_metric':'S_D/D'}, axis = 1, inplace = True)
print(df.columns)

#cleaning the data by removing rows with important missing values,
# we can't replace a missing GDP or similar parameters by the average of other countries
df.index = range(len(df.index))
df.drop(columns = ['S_alc', 'E_te'], axis = 1, inplace = True)
df.dropna(subset = ['E_gdp', 'E_pop', 'D_t5-9', 'D_hb', 'E_icr', 'D_le', 'D_am'], inplace = True)
df.index = range(len(df.index)) # reindexing the rows


# Separating the data into columns which signify health indicators and economic indicators
X_eco = df.drop(columns = ['C', 'Y', 'S', 'D_t1-19', 'D_t5-9', 'D_le',
                           'D_am', 'D_infd', 'E_pe', 'D_hb',
                           'D_mea', 'D_bmi', 'D_u5d', 'D_pol',
                           'D_dip', 'D_hiv', 'S_sch'], axis =1)
X_health = df.drop(columns = ['C', 'Y', 'S', 'D_t5-9', 'D_le',
                              'E_pe', 'D_infd',
                              'E_pop','E_icr', 'S_sch',
                              'S_D/D'], axis = 1)

print('-----------#-------------#----------We Begin From Here---------#-------------#--------------#')
X_health_const = add_constant(X_health)
X_eco_const = add_constant(X_eco)
Y = df['D_le']
dummy_data = 5*rand(100) + 50

# Function for plotting the Scatter matrices for health and economic indicators
def Scatter_Matrices(X_economic, X_health_indicators):
    pd.plotting.scatter_matrix(X_economic,
                               figsize=[10, 10],
                               diagonal='kde')
    plt.show()
    pd.plotting.scatter_matrix(X_health_indicators,
                               figsize=[10, 10],
                               diagonal='kde')
    plt.show()

# Function for calculating the correlation matrices for health and economic indicators
def Corr_matrices(X_economic, X_health_indicators):
    corrmat_economic = X_economic.corr()
    corrmat_health_indicators = X_health_indicators.corr()
    # print(corrmat_economic)
    # print(corrmat_health_indicators)
    return(corrmat_economic, corrmat_health_indicators)
    #print(corrmat_health.shape)

# Heat map for the corresponding correlation matrices
def Heat_maps(X_economic, X_health_indicators):
    corrmat_eco, corrmat_health = Corr_matrices(X_eco, X_health)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corrmat_eco, vmax=1., ax=ax, square=False).xaxis.tick_top()
    plt.show()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corrmat_health, vmax=1., ax=ax, square=False).xaxis.tick_bottom()
    plt.show()
# Valuation Inflation Factor for all the independent variables for checking multicollinearity
def VIF(X_economic_const, X_health_indicator_const):
    vif_eco_result = pd.Series([variance_inflation_factor(X_economic_const.values, i)
                                for i in range(X_economic_const.shape[1])],
                               index=X_economic_const.columns)
    vif_health_result = pd.Series([variance_inflation_factor(X_health_indicator_const.values, i)
                                   for i in range(X_health_indicator_const.shape[1])],
                                  index=X_health_indicator_const.columns)
    print(vif_eco_result)
    print(vif_health_result)
# Multiple Linear Regression Model to obtain the fitted Y-values, coefficients and intercept
def regression(X_economic, y):
    regr = linear_model.LinearRegression()
    regr.fit(X_economic, y)
    print("Intercept : \n", regr.intercept_ )
    print(X_economic.columns)
    print("Coefficients : \n", regr.coef_)
    #print(regr.summary)
    
# Residuals and Breusch-Pagan test for checking Heteroskedasticity and testing the Null-Hypothesis that there is no Heterscedasticity
def regression_model(X_dep, Y_indep):
    labels = ['LM Statistic', 'LM-test p-value', 'F-Staistic', 'F-test p-value']
    model = sm.OLS(Y_indep, X_dep).fit()
    print(model.summary())
    Y_model_fitted = model.fittedvalues
    model_norm_residuals = model.get_influence().resid_studentized_internal
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
    data_checked = model.resid
    print('Breusch-Pagan Test:')
    bp_test = sms.het_breuschpagan(data_checked, model.model.exog)
    print(dict(zip(labels, bp_test)))
    if(bp_test[-1]>=0.05):
        print("With 95% confidence, cannot reject the null-hypothesis that there is no heteroscedasticity")
    elif(bp_test[-1]<0.05):
        print("With 95% confidence, we can reject the null-hypothesis that there is no heteroscedasticity")
    #print(F_test_p_value)
    return (Y_model_fitted, model_norm_residuals, model_norm_residuals_abs_sqrt, data_checked)

Y_fitted, norm_resids, norm_resids_abs_sqrt, data_checked = regression_model(X_health, Y)


# QQplot for checking heteroscedasticity
def QQplots(Y_fit, normal_resids_abs_sqrt):
    plot_lm_3 = plt.figure()
    plt.scatter(Y_fit, normal_resids_abs_sqrt, alpha=0.5)
    sns.regplot(Y_fit, normal_resids_abs_sqrt,
                scatter=False,
                ci=False,
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
    plot_lm_3.axes[0].set_title('Scale-location')
    plot_lm_3.axes[0].set_xlabel('Fitted Values')
    plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized-Residuals|}$')
    plt.show()




print("Here is the complete data with no missing/categorical value")
print(df)
print('-----------#-------------#-------------------#-------------#--------------#')
print("These are the economic indicators which affect Life Expectancy")
print(X_eco)
print('-----------#-------------#-------------------#-------------#--------------#')
print("These are the health indicators which affect Life Expectancy")
print(X_health)
print('-----------#-------------#-------------------#-------------#--------------#')
my_abbrev = {'Country':'C', 'Year':'Y', 'Status':'S', 'Life expectancy ':'D_le', 'Adult Mortality':'D_am',
       'Infant deaths':'D_infd', 'Alcohol':'S_alc', 'Percentage expenditure':'E_pe', 'Hepatitis B':'D_hb',
       'Measles ':'D_mea', ' BMI ':'D_bmi', 'under-five deaths ':'D_u5d', 'Polio':'D_pol', 'Total expenditure':'E_te',
       'Diphtheria ':'D_dip', ' HIV/AIDS':'D_hiv', 'GDP':'E_gdp', 'Population':'E_pop',
       ' Thinness  1-19 years':'D_t1-19', ' Thinness 5-9 years':'D_t5-9',
       'Income composition of resources':'E_icr', 'Schooling':'S_sch', 'Status_metric':'S_D/D'}
print('-----------#-------------#--------Abbreviations-----------#-------------#--------------#')
print('   ')
for key in my_abbrev.keys():
    print(my_abbrev[key], end = ' ')
    print("                ------------->", end = ' ')
    print(key)
Scatter_Matrices(X_eco, X_health)
Corr_matrices(X_eco, X_health)
Heat_maps(X_eco, X_health)
VIF(X_eco_const, X_health_const)
regression(X_eco, Y)
regression(X_health, Y)
model = sm.OLS(Y, X_health).fit()
regression_model(X_health, Y)
model = sm.OLS(Y, X_eco).fit()
regression_model(X_eco, Y)
QQplots(Y_fitted, norm_resids_abs_sqrt)
