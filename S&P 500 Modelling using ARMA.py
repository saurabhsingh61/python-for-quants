#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[2]:


raw_data_csv = pd.read_csv("C:/Users/saura/Downloads/Jupyter/Index2018.xls")

df = raw_data_csv.copy()

df.head()


# In[3]:


df.describe()


# In[4]:


df.isna().sum()


# # Converting dates using to_datetime
# 
# 

# In[5]:


df.date = pd.to_datetime(df.date, dayfirst = True)
df.head()


# In[6]:


df.set_index('date', inplace = True)
df.head()


# # Converting to daily frequency and filling the missing values using the last filled entries.

# In[7]:


df = df.asfreq('d')
df.head()


# In[8]:


df = df.fillna(method = 'ffill')
df.isna().sum()


# # Splitting the stock indices into individual indicies

# In[9]:


df['spx_price'] = df.spx
df.head()


# In[10]:


df['ftse_price'] = df.ftse
df.head()


# In[11]:


df['dax_price'] = df.dax
df.head()


# In[12]:


df['nik_price'] = df.nikkei
df.head()


# # Adding returns columns to each of the 4 indicies dataframe

# In[13]:


df['spx_returns'] = df.spx.pct_change(1).mul(100)
df['ftse_returns'] = df.dax.pct_change(1).mul(100)
df['dax_returns'] = df.ftse.pct_change(1).mul(100)
df['nik_returns'] = df.nikkei.pct_change(1).mul(100)
df = df.iloc[1:]
df.head()


# In[14]:


df_sap = df[['spx_price', 'spx_returns']]
df_dax = df[['dax_price', 'dax_returns']]
df_fts = df[['ftse_price', 'ftse_returns']]
df_nik = df[['nik_price', 'nik_returns']]


# In[15]:


df_sap.head()


# # Importing Stats packages

# In[16]:


import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.arima_model import ARMA
from scipy.stats.distributions import chi2
from IPython.display import display, HTML
css = """
.output {
    flex-direction: row;
}
"""

HTML('<style>{}</style>'.format(css))
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    


# # The ACF and PACF plots

# In[17]:


sgt.plot_acf(df_sap.spx_price, zero = False, lags = 40)
plt.title('ACF for S&P 500 daily prices')
sgt.plot_acf(df_sap.spx_returns, zero = False, lags = 40)
plt.title('ACF for S&P 500 daily returns')
plt.show()


# In[18]:


model_ar_1 = ARMA(df_sap.spx_returns, order = (1, 0))
results_ar_1 = model_ar_1.fit()
results_ar_1.summary()


# In[19]:


model_ar_2 = ARMA(df_sap.spx_returns, order = (2,0))
results_ar_2 = model_ar_2.fit()
results_ar_2.summary()


# In[23]:


model_ar_3 = ARMA(df_sap.spx_returns, order = (3,0))
results_ar_3 = model_ar_3.fit()
results_ar_3.summary()


# # Selecting the values of p and q using LLR and AIC Values.

# In[31]:


model_ar_5_ma_1 = ARMA(df_sap.spx_returns, order = (5,1))
results_ar_5_ma_1 = model_ar_5_ma_1.fit()
print(results_ar_5_ma_1.summary())
df_sap['res_ar_5_ma_1'] = results_ar_5_ma_1.resid[1:]
df_sap.res_ar_5_ma_1.plot(figsize = (20,5))
plt.title('Residuals from ARMA(5,1)')
plt.show()

# model_ar_1_ma_5 = ARMA(df_sap.spx_returns, order = (1,5))
# results_ar_1_ma_5 = model_ar_1_ma_5.fit()
# print(results_ar_1_ma_5.summary())
# df_sap['res_ar_1_ma_5'] = results_ar_1_ma_5.resid[1:]
# df_sap.res_ar_1_ma_5.plot(figsize = (20,5))
# plt.title('Residuals from ARMA(1,5)')
# plt.show()

model_ar_5_ma_5 = ARMA(df_sap.spx_returns, order = (5,5))
results_ar_5_ma_5 = model_ar_5_ma_5.fit()
print(results_ar_5_ma_5.summary())
df_sap['res_ar_5_ma_5'] = results_ar_5_ma_5.resid[1:]
df_sap.res_ar_5_ma_5.plot(figsize = (20,5))
plt.title('Residuals from ARMA(5,5)')
plt.show()

model_ar_3_ma_2 = ARMA(df_sap.spx_returns, order = (3,2))
results_ar_3_ma_2 = model_ar_3_ma_2.fit()
print(results_ar_3_ma_2.summary())
df_sap['res_ar_3_ma_2'] = results_ar_3_ma_2.resid[1:]
df_sap.res_ar_3_ma_2.plot(figsize = (20,5))
plt.title('Residuals from ARMA(3,2)')
plt.show()


# llc_mat = np.empty((5, 5))
# aic_mat = np.empty((5, 5))
# for i in range(4):
#     for j in range(3,5):
#         llc_mat[i][j] = ARMA(df_sap.spx_returns, order = (i,j)).fit().llf
#         print('LLR (', i,  ',' ,j, ')', llc_mat[i][j])
#         print(ARMA(df_sap.spx_returns, order = (i,j)).fit().summary())
#         #aic_mat[i][j] = ARMA(df_sap.spx_returns, order = (i,j)).fit().aic
#         #print('AIC (', i, ',' ,j, ')', aic_mat[i][j])


# In[32]:


print('\n ARMA(3,2)\t LL = ',results_ar_3_ma_2.llf, '\t AIC = ', results_ar_3_ma_2.aic)
print('\n ARMA(5,5)\t LL = ',results_ar_5_ma_5.llf, '\t AIC = ', results_ar_5_ma_5.aic)
# print('\n ARMA(1,5)\t LL = ',results_ar_1_ma_5.llf, '\t AIC = ', results_ar_1_ma_5.aic)
print('\n ARMA(5,1)\t LL = ',results_ar_5_ma_1.llf, '\t AIC = ', results_ar_5_ma_1.aic)


# In[33]:


model_ar_5_ma_1 = ARMA(df_sap.spx_returns, order = (5,1))
results_ar_5_ma_1 = model_ar_5_ma_1.fit()
print(results_ar_5_ma_1.summary())
df_sap['res_ar_5_ma_1'] = results_ar_5_ma_1.resid[1:]
df_sap.res_ar_5_ma_1.plot(figsize = (20,5))
plt.title('Residuals from ARMA(5,1)')
plt.show()


# In[38]:


sgt.plot_acf(df_sap.res_ar_5_ma_1[1:], zero = False, lags = 40)
plt.title('ACF for residuals in ARMA(5,1)')
plt.show()
sgt.plot_acf(df_sap.res_ar_5_ma_5[1:], zero = False, lags = 40)
plt.title('ACF for residuals in ARMA(5,5)')
plt.show()


# # Final Model: ARMA(5,1)

# In[ ]:




