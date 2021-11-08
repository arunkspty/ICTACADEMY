#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[18]:


data=pd.read_csv('iris.csv')


# In[19]:


data.mean(axis =0)


# In[20]:


data.mean(axis=None, skipna=None, level=None, numeric_only=None)


# In[21]:


data.head()


# In[23]:


data.columns


# In[65]:


plt.figure()
plt.hist(data['SL'],color='yellow',rwidth=0.9,density=True)
plt.title('Distribution Vs Frequency of SL')
plt.xlabel('Distribution')
plt.ylabel('Frequency')


# In[64]:


plt.figure()
plt.hist(data['SL'],color='yellow',rwidth=0.9,density=True,cumulative=True)
plt.title('Distribution Vs Cumulative Frequency of SL')
plt.xlabel('Distribution')
plt.ylabel('Frequency')


# In[63]:


plt.hist(data['PL'],color='yellow',rwidth=0.9,density=True)
plt.title('Distribution Vs Frequency of PL')
plt.xlabel('Distribution')
plt.ylabel('Frequency')


# In[51]:


data[['SL','PL']].plot.hist()


# In[52]:


data[['SL','PL']].plot.kde()


# In[61]:


plt.figure()
plt.scatter(data['SL'],data['PL'],s=10)
plt.title('Plot of SL Vs PL')
plt.xlabel('SL')
plt.ylabel('PL')


# In[68]:


plt.figure()
plt.scatter(data['SW'],data['PW'],s=10,c='red')
plt.title('Plot of SW Vs PW')
plt.xlabel('SW')
plt.ylabel('PW')


# In[81]:


plt.figure(figsize=(8,6))

plt.subplot(2,1,1)
plt.scatter(data['SL'],data['PL'],s=10)
plt.title('Plot of SL Vs PL')
plt.xlabel('SL')
plt.ylabel('PL')

plt.subplot(2,1,2)
plt.scatter(data['SW'],data['PW'],s=10,c='red')
plt.title('Plot of SW Vs PW')
plt.xlabel('SW')
plt.ylabel('PW')

plt.tight_layout()


# In[83]:


sns.pairplot(data)


# In[88]:


slice_plot=data[['SL','PL']]
sns.pairplot(slice_plot)


# In[93]:


sns.heatmap(slice_plot.corr(),annot=True)


# In[112]:


plt.boxplot(data['SL'])
plt.title('SL')


# In[113]:


plt.boxplot(data['PL'])
plt.title('PL')


# In[121]:


sns.boxplot(x='SL',y='PL',data=data)


# In[130]:


plt.figure(figsize=(8,6))

plt.subplot(2,1,1)
sns.stripplot(data['SL'])

plt.subplot(2,1,2)
sns.stripplot(data['PL'])


# In[140]:


sns.stripplot(x='SL',y='PL',data=data)
sns.swarmplot(x='SL',y='PL',data=data)


# In[139]:


sns.violinplot(x='SL',y='PL',data=data)


# In[144]:


sns.countplot(data['SL'])
plt.grid()

