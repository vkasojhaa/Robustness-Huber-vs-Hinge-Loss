
# coding: utf-8

# In[13]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec 
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


df = pd.read_csv('robust.csv')
df.fillna("", inplace=True)
df.rename( columns={'Unnamed: 4':'(Noise %)', 'Unnamed: 5':'', 'Unnamed: 7':'(Noise %)', 'Unnamed: 8':''}, inplace=True )

X =[0, 15, 30]
df


# In[5]:


plt.plot(X, list(df.iloc[5, 3:6]), 'o-', label='Hinge')
plt.plot(X, list(df.iloc[5, 6:9]), 'o-', label='Huber')

plt.title("Breast Cancer Wisconsin - Linear SVM")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.grid(True)
plt.legend()
plt.show()


# In[6]:


plt.plot(X, list(df.iloc[6, 3:6]), 'o-', label='Hinge')
plt.plot(X, list(df.iloc[6, 6:9]), 'o-', label='Huber')

plt.title("Breast Cancer Wisconsin - SGD")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.grid(True)
plt.legend()
plt.show()


# In[7]:


X = [0, 15, 30]

plt.plot(X, list(df.iloc[1, 3:6]), 'o-', label='Hinge')
plt.plot(X, list(df.iloc[1, 6:9]), 'o-', label='Huber')

plt.title("Pima Indian Dataset - Linear SVM")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.grid(True)
plt.legend()
plt.show()


# In[8]:


plt.plot(X, list(df.iloc[2, 3:6]), 'o-', label='Hinge')
plt.plot(X, list(df.iloc[2, 6:9]), 'o-', label='Huber')

plt.title("Pima Indian Dataset - SGD")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.grid(True)
plt.legend()
plt.show()


# In[9]:


plt.plot(X, list(df.iloc[7, 3:6]), 'o-', label='Hinge')
plt.plot(X, list(df.iloc[7, 6:9]), 'o-', label='Huber')

plt.title("ILPD - Linear SVM")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.grid(True)
plt.legend()
plt.show()


# In[10]:


plt.plot(X, list(df.iloc[8, 3:6]), 'o-', label='Hinge')
plt.plot(X, list(df.iloc[8, 6:9]), 'o-', label='Huber')

plt.title("ILPD - SGD")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.grid(True)
plt.legend()
plt.show()


# In[11]:


plt.plot(X, list(df.iloc[3, 3:6]), 'o-', label='Hinge')
plt.plot(X, list(df.iloc[3, 6:9]), 'o-', label='Huber')

plt.title("Abalone - Linear SVM")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.grid(True)
plt.legend()
plt.show()


# In[12]:


plt.plot(X, list(df.iloc[4, 3:6]), 'o-', label='Hinge')
plt.plot(X, list(df.iloc[4, 6:9]), 'o-', label='Huber')

plt.title("Abalone - SGD")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.grid(True)
plt.legend()
plt.show()

