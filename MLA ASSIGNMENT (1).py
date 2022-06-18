#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
df = pd.read_csv('C:/Users/HP/Downloads/NAICExpense.csv')
df.head()


# In[7]:


df.describe()


# In[10]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# In[14]:


print('abc')


# In[ ]:


##REGRESSION


# In[36]:


x = df['STAFFWAGE']
y = df['EXPENSES']
def estimate_coef(x,y):
    n = np.size(x)
    m_x = np.mean(x)
    m_y = np.mean(y)
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(y*y) - n*m_x*m_x
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    return (b_0, b_1)
def plot_regression_line(x, y, b):
    plt.scatter(x, y, color = "m",
               marker = "o", s = 30)
    y_pred = b[0] + b[1]*x
    plt.plot(x, y_pred, color = "g")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
        
def main():
    x = np.array([x])
    y = np.array([y])
    b = estimate_coef(x,y)
    print("Estimated coefficients:\nb_0 = {}            \nb_1 = {}".format(b[0], b[1]))
    plot_regression_line(x, y, b)
    if __name__ == "__main__":
        main()
estimate_coef(x,y)


# In[15]:


print('null')


# In[18]:


print('namaste')


# In[17]:



    print('hello')


# In[16]:


print('hi')


# In[14]:


pip install statsmodels


# In[25]:


import statsmodels.api as sm


# In[22]:


y = df['EXPENSES']
x = df['STAFFWAGES']


# In[35]:


x = df['STAFFWAGE']
y = df['EXPENSES']
sns.scatterplot(x=x['STAFFWAGE'], y=y)


# In[9]:


plt.scatter(list1, list2)
plt.plot(np.unique(list1), np.poly1d(np.polyfit(list1, list2, 1))
         (np.unique(list1)), color='red')


# In[20]:


list1 = df['STAFFWAGE']
list2 = df['EXPENSES']
corr, _ = pearsonr(list1, list2)
print('Pearsons correlation: %.3f' % corr)


# In[ ]:


##correlation scatter plot and calculations


# In[10]:


x = df['STAFFWAGE']
y = df['EXPENSES']
def estimate_coef(x,y):
    n = np.size(x)
    m_x = np.mean(x)
    m_y = np.mean(y)
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(y*y) - n*m_x*m_x
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    return (b_0, b_1)
def plot_regression_line(x, y, b):
    plt.scatter(x, y, color = "m",
               marker = "o", s = 30)
    y_pred = b[0] + b[1]*x
    plt.plot(x, y_pred, color = "g")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
        
def main():
    x = np.array([x])
    y = np.array([y])
    b = estimate_coef(x,y)
    print("Estimated coefficients:\nb_0 = {}            \nb_1 = {}".format(b[0], b[1]))
    plot_regression_line(x, y, b)
    if __name__ == "__main__":
        main()
estimate_coef(x,y)


# In[15]:


sb.regplot(x = "STAFFWAGE",
            y = "EXPENSES", 
            ci = None,
            data = df)


# In[ ]:


sb.regplot(x = "STAFFWAGE",
            y = "EXPENSES", 
            ci = None,
            data = df)


# In[1]:


import seaborn as sb


# In[ ]:


##regression scatterplot and calculations


# In[ ]:





# In[4]:


sb.regplot(x = "AGENTWAGE",
            y = "EXPENSES", 
            ci = None
            data = df)


# In[6]:


sb.lineplot(x = "AGENTWAGE",
            y = "EXPENSES", 
            ci = None,
            data = df)


# In[ ]:





# In[12]:


print("hellow")


# In[ ]:

print("hellow")
print("hellow world")


