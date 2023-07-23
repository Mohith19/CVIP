#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[10]:


data=pd.read_csv('IMDB-Movie-Data.csv')


# In[4]:


data.head()


# In[5]:


data.head(10)


# In[6]:


data.tail()


# In[7]:


data.tail(10)


# In[8]:


data.shape


# In[9]:


print("Number of Rows",data.shape[0])
print("Number of Columns",data.shape[1])


# In[10]:


data.info()


# In[11]:


print("Any missing values?",data.isnull().values.any())


# In[12]:


data.isnull()


# In[13]:


data.isnull().sum()


# In[14]:


sns.heatmap(data.isnull())


# In[15]:


per_missing = data.isnull().sum() * 100 / len(data)
per_missing


# In[19]:


data.dropna(axis=0)


# In[20]:


dup_data=data.duplicated().any()


# In[21]:


print("Are there any duplicate values?",dup_data)


# In[22]:


data=data.drop_duplicates()
data


# In[23]:


data.describe()


# In[24]:


data.describe(include='all')


# In[25]:


data.columns


# In[26]:


data['Runtime (Minutes)']>=180


# In[27]:


data[data['Runtime (Minutes)']>=180]


# In[28]:


data[data['Runtime (Minutes)']>=180]['Title']


# In[29]:


data.columns


# In[31]:


data.groupby('Year')['Votes'].mean()


# In[32]:


data.groupby('Year')['Votes'].mean().sort_values()


# In[33]:


data.groupby('Year')['Votes'].mean().sort_values(ascending=False)


# In[34]:


sns.barplot(x='Year',y="Votes",data=data)
plt.title("Votes By Year")


# In[35]:


sns.barplot(x='Year',y="Votes",data=data)
plt.title("Votes By Year")
plt.show()


# In[36]:


data.columns


# In[37]:


data.groupby('Year')['Revenue (Millions)'].mean().sort_values(ascending=False)


# In[38]:


sns.barplot(x='Year',y="Revenue (Millions)",data=data)
plt.title("Revenue By Year")
plt.show()


# In[39]:


data.columns


# In[40]:


data.groupby('Director')['Rating'].mean()


# In[41]:


data.groupby('Director')['Rating'].mean().sort_values(ascending=False)


# In[42]:


data.columns


# In[43]:


data.nlargest(10,'Runtime (Minutes)')


# In[44]:


data.nlargest(10,'Runtime (Minutes)')[['Title','Runtime (Minutes)']]


# In[46]:


data.nlargest(10,'Runtime (Minutes)')[['Title','Runtime (Minutes)']]\
.set_index('Title')


# In[47]:


top10_len=data.nlargest(10,'Runtime (Minutes)')[['Title','Runtime (Minutes)']]\
.set_index('Title')


# In[48]:


top10_len


# In[50]:


sns.barplot(x='Runtime (Minutes)',y=top10_len.index,data=top10_len)


# In[11]:


data.columns


# In[12]:


data['Year'].value_counts()


# In[15]:


sns.countplot(x='Year',data=data)
plt.title("Number Of Movies Per Year")
plt.show()


# In[16]:


data.columns


# In[19]:


data[data['Revenue (Millions)'].max()==data['Revenue (Millions)']]


# In[20]:


data.columns


# In[21]:


top10_len=data.nlargest(10,'Rating')[['Title','Rating','Director']]\
.set_index('Title')


# In[22]:


top10_len


# In[26]:


sns.barplot(x='Rating',y=top10_len.index,data=top10_len,hue='Director',dodge=False)
plt.legend(bbox_to_anchor=(1,1),loc=2)


# In[27]:


data.columns


# In[29]:


data.nlargest(10,'Revenue (Millions)')['Title']


# In[32]:


top_10=data.nlargest(10,'Revenue (Millions)')[['Title','Revenue (Millions)']].\
set_index('Title')


# In[33]:


top_10


# In[37]:


sns.barplot(x='Revenue (Millions)',y=top_10.index,data=top_10)
plt.title("Top 10 Highest Revenue Movies")
plt.show()


# In[38]:


data.columns


# In[39]:


data.groupby('Year')['Rating'].mean().sort_values(ascending=False)


# In[40]:


data.columns


# In[42]:


sns.scatterplot(x="Rating",y="Revenue (Millions)",data=data)


# In[43]:


data.columns


# In[44]:


def rating(rating):
    if rating>=7.0:
        return "Very Good"
    elif rating>=6.0:
        return "Good"
    else:
        return "Average"


# In[45]:


data['rating_cat']=data['Rating'].apply(rating)


# In[46]:


data.head()


# In[47]:


data.columns


# In[48]:


data['Genre'].dtype


# In[50]:


data[data['Genre'].str.contains('Action',case=False)]


# In[51]:


len(data[data['Genre'].str.contains('Action',case=False)])


# In[ ]:




