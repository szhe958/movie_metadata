import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
data = pd.read_csv('C:/Users/Administrator/Desktop/p1/movie_metadata.csv')

#overview
print(data.head(3))
data.info()
data.describe()
data.isnull().sum()
#EDA
data1 = data.copy()
# what kinds of genres there are and how many movies are fall into these categories
data1['genres']=data1['genres'].str.strip('[]').str.replace(' ','').str.replace("'",'')
data1['genres']=data1['genres'].str.split('|')

plt.subplots(figsize=(12,10))
list1=[]
for i in data1['genres']:
    list1.extend(i)
ax=pd.Series(list1).value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('summer_r',10))
for i, v in enumerate(pd.Series(list1).value_counts()[:10].sort_values(ascending=True).values):
  ax.text(.8, i, v,fontsize=12,color='white',weight='bold')
ax.patches[9].set_facecolor('r')
plt.title('Top Genres')
plt.show()
#Directors with highest movies
def xstr(s):
    if s is None:
        return 'A'
    return str(s)
data1['director_name']=data1['director_name'].apply(xstr)

plt.subplots(figsize=(12,10))
ax=data1[data1['director_name']!=''].director_name.value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.85,color='g')
for i, v in enumerate(data1[data1['director_name']!='A'].director_name.value_counts()[:10].sort_values(ascending=True).values):
    ax.text(.5, i, v,fontsize=12,color='white',weight='bold')
ax.patches[9].set_facecolor('r')
plt.title('Directors with highest movies')
plt.show()

# distribution of imdb_score
labels=data1["imdb_score"]
font = {'fontname':'Arial', 'size':'14'}
title_font = { 'weight' : 'bold','size':'16'}
plt.hist(labels, bins=20)
plt.title("Distribution of the IMDB_score")
plt.show()

#first correlation
fig, axs = plt.subplots(nrows=1, figsize=(13, 13))
sns.heatmap(data.corr(), annot=True, square=True, cmap='YlGnBu', linewidths=2, linecolor='black', annot_kws={'size':12})


#data cleaning
#column: color
data['color'] = data['color'].fillna('Color')

rcParams['figure.figsize'] = 10,5
sns.barplot(x = data1['color'].value_counts().index, y = data1['color'].value_counts().values)
plt.title('color')
plt.xlabel('Type of movies')
plt.ylabel('count')
plt.show()

encoded_content = pd.get_dummies(data['color'])
data= pd.concat([data,encoded_content], axis='columns')

#column: rating
data['content_rating'] = data['content_rating'].replace('Not Rated','Unrated')


rcParams['content_rating'] = 10,5
sns.barplot(x = data1['content_rating'].value_counts().index, y = data1['content_rating'].value_counts().values)
plt.title('content_rating')
plt.xlabel('content_rating')
plt.ylabel('count')
plt.show()

encoded_content = pd.get_dummies(data['content_rating'])
data= pd.concat([data,encoded_content], axis='columns')



#budget
import statistics
sd = statistics.stdev(data.budget)
mean = data.budget.mean()
max = data.budget.max()
min = data.budget.min()

data['VeryLowBud'] = data['budget'].map(lambda s: 1 if s < 10000000 else 0)
data['LowBud'] = data['budget'].map(lambda s: 1 if 10000000 <= s < mean else 0)
data['MedBud'] = data['budget'].map(lambda s: 1 if  mean <= s < mean+sd  else 0)
data['HighBud'] = data['budget'].map(lambda s: 1 if mean+sd <= s < 100000000 else 0)
data['VeryHighBud'] = data['budget'].map(lambda s: 1 if s >= 100000000 else 0)

# g = sns.factorplot(x="imdb_score",y="VeryLowBud",data=data,kind="bar",palette = "husl")
# g = g.set_ylabels("VeryLowBud")

#Genres
def Obtain_list_Occurences(columnName):
    # Obtaining a list of columnName
    list_details = list(map(str,(data[columnName])))
    listOcc = []
    for i in data[columnName]:
        split_genre = list(map(str, i.split('|')))
        for j in split_genre:
            if j not in listOcc:
                listOcc.append(j)
    return listOcc

genre = []
genre = Obtain_list_Occurences("genres")

for word in genre:
    data[word] = data['genres'].map(lambda s: 1 if word in str(s) else 0)
data.loc[:,'Action': 'Musical'].head(5)


#actor
data['actor_1_name'].fillna('unknown',inplace=True)
data['actor_2_name'].fillna('unknown',inplace=True)

data['actors_name'] = data[['actor_1_name', 'actor_2_name']].apply(lambda x: '|'.join(x), axis=1)

actor = []
actor = Obtain_list_Occurences("actors_name")

for word in actor:
    data[word] = data['actors_name'].map(lambda s: 1 if word in str(s) else 0)

#director

data['director_name'].fillna('unknown',inplace=True)
encoded_content = pd.get_dummies(data['director_name'])
data= pd.concat([data,encoded_content], axis='columns')


#imdb_score


data.loc[ data['imdb_score'] < 4, 'imdb_score'] = 0
data.loc[(data['imdb_score'] >=4) & (data['imdb_score'] < 6), 'imdb_score'] = 1
data.loc[(data['imdb_score'] >=6) & (data['imdb_score'] < 8), 'imdb_score'] = 2
data.loc[(data['imdb_score'] >=8) & (data['imdb_score'] < 10), 'imdb_score'] = 3


data.loc[:, ['imdb_score']].head()
data['imdb_score'].value_counts(sort = False)

data.drop(['color','director_name','num_critic_for_reviews','director_facebook_likes','actor_3_facebook_likes','actor_1_facebook_likes','gross',
          'genres','movie_title','language','director_name','actor_1_name','actor_2_name','actor_3_name','cast_total_facebook_likes','facenumber_in_poster',
          'plot_keywords','movie_imdb_link','country','genres','duration','num_user_for_reviews','actors_name','num_voted_users','country','content_rating','budget','actor_2_facebook_likes','aspect_ratio','movie_facebook_likes','title_year'], axis=1, inplace=True)

#model
data['is_train'] = np.random.uniform(0, 1, len(data)) <= .75
train, test = data[data['is_train']==True], data[data['is_train']==False]

train.drop(['is_train'], axis=1, inplace=True)
test.drop(['is_train'], axis=1, inplace=True)

train["imdb_score"] = train["imdb_score"].astype(int)

Y_train = train["imdb_score"]
X_train = train.drop(labels = ["imdb_score"],axis = 1)
Y_test = test["imdb_score"]
X_test = test.drop(labels = ["imdb_score"],axis = 1)


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import  cross_val_score

# Decision Tree
c_dec = cross_val_score(DecisionTreeClassifier(), X_train, Y_train).mean()
print("Decision Tree:", c_dec)

# Logistic Regression
c_lr = cross_val_score(LogisticRegression(), X_train, Y_train).mean()
print("Logistic Regression:", c_lr)

# Random Forest
c_rf = cross_val_score(RandomForestClassifier(), X_train, Y_train).mean()
print("Random Forest:", c_rf)

# SVC
c_s = cross_val_score(SVC(), X_train, Y_train).mean()
print("SVC:", c_s)


# GaussianNB
c_g = cross_val_score(GaussianNB(), X_train, Y_train).mean()
print("GaussianNB:", c_g)

#cross validation scores
cv_means = []
cv_means.append(c_dec.mean())
cv_means.append(c_lr.mean())
cv_means.append(c_rf.mean())
cv_means.append(c_s.mean())
cv_means.append(c_g.mean())

cv_std = []
cv_std.append(c_dec.std())
cv_std.append(c_lr.std())
cv_std.append(c_rf.std())
cv_std.append(c_s.std())
cv_std.append(c_g.std())

res1 = pd.DataFrame({"ACC":cv_means,"Std":cv_std,"Algorithm":["DecisionTree","logistic regression","Random Forest","SVC","GaussianNB"]})
res1["Type"]= "CrossValid"
g = sns.barplot("ACC","Algorithm",data = res1, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


