# Import libraries
#!pip install rake_nltk
from rake_nltk import Rake   # ensure this is installed

import nltk
nltk.download('all')
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings("ignore")


import os

from flask import Flask
from flask import request, url_for, redirect, render_template, jsonify


app = Flask(__name__)
df = pd.read_csv('https://query.data.world/s/uikepcpffyo2nhig52xxeevdialfl7')   # 250 rows × 38 columns
#df = pd.read_csv('IMDB_Top250Engmovies2_OMDB_Detailed.csv')   # same data 250 rows × 38 columns

# data overview
print('Rows x Columns : ', df.shape[0], 'x', df.shape[1])
print('Features: ', df.columns.tolist())
print('\nUnique values:')
print(df.nunique())
for col in df.columns:
    print(col, end=': ')
    print(df[col].unique())


# type of entries, how many missing values/null fields
df.info()
print('\nMissing values:  ', df.isnull().sum().values.sum())
df.isnull().sum()


# summary statistics for all numerical columns
df.describe().T

# keep only these 5 useful columns, 250 rows with no NaN field
df = df[['Title','Director','Actors','Plot','Genre']]


df.loc[(df.Genre == 'Drama')]


# top genres (from 110 unique genres)
df['Genre'].value_counts()

# 10 popular directors (from 155 unique directors)
df['Director'].value_counts()[0:10].plot(figsize=[8,5], fontsize=15, color='navy').invert_yaxis()

##Data Preprocessing##

# to remove punctuations from Plot
df['Plot'] = df['Plot'].str.replace('[^\w\s]','')

# # alternative way to remove punctuations, same result
# import string
# df['Plot'] = df['Plot'].str.replace('[{}]'.format(string.punctuation), '')


# to extract key words from Plot to a list
df['Key_words'] = ''   # initializing a new column
r = Rake()   # use Rake to discard stop words (based on english stopwords from NLTK)

for index, row in df.iterrows():
    r.extract_keywords_from_text(row['Plot'])   # to extract key words from Plot, default in lower case
    key_words_dict_scores = r.get_word_degrees()    # to get dictionary with key words and their scores
    row['Key_words'] = list(key_words_dict_scores.keys())   # to assign list of key words to new column



# to see last item in Plot
df['Plot'][249]

# to see last dictionary extracted from Plot
key_words_dict_scores

# to see last item in Key_words
df['Key_words'][249]

# to extract all genre into a list, only the first three actors into a list, and all directors into a list
df['Genre'] = df['Genre'].map(lambda x: x.split(','))
df['Actors'] = df['Actors'].map(lambda x: x.split(',')[:3])
df['Director'] = df['Director'].map(lambda x: x.split(','))

# create unique identity names by merging first & last name into one word, convert to lowercase 
for index, row in df.iterrows():
    row['Genre'] = [x.lower().replace(' ','') for x in row['Genre']]
    row['Actors'] = [x.lower().replace(' ','') for x in row['Actors']]
    row['Director'] = [x.lower().replace(' ','') for x in row['Director']]



####create word representation by combining column attributes to Bag_of_words#########

# to combine 4 lists (4 columns) of key words into 1 sentence under Bag_of_words column
df['Bag_of_words'] = ''
columns = ['Genre', 'Director', 'Actors', 'Key_words']

for index, row in df.iterrows():
    words = ''
    for col in columns:
        words += ' '.join(row[col]) + ' '
    row['Bag_of_words'] = words
    
# strip white spaces infront and behind, replace multiple whitespaces (if any)
df['Bag_of_words'] = df['Bag_of_words'].str.strip().str.replace('   ', ' ').str.replace('  ', ' ')

df = df[['Title','Bag_of_words']]



# an example to see what is in the Bag_of_words
df['Bag_of_words'][0]

#######create vector representation for Bag_of_words and the similarity matrix##

# to generate the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(df['Bag_of_words'])
count_matrix

# to generate the cosine similarity matrix (size 250 x 250)
# rows represent all movies; columns represent all movies
# cosine similarity: similarity = cos(angle) = range from 0 (different) to 1 (similar)
# all the numbers on the diagonal are 1 because every movie is identical to itself (cosine value is 1 means exactly identical)
# matrix is also symmetrical because the similarity between A and B is the same as the similarity between B and A.
# for other values eg 0.1578947, movie x and movie y has similarity value of 0.1578947

cosine_sim = cosine_similarity(count_matrix, count_matrix)
print(cosine_sim)

# to create a Series for movie titles which can be used as indices (each index is mapped to a movie title)
indices = pd.Series(df['Title'])
indices[:5]
@app.route('/')
def home():
    return render_template("movie.html")

@app.route('/movie',methods=['POST'])
def questions():
    
    int_features = [x for x in request.form.values()]
    print(int_features)

    


    

    return redirect(url_for('recommend'))

    ####### run and test the recommender model#########

    # this function takes in a movie title as input and returns the top 10 recommended (similar) movies

@app.route('/result', methods=['GET', 'POST'])
def recommend():
    
    if request.method == 'GET':
        return(render_template('movie.html'))

    if request.method == 'POST':
        
        title = request.form['title']
        recommended_movies = []
        idx = indices[indices == title].index[0]   # to get the index of the movie title matching the input movie
        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)   # similarity scores in descending order
        top_10_indices = list(score_series.iloc[1:11].index)   # to get the indices of top 10 most similar movies
        # [1:11] to exclude 0 (index 0 is the input movie itself)
    
        for i in top_10_indices:   # to append the titles of top 10 similar movies to the recommended_movies list
            recommended_movies.append(list(df['Title'])[i])
        #print('Movies:',recommended_movies) 
        
        movie_1 = recommended_movies[0]
        movie_2 = recommended_movies[1]
        #print('11111111111',movie_1)
        #print('22222222222',movie_2)
      
        # return recommended_movies
    
        mv = request.args.get('title')
        return (render_template('result.html', title=mv,m1 = movie_1, m2 = movie_2))
           
      

#recommend('The Dark Knight')
#recommend('Fargo')
#recommend('The Avengers')
#recommend('Fight Club')
#, cosine_sim = cosine_sim

if __name__ == '__main__':
   app.run(host = "127.0.0.1", port = 5000, debug=True)
