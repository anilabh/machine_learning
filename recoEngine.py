#Dataframe manipulation library
import pandas as pd
#Math functions, we'll only need the sqrt function so let's import only that
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

#Storing the movie information into a pandas dataframe
movies_df = pd.read_csv('/Users/anipandey/Documents/ML/coursera/ml-latest/movies.csv')
#Storing the user information into a pandas dataframe
ratings_df = pd.read_csv('/Users/anipandey/Documents/ML/coursera/ml-latest/ratings.csv')
#Head is a function that gets the first N rows of a dataframe. N's default is 5.
movies_df.head()

#Using regular expressions to find a year stored between parentheses
#We specify the parantheses so we don't conflict with movies that have years in their titles
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
#Removing the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
#Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
movies_df.head()

#Every genre is separated by a | so we simply have to call the split function on |
movies_df['genres'] = movies_df.genres.str.split('|')
print(movies_df.head())

#Since keeping genres in a list format isn't optimal for the content-based recommendation system technique, 
# we will use the One Hot Encoding technique to convert the list of genres to a vector where each column 
# corresponds to one possible value of the feature. This encoding is needed for feeding categorical data. 
# In this case, we store every different genre in columns that contain either 1 or 0. 1 shows that a movie 
# has that genre and 0 shows that it doesn't. Let's also store this dataframe in another variable since genres 
# won't be important for our first recommendation system.

#Copying the movie dataframe into a new one since we won't need to use the genre information in our first case.
moviesWithGenres_df = movies_df.copy()

#For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1
#Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
moviesWithGenres_df.head()

ratings_df.head()

#Drop removes a specified row or column from a dataframe
ratings_df = ratings_df.drop('timestamp', 1)
ratings_df.head()

userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ] 
inputMovies = pd.DataFrame(userInput)

#Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)
#Final input dataframe
#If a movie you added in above isn't here, then it might not be in the original 
#dataframe or it might spelled differently, please check capitalisation.

#Filtering out the movies from the input
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]

#Resetting the index to avoid future issues
userMovies = userMovies.reset_index(drop=True)
#Dropping unnecessary issues due to save memory and to avoid issues
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

#Now we're ready to start learning the input's preferences!
#To do this, we're going to turn each genre into weights. We can do this by using the input's reviews and multiplying 
# them into the input's genre table and then summing up the resulting table by column. 
#This operation is actually a dot product between a matrix and a vector, so we can simply accomplish by calling Pandas's "dot" function.

#Dot produt to get weights

def contentbasedfiltering():
    #Now we're ready to start learning the input's preferences!
    #To do this, we're going to turn each genre into weights. We can do this by using the input's reviews and multiplying 
    # them into the input's genre table and then summing up the resulting table by column. 
    #This operation is actually a dot product between a matrix and a vector, so we can simply accomplish by calling Pandas's "dot" function.

    #Dot produt to get weights
    userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
    #The user profile

    #Now, we have the weights for every of the user's preferences. 
    # This is known as the User Profile. Using this, we can recommend movies that satisfy the user's preferences.
    #Now let's get the genres of every movie in our original dataframe
    genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
    #And drop the unnecessary information
    genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
    genreTable.head()

    #With the input's profile and the complete list of movies and their genres in hand, we're going to take the weighted average 
    # of every movie based on the input profile and recommend the top twenty movies that most satisfy it.
    #Multiply the genres by the weights and then take the weighted average
    recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
    recommendationTable_df.head()

    #Sort our recommendations in descending order
    recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
    #Just a peek at the values
    recommendationTable_df.head()

    #The final recommendation table
    movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())]
    print(recommendationTable_df)

    # Advantages and Disadvantages of Content-Based Filtering
    # Advantages
    # Learns user's preferences
    # Highly personalized for the user
    # Disadvantages
    # Doesn't take into account what others think of the item, so low quality item recommendations might happen
    # Extracting data is not always intuitive
    # Determining what characteristics of the item the user dislikes or likes is not always obvious

def collaborativefintering(inputMovies):
    # The process for creating a User Based recommendation system is as follows:

    # Select a user with the movies the user has watched
    # Based on his rating to movies, find the top X neighbours
    # Get the watched movie record of the user for each neighbour.
    # Calculate a similarity score using some formula
    # Recommend the items with the highest score

    #Filtering out users that have watched movies that the input has watched and storing it
    userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
    userSubset.head()
    #Groupby creates several sub dataframes where they all have the same value in the column specified as the parameter
    userSubsetGroup = userSubset.groupby(['userId'])
    #lets look at one of the users, e.g. the one with userID=1130
    userSubsetGroup.get_group(1130)
    #Let's also sort these groups so the users that share the most movies in common with the input have higher priority. 
    # This provides a richer recommendation since we won't go through every single user.
    #Sorting it so users with movie most in common with the input will have priority
    userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)
    #first user
    print(userSubsetGroup[0:3])

    #Next, we are going to compare all users (not really all !!!) to our specified user and find the one that is most similar.
    #we're going to find out how similar each user is to the input through the Pearson Correlation Coefficient.
    #In our case, a 1 means that the two users have similar tastes while a -1 means the opposite.
    #take a sub set to optimise time
    userSubsetGroup = userSubsetGroup[0:100]

    #Now, we calculate the Pearson Correlation between input user and subset group, and store it in a dictionary, where the key is the user Id and the value is the coefficient
    #Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
    pearsonCorrelationDict = {}

    #For every user group in our subset
    for name, group in userSubsetGroup:
        #Let's start by sorting the input and current user group so the values aren't mixed up later on
        group = group.sort_values(by='movieId')
        inputMovies = inputMovies.sort_values(by='movieId')
        #Get the N for the formula
        nRatings = len(group)
        #Get the review scores for the movies that they both have in common
        temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
        #And then store them in a temporary buffer variable in a list format to facilitate future calculations
        tempRatingList = temp_df['rating'].tolist()
        #Let's also put the current user group reviews in a list format
        tempGroupList = group['rating'].tolist()
        #Now let's calculate the pearson correlation between two users, so called, x and y
        Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
        Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
        Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
        
        #If the denominator is different than zero, then divide, else, 0 correlation.
        if Sxx != 0 and Syy != 0:
            pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
        else:
            pearsonCorrelationDict[name] = 0
    
    pearsonCorrelationDict.items()
    pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
    pearsonDF.columns = ['similarityIndex']
    pearsonDF['userId'] = pearsonDF.index
    pearsonDF.index = range(len(pearsonDF))
    pearsonDF.head()

    #Now let's get the top 50 users that are most similar to the input.
    topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
    print(topUsers.head()) 

    #Now, let's start recommending movies to the input user.

    #Rating of selected users to all movies
    #We're going to do this by taking the weighted average of the ratings of the movies using the 
    # Pearson Correlation as the weight. But to do this, we first need to get the movies watched by 
    # the users in our pearsonDF from the ratings dataframe and then store their correlation in a new 
    # column called _similarityIndex". This is achieved below by merging of these two tables.
    topUsersRating=topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
    topUsersRating.head()

    #Now all we need to do is simply multiply the movie rating by its weight (The similarity index), then sum up the new ratings and divide it by the sum of the weights.
    #We can easily do this by simply multiplying two columns, then grouping up the dataframe by movieId and then dividing two columns:
    #It shows the idea of all similar users to candidate movies for the input user:
    #Multiplies the similarity by the user's ratings
    topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
    topUsersRating.head()
    #Applies a sum to the topUsers after grouping it up by userId
    tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
    tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
    tempTopUsersRating.head()
    #Creates an empty dataframe
    recommendation_df = pd.DataFrame()
    #Now we take the weighted average
    recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
    recommendation_df['movieId'] = tempTopUsersRating.index
    recommendation_df.head()

    #Now let's sort it and see the top 20 movies that the algorithm recommended!
    recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
    recommendation_df.head(10)
    print(movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())])


#contentbasedfiltering()
collaborativefintering(inputMovies)




