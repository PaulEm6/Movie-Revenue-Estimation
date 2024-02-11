#https://www.kaggle.com/code/aschittko/movie-revenue-prediction

from matplotlib.widgets import Lasso
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np
import time

# Record the start time
start_time = time.time()

nltk.download('stopwords')

#Get dataset
data =pd.read_csv('tmdb_5000_movies.csv')

###################
# FEATURE SELECTION
###################

features = [
    'genres', 'runtime', 'keywords', 'budget', 'release_date', 'revenue','vote_average','vote_count','production_companies'
]

X = data[features]

##################################################################
# DATA PREPROCESSING: REMOVING MISSING VALUES AND TOO SMALL VALUES
##################################################################

X = X.dropna()

zeros_budget = X['budget'].eq(0).sum()
zeros_revenue = X['revenue'].eq(0).sum()

X = X.loc[X['revenue'] > 1000]
X = X.loc[X['budget'] > 1000]

#########################################################################################
# DATA PREPROCESSING: ENCODING CATEGORICAL DATA AND TRANSFORMING DATES TO NUMERICAL VALUE
#########################################################################################

#Using TFDIF to encode: (keywords, genres, titles and production companies), first we remove stopwords

stop_words = set(stopwords.words('english'))

X['keywords'] = X['keywords'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
X['genres'] = X['genres'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
#X['title'] = X['title'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
X['production_companies'] = X['production_companies'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
#X['production_countries'] = X['production_countries'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

#We create the vectorizer that will be used to encode the string values
tfidf_vectorizer_keywords = TfidfVectorizer(min_df=10)  # Adjust min_df as needed
tfidf_vectorizer_genres = TfidfVectorizer(min_df=10)  # Adjust min_df as needed
#tfidf_vectorizer_title = TfidfVectorizer(min_df=10)  # Adjust min_df as needed
tfidf_vectorizer_production_companies = TfidfVectorizer(min_df=10)  # Adjust min_df as needed
#tfidf_vectorizer_production_countries = TfidfVectorizer(min_df=10)  # Adjust min_df as needed

#We fit the vectorizer to the column of string values
keyword_tfidf_keywords = tfidf_vectorizer_keywords.fit_transform(X['keywords'])
keyword_tfidf_genres = tfidf_vectorizer_genres.fit_transform(X['genres'])
#keyword_tfidf_title = tfidf_vectorizer_title.fit_transform(X['title'])
keyword_tfidf_production_companies = tfidf_vectorizer_production_companies.fit_transform(X['production_companies'])
#keyword_tfidf_production_countries = tfidf_vectorizer_production_countries.fit_transform(X['production_countries'])

#We add the encoded values to the dataframe
X = pd.concat([X, pd.DataFrame(keyword_tfidf_keywords.toarray())], axis=1)
X = pd.concat([X, pd.DataFrame(keyword_tfidf_genres.toarray())], axis=1)
#X = pd.concat([X, pd.DataFrame(keyword_tfidf_title.toarray())], axis=1)
X = pd.concat([X, pd.DataFrame(keyword_tfidf_production_companies.toarray())], axis=1)
#X = pd.concat([X, pd.DataFrame(keyword_tfidf_production_countries.toarray())], axis=1)

# Feature Engineering
# Convert release_date to the year, month, and day
X['release_year'] = pd.to_datetime(X['release_date']).dt.year
X['release_month'] = pd.to_datetime(X['release_date']).dt.month
X['release_day'] = pd.to_datetime(X['release_date']).dt.day

#We drop the old columns that contain the string values
X.drop('keywords', axis = 1, inplace=True)
X.drop('genres', axis = 1, inplace=True)
X.drop('release_date', axis = 1, inplace=True)
#X.drop('title', axis = 1, inplace=True)
X.drop('production_companies', axis = 1, inplace=True)
#X.drop('production_countries', axis = 1, inplace=True)

####################################
#DATA PREPROCESSING: FEATURE SCALING
####################################

# Create a StandardScaler instance for each numerical column, this instance will contain the std and mean of the column
scaler_budget = StandardScaler()
scaler_year = StandardScaler()
scaler_runtime = StandardScaler()
scaler_votecount = StandardScaler()
#scaler_popularity = StandardScaler()

# Fit the scaler to the data and transform the columns
column_budget = 'budget'
column_year = 'release_year'
column_runtime = 'runtime'
column_vote_count = 'vote_count'
#column_popularity = 'popularity'

X[column_budget] = scaler_budget.fit_transform(X[[column_budget]])
X[column_year] = scaler_year.fit_transform(X[[column_year]])
X[column_runtime] = scaler_runtime.fit_transform(X[[column_runtime]])
X[column_vote_count] = scaler_votecount.fit_transform(X[[column_vote_count]])
#X[column_popularity] = scaler_votecount.fit_transform(X[[column_popularity]])

#####################
#MODEL TRAINING MODEL
#####################

X_def = X.dropna()

model1 = []
model2 = []
model3 = []
model4 = []

for i in range(1,13):
    
    # Select relevant columns
    X = X_def[X_def['release_month'] == i]
    target = 'revenue'
    y = X[target]
    X.drop(['revenue', "release_month"], axis = 1)
    X.columns = X.columns.astype(str)

    print("\nTraining model for month number: " + str(i))
    print("Shape of dataset for this month is: " + str(X.shape))

    pca = PCA(n_components=0.9)
    X_pca = pca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2)

    lasso = Lasso()
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}
    grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_lasso_model = grid_search.best_estimator_
    lasso_predictions = best_lasso_model.predict(X_test)
    lasso_rmse = mean_squared_error(y_test, lasso_predictions, squared=False)
    #print(f"Lasso RMSE on test set: {lasso_rmse}")
    model1.append(lasso_rmse)

    ridge = Ridge()
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}
    ridge_grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
    ridge_grid_search.fit(X_train, y_train)
    best_ridge_model = ridge_grid_search.best_estimator_
    ridge_predictions = best_ridge_model.predict(X_test)
    ridge_rmse = mean_squared_error(y_test, ridge_predictions, squared=False)
    #print(f"Ridge RMSE on test set: {ridge_rmse}")
    model2.append(ridge_rmse)

    elastic_net = ElasticNet()
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1.0],
        'l1_ratio': [0.1, 0.5, 0.7, 0.9]
    }
    elastic_net_grid_search = GridSearchCV(elastic_net, param_grid, cv=5, scoring='neg_mean_squared_error')
    elastic_net_grid_search.fit(X_train, y_train)
    best_elastic_net_model = elastic_net_grid_search.best_estimator_
    elastic_net_predictions = best_elastic_net_model.predict(X_test)
    elastic_net_rmse = mean_squared_error(y_test, elastic_net_predictions, squared=False)
    #print(f"Elastic Net RMSE on test set: {elastic_net_rmse}")
    model3.append(elastic_net_rmse)
    
    rf_regressor = RandomForestRegressor()
    n_estimators = [x * 10 for x in list(range(1,30))]
    param_grid = {
        'n_estimators': n_estimators
    }
    grid_search = GridSearchCV(rf_regressor, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    rmse_scorer = make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)), greater_is_better=False)
    best_rf_regressor = RandomForestRegressor(**best_params)
    best_rf_regressor.fit(X_train, y_train)
    y_pred = best_rf_regressor.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Forest RMSE on test set: {rf_rmse}")
    model4.append(rf_rmse)

print("\nFinal results")
#print("Lasso rmse average " + str(sum(model1)/len(model1)))
#print("Ridge rmse average " + str(sum(model2)/len(model2)))
#print("Elastic Net rmse average " + str(sum(model3)/len(model3)))
print("Random forest rmse average " + str(sum(model4)/len(model4)))

# Record the end time
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Time taken to train the model: {elapsed_time:.2f} seconds")
