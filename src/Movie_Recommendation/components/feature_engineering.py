import os
import pandas as pd
from .. import logger
from ..entity.config_entity import FeatureEngineeringConfig
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from thefuzz import fuzz
from thefuzz import process
import numpy as np
from surprise import Dataset
from surprise import Reader


class FeatureEngineering:
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

    def get_data(self):
        """
        Load and return the rating and movie and data
        """
        try:
            data_path = self.config.data_path
            
            ratings_file = None
            movies_file = None
            data_file = None
            
            # Search in current directory and subdirectories
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if 'ratings' in file.lower() and file.endswith('.csv'):
                        ratings_file = os.path.join(root, file)
                    elif 'movies' in file.lower() and file.endswith('.csv'):
                        movies_file = os.path.join(root, file)
                    elif 'data' in file.lower() and file.endswith('.csv'):
                        data_file = os.path.join(root, file)    
            
            if ratings_file:
                ratings = pd.read_csv(ratings_file)
                logger.info(f"Loaded ratings data from {ratings_file}: {ratings.shape}")
            else:
                logger.warning(f"Ratings file not found in {data_path}")
                ratings = None
            
            if movies_file:
                movies = pd.read_csv(movies_file)
                logger.info(f"Loaded movies data from {movies_file}: {movies.shape}")
            else:
                logger.warning(f"Movies file not found in {data_path}")
                movies = None

            if data_file:
                data = pd.read_csv(data_file)
                logger.info(f"Loaded ratings data from {data_file}: {data.shape}")
            else:
                logger.warning(f"Ratings file not found in {data_path}")
                ratings = None    
            
            return ratings, movies , data

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise e
        
    
    def create_train_test_data(self):
        # Creating the train test set
        ratings, movies , movie_ratings = self.get_data()
        file_path = self.config.root_dir

        if not os.path.isfile(file_path + "/TrainData.pkl"):
            print("Creating Train Data and saving it..")
            movie_ratings.iloc[:int(movie_ratings.shape[0] * 0.80)].to_pickle(file_path + "/TrainData.pkl")
            Train_Data = pd.read_pickle(file_path + "/TrainData.pkl")
            Train_Data.reset_index(drop = True, inplace = True)
        else:
            print("Loading Train Data..")
            Train_Data = pd.read_pickle(file_path + "/TrainData.pkl")
            Train_Data.reset_index(drop = True, inplace = True)

        if not os.path.isfile(file_path + "/TestData.pkl"):
            print("Creating Test Data and saving it..")
            movie_ratings.iloc[int(movie_ratings.shape[0] * 0.80):].to_pickle(file_path + "/TestData.pkl")
            Test_Data = pd.read_pickle(file_path + "/TestData.pkl")
            Test_Data.reset_index(drop = True, inplace = True)
        else:
            print("Loading Test Data..")
            Test_Data = pd.read_pickle(file_path + "/TestData.pkl")
            Test_Data.reset_index(drop = True, inplace = True)
        logger.info(f"create/save {Train_Data} and shape : {Train_Data.shape}")
        logger.info(f"create/save {Test_Data} and shape : {Test_Data.shape}")
        return Train_Data ,Test_Data   

    def create_matrices(self):
        Train_Data, Test_Data = self.create_train_test_data()
        file_path = self.config.root_dir
        # Creating/loading user-movie sparse matrix for train data

        startTime = datetime.now()

        print("Creating USER_ITEM sparse matrix for train Data..")
 
        if os.path.isfile(file_path + "/TrainUISparseData.npz"):
            print("Sparse Data is already present in your disk, no need to create further. Loading Sparse Matrix")
            TrainUISparseData = sparse.load_npz(file_path + "/TrainUISparseData.npz")
            print("Shape of Train Sparse matrix = "+str(TrainUISparseData.shape))    
        else:
            print("We are creating sparse data..")
            TrainUISparseData = sparse.csr_matrix((Train_Data.rating, (Train_Data.userId, Train_Data.movieId)))
            print("Creation done. Shape of sparse matrix : ", str(TrainUISparseData.shape))
            print("Saving it into disk for furthur usage.")
            sparse.save_npz(file_path + "/TrainUISparseData.npz", TrainUISparseData)
            print("Done\n")

        print("Time taken : ", datetime.now() - startTime)
        
        rows,cols = TrainUISparseData.shape
        presentElements = TrainUISparseData.count_nonzero()

        print("Sparsity Of Train matrix : {}% ".format((1-(presentElements/(rows*cols)))*100))


        # Creating/loading user-movie sparse matrix for test data

        startTime = datetime.now()

        print("Creating USER_ITEM sparse matrix for test Data..")

        if os.path.isfile(file_path + "/TestUISparseData.npz"):
            print("Sparse Data is already present in your disk, no need to create further. Loading Sparse Matrix")
            TestUISparseData = sparse.load_npz(file_path + "/TestUISparseData.npz")
            print("Shape of Test Sparse Matrix : ", str(TestUISparseData.shape))
        else:
            print("We are creating sparse data..")
            TestUISparseData = sparse.csr_matrix((Test_Data.rating, (Test_Data.userId, Test_Data.movieId)))
            print("Creation done. Shape of sparse matrix : ", str(TestUISparseData.shape))
            print("Saving it into disk for furthur usage.")
            sparse.save_npz(file_path + "/TestUISparseData.npz", TestUISparseData)
            print("Done\n")

        print("Time Taken : ", datetime.now() - startTime)
        rows,cols = TestUISparseData.shape
        presentElements = TestUISparseData.count_nonzero()

        print("Sparsity Of Test matrix : {}% ".format((1-(presentElements/(rows*cols)))*100))
        return TrainUISparseData , TestUISparseData
        
    def getAverageRatings(self,sparseMatrix, if_user):

        #axis = 1 means rows and axis = 0 means columns 
        ax = 1 if if_user else 0

        sumOfRatings = sparseMatrix.sum(axis = ax).A1  
        noOfRatings = (sparseMatrix!=0).sum(axis = ax).A1  
        rows, cols = sparseMatrix.shape
        averageRatings = {i: sumOfRatings[i]/noOfRatings[i] for i in range(rows if if_user else cols) if noOfRatings[i]!=0}
    
        return averageRatings
    
    def computing_item_item_similarity_matrices(self):
        # Computing item-item similarity matrix for the train data
        # We have 138K sized sparse vectors using which a 19K x 19K movie similarity matrix would be calculated
        file_path = self.config.root_dir
        start = datetime.now()
        TrainUISparseData , TestUISparseData= self.create_matrices()

        if not os.path.isfile(file_path + "/m_m_similarity.npz"):
            print("Movie-Movie Similarity file does not exist in your disk. Creating Movie-Movie Similarity Matrix...")
            m_m_similarity = cosine_similarity(TrainUISparseData.T, dense_output = False)
            print("Dimension of Matrix : ", m_m_similarity.shape)
            print("Storing the Movie Similarity matrix on disk for further usage")
            sparse.save_npz(file_path + "/m_m_similarity.npz", m_m_similarity)
        else:
            print("File exists in the disk. Loading the file...")
            m_m_similarity = sparse.load_npz(file_path + "/m_m_similarity.npz")
            print("Dimension of Matrix : ", m_m_similarity.shape)
    
        print("The time taken to compute movie-movie similarity matrix is : ", datetime.now() - start)
        return m_m_similarity


    def get_movie_list_in_training(self):
        Train_Data, Test_Data = self.create_train_test_data()
        movie_list_in_training = Train_Data.drop_duplicates(subset=["title"], keep="first")[["movieId", "title", "genres"]]
        movie_list_in_training = movie_list_in_training.reset_index(drop=True)
        return movie_list_in_training
    

    def GetSimilarMoviesUsingMovieMovieSimilarity(self,movie_name, num_of_similar_movies):
        movie_list_in_training = self.get_movie_list_in_training()
        m_m_similarity = self.computing_item_item_similarity_matrices()
        matches = process.extract(movie_name, movie_list_in_training["title"], scorer=fuzz.partial_ratio)
        if len(matches) == 0:
            return "No Match Found"
        movie_id = movie_list_in_training.iloc[matches[0][2]]["movieId"]
        similar_movie_id_list = np.argsort(-m_m_similarity[movie_id].toarray().ravel())[0:num_of_similar_movies+1]
        sm_df = movie_list_in_training[movie_list_in_training["movieId"].isin(similar_movie_id_list)]
        sm_df["order"] = sm_df.apply(lambda x: list(similar_movie_id_list).index(x["movieId"]), axis=1)
    
        return sm_df.sort_values("order")    
    

    # Here, we are calculating user-user similarity matrix only for first 100 users in our sparse matrix. And we are calculating 
    # Top 100 most similar users with them.

    def getUser_UserSimilarity(self,sparseMatrix, top = 100):
        startTimestamp20 = datetime.now()  
    
        row_index, col_index = sparseMatrix.nonzero()
        rows = np.unique(row_index)
        similarMatrix = np.zeros(13849300).reshape(138493,100)    # 138493*100 = 13849300. As we are building similarity matrix only 
        #for top 100 most similar users.
        timeTaken = []
        howManyDone = 0
        for row in rows[:top]:
            howManyDone += 1
            startTimestamp = datetime.now().timestamp()  #it will give seconds elapsed
            sim = cosine_similarity(sparseMatrix.getrow(row), sparseMatrix).ravel()
            top100_similar_indices = sim.argsort()[-top:]
            top100_similar = sim[top100_similar_indices]
            similarMatrix[row] = top100_similar
            timeforOne = datetime.now().timestamp() - startTimestamp
            timeTaken.append(timeforOne)
            if howManyDone % 20 == 0:
                print("Time elapsed for {} users = {}sec".format(howManyDone, (datetime.now() - startTimestamp20)))
        print("Average Time taken to compute similarity matrix for 1 user = "+str(sum(timeTaken)/len(timeTaken))+"seconds")
        return similarMatrix
  
    # Calculating user-user similarity only for particular users in our sparse matrix and return user_ids

    def Calculate_User_User_Similarity(self, sparseMatrix, user_id, num_of_similar_users=10):
        TrainUISparseData, TestUISparsedata = self.create_matrices()
        row_index, col_index = TrainUISparseData.nonzero()
        unique_user_id = np.unique(row_index)
        print("Max User id is :", np.max(unique_user_id)) 
        if user_id in unique_user_id:
            # Calculating the cosine similarity for user_id with all the "userId"
            sim = cosine_similarity(sparseMatrix.getrow(user_id), sparseMatrix).ravel()
            # Sorting the indexs(user_id) based on the similarity score for all the user ids
            top_similar_user_ids = sim.argsort()[::-1]
            # Sorted the similarity values
            top_similarity_values = sim[top_similar_user_ids]

        return top_similar_user_ids[1: num_of_similar_users+1]

    def extract_features(self):
        file_path = self.config.root_dir
        train_sample_sparse, test_sample_sparse = self.create_sample_sparse_matrix()
        data_rows = self.CreateFeaturesForTrainData(train_sample_sparse, train_sample_sparse)
        test_data_rows = self.CreateFeaturesForTrainData(test_sample_sparse, train_sample_sparse)
        names = ["User_ID", "Movie_ID", "Global_Average", "User_Average", "Movie_Average", "SUR1", "SUR2", "SUR3", "SUR4", "SUR5", "SMR1", "SMR2", "SMR3", "SMR4", "SMR5", "Rating"]
        train_regression_data = pd.DataFrame(data_rows, columns=names)
        test_regression_data = pd.DataFrame(test_data_rows, columns=names)
        train_regression_data.to_csv(file_path + "/Training_Data_For_Regression.csv")
        test_regression_data.to_csv(file_path + "/Testing_Data_For_Regression.csv")

    # Since the given dataset might not completely fit into computaton capacity that we have, we will sample the data and work it

    # Function for Sampling random movies and users to reduce the size of rating matrix
    def get_sample_sparse_matrix(self,sparseMatrix, n_users, n_movies, matrix_name):

        np.random.seed(15)   #this will give same random number everytime, without replacement
        startTime = datetime.now()
        file_path = self.config.root_dir
        users, movies, ratings = sparse.find(sparseMatrix)
        uniq_users = np.unique(users)
        uniq_movies = np.unique(movies)

        userS = np.random.choice(uniq_users, n_users, replace = False)
        movieS = np.random.choice(uniq_movies, n_movies, replace = False)
        mask = np.logical_and(np.isin(users, userS), np.isin(movies, movieS))
        sparse_sample = sparse.csr_matrix((ratings[mask], (users[mask], movies[mask])), shape = (max(userS)+1, max(movieS)+1))
 
        print("Sparse Matrix creation done. Saving it for later use.")
        sparse.save_npz(file_path + "/" + matrix_name, sparse_sample)
        print("Shape of Sparse Sampled Matrix = " + str(sparse_sample.shape))    
        print("Time taken : ", datetime.now() - startTime) 
             

    def create_sample_sparse_matrix(self):
        TrainUISparseData,TestUISparseData = self.create_matrices()
        file_path = self.config.root_dir
        if not os.path.isfile(file_path + "/TrainUISparseData_Sample.npz"):
            print("Sample sparse matrix is not present in the disk. We are creating it...")
            train_sample_sparse = self.get_sample_sparse_matrix(TrainUISparseData, 5000, 1000, "TrainUISparseData_Sample.npz")
        else:
            print("File is already present in the disk. Loading the file...")
            train_sample_sparse = sparse.load_npz(file_path + "/TrainUISparseData_Sample.npz")
            print("Shape of Train Sample Sparse Matrix = " + str(train_sample_sparse.shape))

        
        if not os.path.isfile(file_path + "/TestUISparseData_Sample.npz"):
            print("Sample sparse matrix is not present in the disk. We are creating it...")
            test_sample_sparse = self.get_sample_sparse_matrix(TestUISparseData, 2000, 200, "TestUISparseData_Sample.npz")
        else:
            print("File is already present in the disk. Loading the file...")
            test_sample_sparse = sparse.load_npz(file_path + "/TestUISparseData_Sample.npz")
            print("Shape of Test Sample Sparse Matrix = " + str(test_sample_sparse.shape))    
        return train_sample_sparse, test_sample_sparse    

    def CreateFeaturesForTrainData(self,SampledSparseData, TrainSampledSparseData):

        startTime = datetime.now()
        globalAvgRating, globalAvgMovies, globalAvgUsers = self.calculate_global_avg_rating_use()
        # Extracting userId list, movieId list and Ratings
        sample_users, sample_movies, sample_ratings = sparse.find(SampledSparseData)
    
        print("No. of rows in the returned dataset : ", len(sample_ratings))
    
        count = 0
        data = []
    
        for user, movie, rating in zip(sample_users, sample_movies, sample_ratings):

            row = list()

#----------------------------------Appending "user Id" average, "movie Id" average & global average rating-----------#
            row.append(user)  
            row.append(movie) 
            row.append(globalAvgRating) 

#----------------------------------Appending "user" average, "movie" average & rating of "user""movie"-----------#
            try:
                row.append(globalAvgUsers[user])
            except (KeyError):
                global_average_rating = globalAvgRating
                row.append(global_average_rating)
            except:
                raise
            try:            
                row.append(globalAvgMovies[movie])
            except (KeyError):
                global_average_rating = globalAvgRating
                row.append(global_average_rating)
            except:
                raise

#----------------------------------Ratings given to "movie" by top 5 similar users with "user"--------------------#
            try:
                similar_users = cosine_similarity(TrainSampledSparseData[user], TrainSampledSparseData).ravel()
                similar_users_indices = np.argsort(-similar_users)[1:]
                similar_users_ratings = TrainSampledSparseData[similar_users_indices, movie].toarray().ravel()
                top_similar_user_ratings = list(similar_users_ratings[similar_users_ratings != 0][:5])
                top_similar_user_ratings.extend([globalAvgMovies[movie]]*(5-len(top_similar_user_ratings)))
            #above line means that if top 5 ratings are not available then rest of the ratings will be filled by "movie" average
            #rating. Let say only 3 out of 5 ratings are available then rest 2 will be "movie" average rating.
                row.extend(top_similar_user_ratings)
        #########Cold Start Problem, for a new user or a new movie######### 
            except (IndexError, KeyError):
                global_average_rating = [globalAvgRating]*5
                row.extend(global_average_rating)
            except:
                raise

#----------------------------------Ratings given by "user" to top 5 similar movies with "movie"------------------#
            try:
                similar_movies = cosine_similarity(TrainSampledSparseData[:,movie].T, TrainSampledSparseData.T).ravel()
                similar_movies_indices = np.argsort(-similar_movies)[1:]
                similar_movies_ratings = TrainSampledSparseData[user, similar_movies_indices].toarray().ravel()
                top_similar_movie_ratings = list(similar_movies_ratings[similar_movies_ratings != 0][:5])
                top_similar_movie_ratings.extend([globalAvgUsers[user]]*(5-len(top_similar_movie_ratings)))
            #above line means that if top 5 ratings are not available then rest of the ratings will be filled by "user" average
            #rating. Let say only 3 out of 5 ratings are available then rest 2 will be "user" average rating.
                row.extend(top_similar_movie_ratings)
        ########Cold Start Problem, for a new user or a new movie#########
            except (IndexError, KeyError):
                global_average_rating = [globalAvgRating] * 5
                row.extend(global_average_rating)
            except:
                raise
              
#----------------------------------Appending rating of "user""movie"-----------#
            row.append(rating)

            count += 1
        
            data.append(row)
        
            if count % 5000 == 0:
                print("Done for {}. Time elapsed: {}".format(count, (datetime.now() - startTime)))

        print("Total Time for {} rows = {}".format(len(data), (datetime.now() - startTime)))
        print("Completed..")
        return data        

    def calculate_global_avg_rating_user(self):
        train_sample_sparse, test_sample_sparse = self.create_sample_sparse_matrix()
        globalAvgRating = np.round((train_sample_sparse.sum()/train_sample_sparse.count_nonzero()), 2)
        globalAvgMovies = self.getAverageRatings(train_sample_sparse, False)
        globalAvgUsers = self.getAverageRatings(train_sample_sparse, True)
        print("Global average of all movies ratings in Train Set is : ", globalAvgRating)
        print("No. of ratings in the train matrix is : ", train_sample_sparse.count_nonzero()) 
        return globalAvgRating, globalAvgMovies, globalAvgUsers