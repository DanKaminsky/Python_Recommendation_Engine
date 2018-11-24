# Recommendation Engine

# Loading libraries
import pandas as pd
import scipy.sparse as sparse
import numpy as np
from scipy.sparse.linalg import spsolve

# Loading the data into a DataFrame and taking a look into
# the content of each file to understand them better.
Sales2014 = pd.read_csv('Sales Transactions - 2014.csv')
print '\nNumber of rows and columns 2014:', Sales2014.shape, '\n'
#print '\nSales 2014:\n', Sales2014.head(n=5), '\n'

Sales2015 = pd.read_csv('Sales Transactions - 2015.csv')
print '\nNumber of rows and columns 2015:', Sales2015.shape, '\n'
#print '\nSales 2015:\n', Sales2015.head(n=5), '\n'

Sales2016 = pd.read_csv('Sales Transactions - 2016.csv')
print '\nNumber of rows and columns 2016:', Sales2016.shape, '\n'
#print '\nSales 2016:\n', Sales2016.head(n=5), '\n'


# Appending Sales 2014 and Sales 2015 in a DataFrame
Sales_2014_2015 = Sales2014.append(Sales2015)
print '\nSales 2014 and 2015 appended:\n', Sales_2014_2015.head(n=5), '\n'
# Checking for missing values in the data
print '\nChecking for missing values in Sales_2014_2015:\n', Sales_2014_2015.info(null_counts=True), '\n' # Segmentation has a few nulls
Sales_2014_2015.to_csv("Sales_2014_2015.csv", index = False)

# Creating an Item lookup table
item_lookup = Sales_2014_2015[['Item Number']].drop_duplicates() # Only get unique items
item_lookup['Item Number'] = item_lookup['Item Number'].astype(str) # Encode as strings for future lookup ease


Sales_2014_2015['Customer Number'] = Sales_2014_2015['Customer Number'].astype(int) # Converting to int for Customer Number
Sales_2014_2015 = Sales_2014_2015[['Item Number', 'Net Sales $', 'Customer Number']] # Getting rid of unnecessary info
grouped_Sales14_15 = Sales_2014_2015.groupby(['Customer Number', 'Item Number']).sum().reset_index() # Groupping together
grouped_Sales14_15['Net Sales $'].loc[grouped_Sales14_15['Net Sales $'] == 0] = 1 # Replace a sum of zero purchases with a one to indicate purchased (taking care of returns)
grouped_purchased = grouped_Sales14_15[grouped_Sales14_15['Net Sales $'] > 0] # Only get customers where purchase totals were positive

# Looking at the final resulting matrix of grouped purchases
print '\nMatrix of grouped purchases:\n', grouped_purchased.head(), '\n'

# Creating a sparse ratings matrix of Customers and Items 
customers = list(np.sort(grouped_purchased['Customer Number'].unique())) # Get unique Customers
items = list(grouped_purchased['Item Number'].unique()) # Get unique Items that were purchased
sales = list(grouped_purchased['Net Sales $']) # All purchases
print '\nCustomers count:\n', len(customers), '\n'
rows = grouped_purchased['Customer Number'].astype('category', categories = customers).cat.codes 
# Get the associated row indices
cols = grouped_purchased['Item Number'].astype('category', categories = items).cat.codes 
# Get the associated column indices
purchases_sparse = sparse.csr_matrix((sales, (rows, cols)), shape=(len(customers), len(items)))


# Creating a Training and Validation Set

# Loading the random library
import random

# Function to create the Training and Validation Set
def make_train(ratings, pct_test = 0.2):
    test_set = ratings.copy() # Make a copy of the original set to be the test set. 
    test_set[test_set != 0] = 1 # Store the test set as a binary preference matrix
    training_set = ratings.copy() # Make a copy of the original data we can alter as our training set. 
    nonzero_inds = training_set.nonzero() # Find the indices in the ratings data where an interaction exists
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1])) # Zip these pairs together of user,item index into list
    random.seed(0) # Set the random seed to zero for reproducibility
    num_samples = int(np.ceil(pct_test*len(nonzero_pairs))) # Round the number of samples needed to the nearest integer
    samples = random.sample(nonzero_pairs, num_samples) # Sample a random number of user-item pairs without replacement
    user_inds = [index[0] for index in samples] # Get the user row indices
    item_inds = [index[1] for index in samples] # Get the item column indices
    training_set[user_inds, item_inds] = 0 # Assign all of the randomly chosen user-item pairs to zero
    training_set.eliminate_zeros() # Get rid of zeros in sparse array storage after update to save space
    return training_set, test_set, list(set(user_inds)) # Output the unique list of user rows that were altered  

# This will return the training, the binary 0/1 (purchase/not) test set, and list of users with item/s masked
product_train, product_test, product_users_altered = make_train(purchases_sparse, pct_test = 0.2)

# Implementing ALS for Implicit Feedback
def implicit_weighted_ALS(training_set, lambda_val = 0.1, alpha = 40, iterations = 10, rank_size = 20, seed = 0):

    # first set up our confidence matrix    
    conf = (alpha*training_set)
                                
    num_user = conf.shape[0]
    num_item = conf.shape[1] # Get the size of our original ratings matrix, m x n
    
    # initialize our X/Y feature vectors randomly with a set seed
    rstate = np.random.RandomState(seed)
    
    X = sparse.csr_matrix(rstate.normal(size = (num_user, rank_size))) # Random numbers in a m x rank shape
    Y = sparse.csr_matrix(rstate.normal(size = (num_item, rank_size)))
                                                                
    X_eye = sparse.eye(num_user)
    Y_eye = sparse.eye(num_item)
    lambda_eye = lambda_val * sparse.eye(rank_size)
     
    # Begin iterations
   
    for iter_step in range(iterations): # Iterate back and forth between solving X given fixed Y and vice versa
        # Compute yTy and xTx at beginning of each iteration to save computing time
        yTy = Y.T.dot(Y)
        xTx = X.T.dot(X)
        # Being iteration to solve for X based on fixed Y
        for u in range(num_user):
            conf_samp = conf[u,:].toarray()
            pref = conf_samp.copy() 
            pref[pref != 0] = 1 # Create binarized preference vector 
            conf_samp = conf_samp + 1 
            CuI = sparse.diags(conf_samp, [0]) # Get Cu - I term
            yTCuIY = Y.T.dot(CuI).dot(Y) # This is the yT(Cu-I)Y term 
            yTCupu = Y.T.dot(CuI + Y_eye).dot(pref.T) # This is the yTCuPu term
                                                      # Cu - I + I = Cu
            X[u] = spsolve(yTy + yTCuIY + lambda_eye, yTCupu) 
            # Solve for Xu = ((yTy + yT(Cu-I)Y + lambda*I)^-1)yTCuPu 
        # Begin iteration to solve for Y based on fixed X 
        for i in range(num_item):
            conf_samp = conf[:,i].T.toarray() 
            pref = conf_samp.copy()
            pref[pref != 0] = 1 # Create binarized preference vector
            conf_samp = conf_samp + 1
            CiI = sparse.diags(conf_samp, [0]) # Get Ci - I term
            xTCiIX = X.T.dot(CiI).dot(X) # This is the xT(Cu-I)X term
            xTCiPi = X.T.dot(CiI + X_eye).dot(pref.T) # This is the xTCiPi term
            Y[i] = spsolve(xTx + xTCiIX + lambda_eye, xTCiPi)
            # Solve for Yi = ((xTx + xT(Cu-I)X) + lambda*I)^-1)xTCiPi
    # End iterations
    return X, Y.T 


# Loading the implicit library
import implicit

alpha = 15
user_vecs, item_vecs = implicit.alternating_least_squares((product_train*alpha).astype('double'), 
                                                           factors=20, 
                                                           regularization = 0.1, 
                                                           iterations = 50)

# Evaluating the Recommendation Engine

# Loading metrics from the sklearn library
from sklearn import metrics

# Functon to calculate AUC
def auc_score(predictions, test):
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)   

# Using the function inside a second function that calculates the AUC for each Customer
def calc_mean_auc(training_set, altered_users, predictions, test_set):
        
    store_auc = [] # An empty list to store the AUC for each customer that had an item removed from the training set
    popularity_auc = [] # To store popular AUC scores
    pop_items = np.array(test_set.sum(axis = 0)).reshape(-1) # Get sum of item iteractions to find most popular
    item_vecs = predictions[1]
    for user in altered_users: # Iterate through each customer that had an item altered
        training_row = training_set[user,:].toarray().reshape(-1) # Get the training set row
        zero_inds = np.where(training_row == 0) # Find where the interaction had not yet occurred
        # Get the predicted values based on the customer/item vectors
        user_vec = predictions[0][user,:]
        pred = user_vec.dot(item_vecs).toarray()[0,zero_inds].reshape(-1)
        # Get only the items that were originally zero
        # Select all ratings from the prediction for this customer that originally had no iteraction
        actual = test_set[user,:].toarray()[0,zero_inds].reshape(-1) 
        # Select the binarized yes/no interaction pairs from the original full data
        # that align with the same pairs in training 
        pop = pop_items[zero_inds] # Get the item popularity for our chosen items
        store_auc.append(auc_score(pred, actual)) # Calculate AUC for the given user and store
        popularity_auc.append(auc_score(pop, actual))
    # End users iteration
    
    # Return the mean AUC rounded to two decimal places
    return float('%.2f'%np.mean(store_auc)) 
   

# AUC for the recommendation system
print '\nAUC for the recommendation system:', calc_mean_auc(product_train, product_users_altered, 
              [sparse.csr_matrix(user_vecs), sparse.csr_matrix(item_vecs.T)], product_test), '\n'

# Recommendation to a particualr Customer
customers_arr = np.array(customers) # Array of customer IDs from the ratings matrix
products_arr = np.array(items) # Array of product IDs from the ratings matrix


# Creating a function that will return a list of the item descriptions from the earlier created item lookup table.
def get_items_purchased(customer_id, mf_train, customers_list, items_list, item_lookup):
    
    # This just tells me which items have been already purchased by a specific user in the training set. 
    
    # parameters: 
    
    # customer_id - Input the customer's id number that you want to see prior purchases of at least once
    
    # mf_train - The initial ratings training set used (without weights applied)
    
    # customers_list - The array of customers used in the ratings matrix
    
    # items_list - The array of products used in the ratings matrix
    
    # item_lookup - A simple pandas dataframe of the unique product ID/product descriptions available
    
    # returns:
    
    # A list of item IDs and item descriptions for a particular customer that were already purchased in the training set
    
    cust_ind = np.where(customers_list == customer_id)[0][0] # Returns the index row of the Customer Number
    purchased_ind = mf_train[cust_ind,:].nonzero()[1] # Get column indices of purchased items
    prod_codes = items_list[purchased_ind] # Get the item numbers for the purchased items
    prod_codes = prod_codes.astype(str)
    return item_lookup.loc[item_lookup['Item Number'].isin(prod_codes)]


# Looking at the list of customers to get some Customer Numbers:
print '\nFirst few Customer Numbers:\n', customers_arr[:5], '\n'

# Loading MinMaxScaler from the sklearn library
from sklearn.preprocessing import MinMaxScaler

# Creating the recommendation function
def rec_items(customer_id, mf_train, user_vecs, item_vecs, customer_list, item_list, item_lookup, num_items = 10):
    
    # This function will return the top recommended items to the users 
    
    # parameters:
    
    # customer_id - Input the customer's id number that you want to get recommendations for
    
    # mf_train - The training matrix you used for matrix factorization fitting
    
    # user_vecs - the user vectors from the fitted matrix factorization
    
    # item_vecs - the item vectors from the fitted matrix factorization
    
    # customer_list - an array of the customer's ID numbers that make up the rows of the ratings matrix 
    #                (in order of matrix)
    
    # item_list - an array of the products that make up the columns of the ratings matrix
    #                (in order of matrix)
    
    # item_lookup - A simple pandas dataframe of the unique product ID/product descriptions available
    
    # num_items - The number of items you want to recommend in order of best recommendations. Default is 10. 
    
    # returns:
    
    # - The top n recommendations chosen based on the user/item vectors for items never interacted with/purchased
    
    
    cust_ind = np.where(customer_list == customer_id)[0][0] # Returns the index row of our customer id
    pref_vec = mf_train[cust_ind,:].toarray() # Get the ratings from the training set ratings matrix
    pref_vec = pref_vec.reshape(-1) + 1 # Add 1 to everything, so that items not purchased yet become equal to 1
    pref_vec[pref_vec > 1] = 0 # Make everything already purchased zero
    rec_vector = user_vecs[cust_ind,:].dot(item_vecs.T) # Get dot item of user vector and all item vectors
    # Scale this recommendation vector between 0 and 1
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0] 
    recommend_vector = pref_vec*rec_vector_scaled 
    # Items already purchased have their recommendation multiplied by zero
    product_idx = np.argsort(recommend_vector)[::-1][:num_items] # Sort the indices of the items into order 
                                                                 # of best recommendations
    rec_list = [] # start empty list to store items
    for index in product_idx:
        code = item_list[index]
        # Append the descriptions to the list
        rec_list.append(code) 
        
    return list(rec_list)

# Looking at purchases from Customer Number 101931
print '\nPurchases from Customer Number: 101931:\n', get_items_purchased(101931, product_train, customers_arr,
                                                                         products_arr, item_lookup), '\n'

# Retrieving the N highest ranking dot items between the Customer and item vectors for a particular Customer.
# Items already purchased are not recommended to the Customer.
print '\nTen recommended items for Customer Number 101931:\n', rec_items(101931, product_train, user_vecs, item_vecs, customers_arr,
                                                                products_arr, item_lookup, num_items = 10), '\n'

# Preparing recommendations output to csv
cust_str = customers_arr # list(customers_arr) # np.array(customers)

for i in xrange(len(cust_str)): # xrange(cust_str.size)
    cust_num = cust_str[i]
    dfRecommend = pd.DataFrame(rec_items(cust_num, product_train, user_vecs, item_vecs, customers_arr,
                                                                products_arr, item_lookup, num_items = 10))
    
    dfRecommend.columns = ['Recommendations']

    dfCustNum = pd.DataFrame({'CustNum' : [cust_num]})
    dfCustNum = pd.concat([dfCustNum]*10, ignore_index=True)       
    dfConcat = pd.concat([dfCustNum, dfRecommend], axis=1)

    if(i == 0):
        dfFinalOutput = dfConcat
    else:
        dfFinalOutput = dfFinalOutput.append(dfConcat)

print '\ndfFinalOutput[0;5]:', dfFinalOutput[0:5], '\n'
# Export Recommendations to CSV
dfFinalOutput.to_csv("Recommendations.csv", index = False)
