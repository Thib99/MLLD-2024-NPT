from random import randint
import time
import numpy as np
from sklearn.calibration import cross_val_predict
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import cross_val_predict
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_validate



def Quadratic() :
    pass

def KNN_range(input_data, array_neighbors, repeat_by_neighbors = 1, size_sample = 0.2) : 
    
    
    # input_data = [X, y]
    X, y = input_data
 
    data = {
        'f1' : [], # f1 score
        'fn' : [] # False Negative rate
    }      
        
    # create the classifier
    for neighbor in array_neighbors:
          
        classifier = KNeighborsClassifier(n_neighbors = neighbor)
        temp_data = [[], []]
        for j in range(repeat_by_neighbors):
            
            # metrics we want 
            scoring = ['f1_weighted']
            
            # get the result
            result = cross_validate(classifier,  X, y, cv=5, scoring=scoring)
            
        
            # Calculate the confusion matrix 
            predicted = cross_val_predict(classifier, X, y, cv=5)

            F_N = confusion_matrix(y, predicted).ravel()[2].mean()
            
            # print(result)
            temp_data[0].append(result['test_f1_weighted'].mean())
            temp_data[1].append(F_N)
            
            
        data['f1'].append(np.mean(temp_data[0]))
        data['fn'].append(np.mean(temp_data[1]))
    
    
    return data
    
def KNN(n_neighbors=5, validation="holdout") : 
    
    # create the classifier
    classifier = KNeighborsClassifier(n_neighbors= n_neighbors)
    
    
    # what to use to validate the model
    if validation == "holdout" :
        pass
    elif validation == "crossvalidation" :
        pass
    else :
        raise ValueError("validation must be 'holdout' or 'kfold'")
    
    


def K_means(input_data, nbr_of_clusters) : 
    
    # input_data = [X, y]
    X, y = input_data
    
    random_state = 42
 
    data = {
        'f1' : [], # f1 score
        'fn' : [] # False Negative rate
    }     
    
     
    # create the classifier
    for nbr in nbr_of_clusters:
        
        classifier = KMeans(n_clusters=nbr, random_state=random_state)
        # metrics we want 
        scoring = ['f1_weighted']
        
        # get the result
        result = cross_validate(classifier,  X, y, cv=5, scoring=scoring)
        
    
        # Calculate the confusion matrix 
        predicted = cross_val_predict(classifier, X, y, cv=5)

        F_N = confusion_matrix(y, predicted).ravel()[2]
        
        data['f1'].append(np.mean(result['test_f1_weighted']))
        data['fn'].append(np.mean(F_N))
        
    
    
    return data
    
def MLP() :
    pass
    
    
    

############################# Custom Scoring Function False Negatives ############################# 
# Define a custom scoring function to calculate false negatives

def false_negatives_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    false_negatives = cm[1, 0]  # False negatives are in the second row, first column of the confusion matrix
    return false_negatives


# Make the custom scoring function into a scorer object
fn_scorer = make_scorer(false_negatives_score, greater_is_better=True)


######################################### END  ######################################################


def getData_allModels_Holdout(X, Y , template_data) : 
        
    # Initialize data within the dictionary
    for key in template_data.keys():
        template_data[key]['f1'] = []
        template_data[key]['false_negatif'] = []
        template_data[key]['accuracy'] = [] 
        template_data[key]['time_fit'] = 0
        template_data[key]['time_predict'] = 0
        

    random_seeds = [914, 895, 365, 264, 59, 500, 129]  # List of random seeds
    # Loop for iterating over different random seeds
    for int_state in random_seeds:


        # Splitting the data into train and test sets
        # Make the sample with houldout
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=int_state)

        
        #Loop for iterating over the models
        for name in template_data.keys(): 
            reg = template_data[name]['model']
            start_time = time.time()          # Start time for fitting
            reg = reg.fit(X_train, Y_train)   # Fitting the model
            end_fit = time.time()             # End time for fitting
            result = reg.predict(X_test)      # Predictions
            end_predict = time.time()         # End time for prediction
            
            # Calculating metrics and times
            template_data[name]['f1'].append(f1_score(Y_test, result, average='weighted'))                                  # F1 score
            template_data[name]['false_negatif'].append(confusion_matrix(Y_test, result).ravel()[2])    # False negatives
            template_data[name]['accuracy'].append(balanced_accuracy_score(Y_test, result))             # Balanced accuracy score
            template_data[name]['time_fit'] += end_fit - start_time                                     # Accumulated fitting time
            template_data[name]['time_predict'] += end_predict - end_fit
            

    # get only mean of the data for easy plotting
    for key in template_data.keys():
        template_data[key]['f1'] = np.mean(template_data[key]['f1']).round(2)
        template_data[key]['false_negatif'] = (np.mean(template_data[key]['false_negatif'])).round(2)
        template_data[key]['accuracy'] = np.mean(template_data[key]['accuracy']).round(2)
        template_data[key]['time_fit'] = np.mean(template_data[key]['time_fit'])
        template_data[key]['time_predict'] = np.mean(template_data[key]['time_predict'])
        
        
    return template_data





def getData_allModels_CrossValidation(X, Y , template_data) :
    # Initialize data within the dictionary
    for key in template_data.keys():
        template_data[key]['f1'] = []
        template_data[key]['false_negatif'] = []
        

    #Loop for iterating over the models
    for name in template_data.keys(): 
        reg = template_data[name]['model']
        scoring = {'f1_weighted': 'f1_weighted', 'false_negatives': fn_scorer}
        
        result = cross_validate(reg,  X, Y, cv=5, scoring=scoring)

        
        # Calculating metrics and times
        template_data[name]['f1'].append(result['test_f1_weighted'])      # F1 score
        template_data[name]['false_negatif'].append(result['test_false_negatives'])    # False negatives
            

    # get only mean of the data for easy plotting
    for key in template_data.keys():
        template_data[key]['f1'] = np.mean(template_data[key]['f1']).round(2)
        template_data[key]['false_negatif'] = (np.mean(template_data[key]['false_negatif'])).round(2) 
    
    return template_data