from random import randint
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def Quadratic() :
    pass

def KNN_range(input_data, array_neighbors, repeat_by_neighbors = 5, size_sample = 0.2) : 
    
    
    
    # test done with holdout 
    seeds = [randint(0, 1000) for i in range(repeat_by_neighbors)]
    data = {
        'f1' : [], # f1 score
        'fn' : [] # False Negative rate
    }      
        
    # create the classifier
    for neighbor in array_neighbors:
        
        classifier = KNeighborsClassifier(n_neighbors = neighbor)
        temp_data = [[], []]
        for j in range(repeat_by_neighbors):
            # split the data
            X_train, X_test, y_train, y_test = train_test_split(input_data[0], input_data[1], test_size=size_sample, random_state=seeds[j])
            
            # train the classifier
            classifier.fit(X_train, y_train)
    
            # get the result
            result = classifier.predict(X_test)
                
            # get f1 score
            temp_data[0].append(f1_score(y_test, result))
            
            # get the False Negative rate 
            temp_data[1].append(confusion_matrix(y_test, result).ravel()[2])
        
        
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
    
    
    
    
    
def MLP() :
    pass
    