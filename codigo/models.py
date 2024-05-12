from random import randint
import numpy as np
from sklearn.calibration import cross_val_predict
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier


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
    
    
    
    
    
def MLP() :
    pass
    
    
    
    

