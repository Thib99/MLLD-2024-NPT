from sklearn.neighbors import KNeighborsClassifier


def Quadratic() :
    pass


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
    