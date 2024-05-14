from random import randint
import time
import numpy as np
from sklearn.calibration import cross_val_predict
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, make_scorer, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import cross_val_predict
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_validate
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy


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
    
    


def K_means(input_data, nbr_of_rep=5) : 
    
    # input_data = [X, y]
    X, y = input_data
    
   
    data = {
        'f1' : [], # f1 score
        'fn' : [] # False Negative rate
    }     
    
     
    # create the classifier
    for nbr in range(0,nbr_of_rep):
        
        classifier = KMeans(n_clusters=2)
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


def getData_allModels_Holdout(X, Y, template_data):
    
    # Initialize data within the dictionary
    for key in template_data.keys():
        template_data[key]['f1'] = []
        template_data[key]['f1_weighted'] = []        
        template_data[key]['false_negatif'] = []
        template_data[key]['accuracy'] = [] 
        template_data[key]['time_fit'] = 0
        template_data[key]['time_predict'] = 0
        template_data[key]['fpr']= []
        template_data[key]['tpr']= [] 

        
    random_seeds = [914, 895, 365, 264, 59, 500, 129]  # List of random seeds
    
    # Loop for iterating over different random seeds
    for int_state in random_seeds:

        # Splitting the data into train and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=int_state, stratify=Y)

        # Loop for iterating over the models
        for name in template_data.keys(): 
            reg = template_data[name]['model']
            start_time = time.time()          # Start time for fitting
            reg = reg.fit(X_train, Y_train)   # Fitting the model
            end_fit = time.time()             # End time for fitting
            result = reg.predict(X_test)      # Predictions
            end_predict = time.time()         # End time for prediction
            
            # Check if the model has predict_proba method
            if hasattr(reg, 'predict_proba'):
                probs = reg.predict_proba(X_test) # Probabilities
            else:
                probs = None
                
            
            # Calculating metrics and times
            template_data[name]['f1'].append(f1_score(Y_test, result))    # F1 score 
            template_data[name]['f1_weighted'].append(f1_score(Y_test, result, average='weighted'))  # F1 score weighted
            template_data[name]['false_negatif'].append(confusion_matrix(Y_test, result).ravel()[2])    # False negatives
            template_data[name]['accuracy'].append(balanced_accuracy_score(Y_test, result))             # Balanced accuracy score
            template_data[name]['time_fit'] += end_fit - start_time                                     # Accumulated fitting time
            template_data[name]['time_predict'] += end_predict - end_fit
            # Calculate ROC curve for each fold and store the values if probabilities available
            if hasattr(reg, 'predict_proba'):
                fpr, tpr, thresholds = roc_curve(Y_test, probs[:, 1])
                # Ensure that fpr and tpr have the same length
                template_data[name]['tpr'].append(tpr)
                template_data[name]['fpr'].append(fpr)
            else:
                template_data[name]['tpr'] = None
                template_data[name]['fpr'] = None

    # Calculate mean metrics and times
    for key in template_data.keys():
        template_data[key]['f1'] = np.mean(template_data[key]['f1']).round(2)
        template_data[key]['f1_weighted'] = np.mean(template_data[key]['f1_weighted']).round(2)
        template_data[key]['false_negatif'] = np.mean(template_data[key]['false_negatif']).round(2)
        template_data[key]['accuracy'] = np.mean(template_data[key]['accuracy']).round(2)
        template_data[key]['time_fit'] = np.mean(template_data[key]['time_fit'])
        template_data[key]['time_predict'] = np.mean(template_data[key]['time_predict'])
        # Calculate mean ROC curve if probabilities available
        if template_data[name].get('tpr') is not None:
            template_data[name]['tpr'] = np.mean(template_data[key]['tpr'], axis=0)
            template_data[name]['fpr'] = np.mean(template_data[key]['fpr'], axis=0)
        else:
            template_data[name]['tpr'] = None
            template_data[name]['fpr'] = None
        
    return template_data




def getData_allModels_CrossValidation(X, Y , template_data) :
    # Initialize data within the dictionary
    for key in template_data.keys():
        template_data[key]['f1'] = []
        template_data[key]['f1_weighted'] = []
        template_data[key]['false_negatif'] = []
        

    #Loop for iterating over the models
    for name in template_data.keys(): 
        reg = template_data[name]['model']
        scoring = {'f1_weighted': 'f1_weighted', 'false_negatives': fn_scorer, 'f1': 'f1'}
        
        result = cross_validate(reg,  X, Y, cv=5, scoring=scoring)

        
        # Calculating metrics and times
        template_data[name]['f1'].append(result['test_f1_weighted'])      # F1 score
        template_data[name]['false_negatif'].append(result['test_false_negatives'])    # False negatives
            

    # get only mean of the data for easy plotting
    for key in template_data.keys():
        template_data[key]['f1'] = np.mean(template_data[key]['f1']).round(2)
        template_data[key]['f1_weighted'] = np.mean(template_data[key]['f1_weighted']).round(2)
        template_data[key]['false_negatif'] = (np.mean(template_data[key]['false_negatif'])).round(2) 
    
    return template_data

def cnn(dense_activation_size, X_train, y_train):
    # Initialize a Sequential model
    model = Sequential()

    # Add Convolutional and Pooling layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))   # Add Conv2D layer with 32 filters, relu activation function, and input shape
    model.add(MaxPooling2D((2, 2)))                                                # Add MaxPooling2D layer with pool size (2,2)
    model.add(Conv2D(32, (3, 3), activation='relu'))                                # Add another Conv2D layer with 32 filters and relu activation function
    model.add(MaxPooling2D((2, 2)))                                                # Add another MaxPooling2D layer with pool size (2,2)

    # Flatten the output from Convolutional layers
    model.add(Flatten())                            # Flatten the output from Convolutional and Pooling layers                                           

    # Add Dense layers
    model.add(Dense(dense_activation_size, activation='relu'))         # Add Dense layer with specified number of neurons and relu activation function
    model.add(Dense(1, activation='sigmoid'))      # Add Dense layer with 1 neuron and sigmoid activation function

    # Compile the model
    model.compile(loss=BinaryCrossentropy(), optimizer=Adam(),  metrics=['accuracy', Precision(), Recall()])

    # Train the model
    history = model.fit(X_train, y_train, epochs=15, batch_size=64, class_weight={0: 155/253, 1: 98/253}, verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=3, min_delta=0.001)])

    return model, history