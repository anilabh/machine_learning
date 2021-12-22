import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
#%matplotlib inline
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score

import itertools

def KNN():
    df = pd.read_csv("/Users/anipandey/Documents/ML/coursera/teleCust1000t.csv")
    # take a look at the dataset
    df.head()
    #fidn number of categories
    df['custcat'].value_counts()
    df.hist(column='income', bins=50)
    #Lets define feature sets, X
    df.columns
    #To use scikit-learn library, we have to convert the Pandas data frame to a Numpy array
    X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
    X[0:5]
    #Data Standardization give data zero mean and unit variance, it is good practice, especially for algorithms such as KNN which is based on distance of cases
    X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
    X[0:5]
    
    y = df['custcat'].values
    y[0:5]
    
    #use scikit-learns test-train split to split the data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
    print ('Train set:', X_train.shape,  y_train.shape)
    print ('Test set:', X_test.shape,  y_test.shape)

    from sklearn.neighbors import KNeighborsClassifier
    #start witj K=4
    k = 4
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
    yhat = neigh.predict(X_test)
    yhat[0:5]

    from sklearn import metrics
    print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
    print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

    #We can calculate the accuracy of KNN for different Ks.
    Ks = 10
    mean_acc = np.zeros((Ks-1))
    std_acc = np.zeros((Ks-1))

    for n in range(1,Ks):
        
        #Train Model and Predict  
        neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
        yhat=neigh.predict(X_test)
        mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

        
        std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
    #Plot model accuracy for Different number of Neighbors
    plt.plot(range(1,Ks),mean_acc,'g')
    plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
    plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
    plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
    plt.ylabel('Accuracy ')
    plt.xlabel('Number of Neighbors (K)')
    plt.tight_layout()
    plt.show()
    print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 

def decisionTii():
    my_data = pd.read_csv("/Users/anipandey/Documents/ML/coursera/drug200.csv", delimiter=",")
    # take a look at the dataset
    my_data[0:5]
    
    print(my_data.count())
    print(my_data.shape)

    X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
    X[0:5]

    #pandas.get_dummies() Convert categorical variable into dummy/indicator variables
    from sklearn import preprocessing
    le_sex = preprocessing.LabelEncoder()
    le_sex.fit(['F','M'])
    X[:,1] = le_sex.transform(X[:,1]) 


    le_BP = preprocessing.LabelEncoder()
    le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
    X[:,2] = le_BP.transform(X[:,2])


    le_Chol = preprocessing.LabelEncoder()
    le_Chol.fit([ 'NORMAL', 'HIGH'])
    X[:,3] = le_Chol.transform(X[:,3]) 

    X[0:5]

    y = my_data["Drug"]
    y[0:5]

    from sklearn.model_selection import train_test_split
    X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

    print('Shape of X training set {}'.format(X_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape))
    print('Shape of X testing set {}'.format(X_testset.shape),'&',' Size of Y training set {}'.format(y_testset.shape))

    drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
    drugTree # it shows the default parameters
    drugTree.fit(X_trainset,y_trainset)

    #Let's make some predictions on the testing dataset and store it into a variable called predTree. 
    predTree = drugTree.predict(X_testset)
    print (predTree [0:5])
    print (y_testset [0:5])
    
    #Next, let's import metrics from sklearn and check the accuracy of our model. 
    from sklearn import metrics
    import matplotlib.pyplot as plt
    print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

    #Lets visualize the tree
    from  io import StringIO
    import pydotplus
    import matplotlib.image as mpimg
    from sklearn import tree
    #%matplotlib inline

    dot_data = StringIO()
    filename = "drugtree.png"
    featureNames = my_data.columns[0:5]
    out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png(filename)
    img = mpimg.imread(filename)
    plt.figure(figsize=(100, 200))
    plt.imshow(img,interpolation='nearest')
    plt.show()

#Another way of looking at accuracy of classifier is to look at confusion matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def logisticReg():
    churn_df = pd.read_csv("ChurnData.csv")
    churn_df.head()
    #Lets select some features for the modeling. Also we change the target data type to be integer, as it is a requirement by the skitlearn algorithm
    churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
    churn_df['churn'] = churn_df['churn'].astype('int')
    churn_df.head()
    #Lets define X, and y for our dataset
    X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
    X[0:5]
    y = np.asarray(churn_df['churn'])
    y [0:5]
    #Also, we normalize the dataset
    X = preprocessing.StandardScaler().fit(X).transform(X)
    X[0:5]
    #test train split
    
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
    print ('Train set:', X_train.shape,  y_train.shape)
    print ('Test set:', X_test.shape,  y_test.shape)

    #The version of Logistic Regression in Scikit-learn, support regularization. Regularization is a technique used to solve the overfitting problem in machine learning models. C parameter indicates inverse of regularization strength which must be a positive float. Smaller values specify stronger regularization.
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

    #Now we can predict using our test set
    yhat = LR.predict(X_test)
    #predict_proba returns estimates for all classes, ordered by the label of classes. So, the first column is the probability of class 0, P(Y=0|X), and second column is probability of class 1, P(Y=1|X):
    yhat_prob = LR.predict_proba(X_test)
    #Lets try jaccard index for accuracy evaluation. we can define jaccard as the size of the intersection divided by the size of the union of two label sets.
    #If the entire set of predicted labels for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.
    from sklearn.metrics import jaccard_score
    jaccard_score(y_test, yhat,pos_label=0)    

    print(confusion_matrix(y_test, yhat, labels=[1,0]))

     # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
    np.set_printoptions(precision=2)


    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
    plt.show()

def plot_confusion_matrix_svm(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def SVM():#support vector machine
    cell_df = pd.read_csv("/Users/anipandey/Documents/ML/coursera/cell_samples.csv")
    cell_df.head()
    #Lets look at the distribution of the classes based on Clump thickness and Uniformity of cell size
    ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
    cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
    plt.show()
    cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
    cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
    cell_df.dtypes
    feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
    X = np.asarray(feature_df)
    X[0:5]
    #We want the model to predict the value of Class (that is, benign (=2) or malignant (=4))
    cell_df['Class'] = cell_df['Class'].astype('int')
    y = np.asarray(cell_df['Class'])
    y [0:5]
    #we split our dataset into train and test set
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
    print ('Train set:', X_train.shape,  y_train.shape)
    print ('Test set:', X_test.shape,  y_test.shape)

    #mapping data into a higher dimensional space is called kernelling. types of kernelling functions:
    #1.Linear
    #2.Polynomial
    #3.Radial basis function (RBF)
    #4.Sigmoid
    #no easy way to know wich func will be best so we need to iterate
    #Let's just use the default, RBF
    from sklearn import svm
    clf = svm.SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    yhat = clf.predict(X_test)
    yhat [0:5]

        # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
    np.set_printoptions(precision=2)

    print (classification_report(y_test, yhat))

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix_svm(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')

    f1_score(y_test, yhat, average='weighted')
    jaccard_score(y_test, yhat,pos_label=2)
    
SVM()
#logisticReg()
#KNN()
#decisionTii()
