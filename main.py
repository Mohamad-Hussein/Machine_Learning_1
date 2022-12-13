from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier


def main():
    # my data contains 10 people from ENGG and NON ENGG 
    #E means someone from ENGG NE means NON ENGG 
    #avg GPA is also included corresponding to the person being in ENGG or not
    #IQ level is also included
    #Age is also given
    #Degree year is given
    data = DataFrame()
    data['Department']=['E','NE','E','NE','E','NE','E','E','NE','E','NE','E','E','NE','E']
    data['IQ']=[110, 81 , 105 , 98 , 129 , 91 , 116 , 120 , 87 , 158,91 , 116 , 120 , 87 , 158]
    data['GPA']=[2.5, 2.8, 3.7, 2.9, 3.5, 2.3, 2.5, 2.6, 3.1, 3.0,2.3, 2.5, 2.6, 3.1, 3.0]
    data['Age']=[23,19,20,26,23,26,46,32,46,21,26,46,32,46,21]
    data['Degree year']=['First','Third','Second','First','Second','Third','Fourth','First','Second','Third','Third','Fourth','First','Second','Third']

    data_attribute= data.drop(columns=['Department','Degree year'])
    data_classes = data.Department

    # decided to turn shuffle to false to not make it random
    # to keep it consistent
    # use stratify=data_classes to split the 
    # train and test arrays to have the same number of 
    # each class equal, ex: there are 5 engineers and 5 non engineers, 10 instances
    # stratify enables the test set to have 2 eng and 2 non eng
    # and train to have 3 eng and 3 non eng     
    data_attribute_train, data_attribute_test, data_classes_train, data_classes_test = train_test_split(data_attribute,data_classes, shuffle=False)

    #preprocessing
    le = preprocessing.LabelEncoder()
    le = le.fit(data_classes_train)
    class_train = le.transform(data_classes_train)
    data_attribute_train.to_numpy()

    n1 = preprocessing.MinMaxScaler()
    n1 = n1.fit(data_attribute_train.to_numpy())
    attribute_train= n1.transform(data_attribute_train.to_numpy())
    
    class_test = le.transform(data_classes_test.to_numpy())
    attribute_test = n1.transform(data_attribute_test.to_numpy())
    
    # training the data
    knn = KNeighborsClassifier(n_neighbors=11)
    knn = knn.fit(attribute_train,class_train)

    # getting the best k, first with minmax scaler
    training= []
    testing= []
    k_values = list(range(1,7))
    for i in k_values:
        knn =KNeighborsClassifier(n_neighbors=i)
        knn = knn.fit(attribute_train,class_train)
        training.append(knn.score(attribute_train,class_train))
        testing.append(knn.score(attribute_test,class_test))
    
    plt.figure(figsize=(25,10))
    plt.scatter(k_values,training,c='r',label='Training Dataset')
    plt.scatter(k_values,testing,c='g',label='Testing Dataset')
    plt.legend(loc=3)
    plt.xlabel("K-Values")
    plt.ylabel("Accuracy")
    # plt.show()
    plt.close()

    # k using standard scaler
    n11 = preprocessing.StandardScaler()
    n11 = n11.fit(data_attribute_train.to_numpy())
    attribute_train2= n1.transform(data_attribute_train.to_numpy())

    training2=[]
    for j in k_values:
        knn = KNeighborsClassifier(n_neighbors=j)
        knn = knn.fit(attribute_train2,class_train)
        training2.append(knn.score(attribute_train2,class_train))

    plt.figure(figsize=(25,10))
    plt.scatter(k_values,training2,c='y',label='Training Dataset')
    plt.scatter(k_values,testing,c='m',label='Testing Dataset')
    plt.legend(loc=3)
    plt.xlabel("K-Values")
    plt.ylabel("Accuracy")
    # plt.show()
    plt.close()

    # found k as 1 and training model
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn = knn.fit(attribute_train2,class_train)
    prediction_array = knn.predict(attribute_test)

    # confusion matrix
    confusion_matrix_diagram = confusion_matrix(class_test, prediction_array, labels = [0,1])
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_diagram, display_labels = ['In Engg', 'Not in Engg']).plot()
    plt.show()
    plt.close()

    #classifying a new instance
    new_instance = np.array([  [145, 4.0, 25]  ]) 
    X_new = n1.transform(new_instance)

    prediction = knn.predict(X_new)

    prediction = le.inverse_transform(prediction)

    # assigning result
    if prediction[0] == 'E': # E (engineer) which is defined as the class 0
        result = 'might be in Engineering'
    else:
        result = 'might not be in Engineering'
    
    print("The machine learning model predicted that the student {}".format(result))

    return 0

if __name__ == '__main__':
    main()