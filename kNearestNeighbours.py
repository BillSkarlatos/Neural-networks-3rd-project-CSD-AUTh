from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import DataHandling as dh
import time

# K Nearest Neighbours
def KNN(num):
    database=dh.loadDatabase()
    # Loading all data from database using load_data which takes 2 arguments: the database and a limitation of the dataset
    # should we want one for execution speed. If we want the whole dataset, we leave the limit as 0.
    input_data, input_labels, test_data, test_labels = dh.load_data(database, 0)

    # K nearest neighbours class that takes the desired number of neighbours as an argument.
    print(f"Fitting the data in the k-Nearest-Neighbours Classifier with k={num}")
    start_time=time.time()
    knn = KNeighborsClassifier(num)
    knn.fit(input_data, input_labels)
    total_time=time.time() - start_time
    minutes= total_time//60
    seconds= total_time - minutes*60
    prediction = knn.predict(test_data)
    accuracy = accuracy_score(test_labels, prediction)
    print(f"Classification complete in {int(minutes)} minutes, {seconds:.10f} seconds")
    print(f"Accuracy with K-Nearest Neighbours with k={num}: {accuracy * 100:.2f}%")
    return accuracy
