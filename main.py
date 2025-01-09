import time
from kNearestNeighbours import KNN
from nearestClassCentroid import NCC
from RBFNetwork import 
from DataHandling import loadDatabase, load_data


def main():

    # K-Nearest Neighbors
    print("Running K-Nearest Neighbors...")
    KNN(1)
    KNN(3) 

    # Nearest Class Centroid
    print("Running Nearest Class Centroid...")
    NCC()

    # Hebbian Learning
    print("Running Hebbian Learning...")
    HBN()



if __name__ == "__main__":
    main()
