import time
from kNearestNeighbours import KNN
from nearestClassCentroid import NCC
from RBFNetwork import RBF


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
    methods=["kmeans", "random", "adaptive"]
    RBF(methods[0])



if __name__ == "__main__":
    main()
