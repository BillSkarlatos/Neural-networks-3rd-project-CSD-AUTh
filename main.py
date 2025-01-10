import time
from kNearestNeighbours import KNN
from nearestClassCentroid import NCC
from RBFNetwork import RBF


def main():
    print(u'\u2501' * 10)
    # K-Nearest Neighbors
    print("Running K-Nearest Neighbors...")
    KNN(3) 
    print(u'\u2501' * 10)
    # Nearest Class Centroid
    print("Running Nearest Class Centroid...")
    NCC()
    print(u'\u2501' * 10)
    # Hebbian Learning
    print("Running Hebbian Learning...")
    methods=["kmeans", "random", "adaptive"]
    RBF(methods[0])
    print(u'\u2501' * 10)


if __name__ == "__main__":
    main()
