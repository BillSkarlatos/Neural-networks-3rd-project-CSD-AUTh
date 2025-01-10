from kNearestNeighbours import KNN
from nearestClassCentroid import NCC
from RBFNetwork import RBF


def main():
    print(u'\u2501' * 35)
    # K-Nearest Neighbors
    print("Running K-Nearest Neighbors...")
    KNN(3) 
    print(u'\u2501' * 35)
    # Nearest Class Centroid
    print("Running Nearest Class Centroid...")
    NCC()
    print(u'\u2501' * 35)
    # Hebbian Learning
    print("Running Hebbian Learning...")
    methods=["kmeans", "random", "adaptive"]
    RBF(methods[2], hidden_neurons = 500, learning_rate = 0.002, batch_size = 128, epochs = 25)
    print(u'\u2501' * 35)


if __name__ == "__main__":
    main()
