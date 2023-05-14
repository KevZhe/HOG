from NearestNeighbor import KNearestNeighbor
from HOG import HOG
import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--knn", type=int, default=3, help="k value for kNN")
#parser.add_argument("-m", "--method", type=str, default="histogram_intersection", help="method for computing distance")
parser.add_argument("-d", "--dataset", type=str, help="path to dataset to use")

args = vars(parser.parse_args())

def main():
    # Crawl through dataset and extract features using HOG
    # Keep track of filenames and labels

    # Training set
    train_filenames = []
    train_labels = []
    for filename in glob.glob(args["dataset"] + "/DB_Images_Neg/*.bmp"):
        train_filenames.append(filename)
        train_labels.append(0)
    for filename in glob.glob(args["dataset"] + "/DB_Images_Pos/*.bmp"):
        train_filenames.append(filename)
        train_labels.append(1)

    # Testing set
    test_filenames = []
    test_labels = []
    for filename in glob.glob(args["dataset"] + "/Test_Images_Neg/*.bmp"):
        test_filenames.append(filename)
        test_labels.append(0)
    for filename in glob.glob(args["dataset"] + "/Test_Images_Pos/*.bmp"):
        test_filenames.append(filename)
        test_labels.append(1)
    
    # Extract features from training set
    train_features = []
    for filename in train_filenames:
        train_features.append(HOG(filename))
    
    # Extract features from testing set
    test_features = []
    for filename in test_filenames:
        test_features.append(HOG(filename))

    # Train kNN classifier
    knn = KNearestNeighbor(k=args["knn"])
    knn.fit(np.array(train_features), np.array(train_labels))

    # Test kNN classifier
    histogram_predictions = []
    hellinger_predictions = []
    for feature in test_features:
        histogram_predictions.append(knn.predict(feature, method="histogram_intersection"))
        hellinger_predictions.append(knn.predict(feature, method="hellinger"))

    # Print results
    print('===========HISTOGRAM INTERSECTION===========')
    for ind, prediction in enumerate(histogram_predictions):
        pred, idxs, dists = prediction
        print(f'histogram_intersection for: {test_filenames[ind]}')
        print(f'prediction: {pred}')
        for idx, dist in zip(idxs, dists):
            print(f'idx: {train_filenames[idx]}, dist: {dist}')
    
    print('===========HELLINGER===========')
    for ind, prediction in enumerate(hellinger_predictions):
        pred, idxs, dists = prediction
        print(f'hellinger for: {test_filenames[ind]}')
        print(f'prediction: {pred}')
        for idx, dist in zip(idxs, dists):
            print(f'idx: {train_filenames[idx]}, dist: {dist}')
    
    # Get accuracies
    histogram_correct = 0
    for prediction, label in zip(histogram_predictions, test_labels):
        if prediction[0] == label:
            histogram_correct += 1
    histogram_accuracy = histogram_correct / len(test_labels)
    
    hellinger_correct = 0
    for prediction, label in zip(hellinger_predictions, test_labels):
        if prediction[0] == label:
            hellinger_correct += 1
    hellinger_accuracy = hellinger_correct / len(test_labels)

    print('===========ACCURACIES===========')
    print(f'histogram_intersection accuracy: {histogram_accuracy}')
    print(f'hellinger accuracy: {hellinger_accuracy}')
        
if __name__ == "__main__":
    main()