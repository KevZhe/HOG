# Histogram of Gradients
Authors: Kevin Zheng, Kevin Lee

## Run
To install the required dependencies
```
pip install requirements.txt
```

`runKNN.py` has two arguments: `--knn` and `--dataset`. You must specify the number of neighbors "k" and the file path to the dataset

```
python runKNN.py --knn 3 -data "/path/to/data/"
```

E.g.,
```
python runKNN.py --knn 3 -data data
```