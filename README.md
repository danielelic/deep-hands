# deep-hands

The aim of this project is to classify the user-shelf interaction. Three classes are considered:

| Neutral | Positive | Negative |
|:-------:|:--------:|:--------:|
| ![neutral](images/type_neutral.jpg) | ![positive](images/type_positive.jpg) | ![negative](images/type_negative.jpg) |

In order to generate `*.npy` files, you can run `data.py` script.
After this, you can use one of the following nets:

* CNN (`train_cnn.py`)
* CNN2 (`train_cnn2.py`)
* AlexNet (`train_alexnet.py`)
* CaffeNet (`train_caffenet.py`)

```
python data.py -images dataset/hands/ [-data data.csv]

python train_cnn.py
python train_cnn2.py
python train_alexnet.py
python train_caffenet.py
```

# annotation-toolbox

This is a useful toolbox to annotate a set of images with different classes.

1. Positive (hand with product)
2. Negative (other)
3. Neutral (only hand)
4. Skip (bad image)

```
python annotation-toolbox.py -images dataset/hands/ [-data data.csv]
```

# Python Environment Setup

```bash
sudo apt-get install python3-pip python3-dev python-virtualenv # for Python 3.n
virtualenv -p python3 venv
. venv/bin/activate
```

The preceding command should change your prompt to the following:

```
(venv)$ 
```
Install TensorFlow in the active virtualenv environment:

```bash
pip3 install --upgrade tensorflow-gpu # for Python 3.n and GPU
```

Install the others library:

```bash
pip3 install --upgrade keras scikit-learn scikit-image h5py
```

## Author
* Daniele Liciotti | [GitHub](https://github.com/danielelic)