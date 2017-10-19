# deep-hands

The aim of this project is to classify the type of shelf-interaction.

First of all you need to run `data.py` scirt to generate the `*.npy` files.
After this, you can use one of the following nets:

* CNN (`train-cnn.py`)
* CNN2 (`train-cnn2.py`)
* AlexNet (`train-alexnet.py`)
* CaffeNet (`train-caffenet.py`)

```
python data.py -images dataset/hands/ [-data data.csv]

python train-cnn.py
python train-cnn2.py
python train-alexnet.py
python train-caffenet.py
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

## Author
* Daniele Liciotti | [GitHub](https://github.com/danielelic)