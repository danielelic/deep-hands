# deep-hands

This project is about hands classification.

First of all you need to run `data.py` scirt to generate the `*.npy` files.
After this, you can use one of the following nets:

* mnist (`train-mnist.py`)
* vgg16 (`train-vgg16.py`)
* vgg19 (`train-vgg19.py`)
* alexnet (`train-alexnet.py`)

# annotation-toolbox

```sh
python annotation-toolbox.py -images dataset/hands1000 [-data data.csv]
```