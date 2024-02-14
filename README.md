## Vision Transformer

This is a PyTorch implementation of the Vision Transformer for the CIFAR-10 dataset. The implementation should be close 
to the [original paper](https://arxiv.org/abs/2010.11929), but some details, such as the location of dropout might differ.
The validation performance is around 76%.

The hyperparameters were chosen according to [this other implementation](https://github.com/omihub777/ViT-CIFAR). 
That implementation achieves better performance (around 90%), with the main possible reason being that it uses autoaugmentation 
and label smoothing. Neither of these techniques have been implemented here (yet). I plan on adding label smoothing in
the future.

The performance of this work is thus closer to [this implementation](https://github.com/kentaroy47/vision-transformers-cifar10), which achieves around 80% accuracy after 200 epochs.

The chosen hyperparameters can be seen as the default parameters for the parser in the `main.py` file. 
