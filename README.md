# CS230 Final Project Code Repository

The repository contains the code used in my [CS230 Project](http://evanmunro.ca/files/restaurantChoice.pdf) "Deep Learning for Restaurant Choice". The repository is made up of a Jupyter Notebook, 5 executable python files, and 1 python file containing classes written for this project.  

- prepareRestaurantData.ipynb is a Jupyter Notebook used to prepare the restaurant choice data and calculate summary statistics.
- testModel.py loads a saved neural network model and calculates test error and accuracy on the test set.
- test_rnn.py loads a saved recurrent neural network model and calculates test error and accuracy on the test set
- choice_nn_estimation.py estimates the basic neural network or the conditional logit model  
- choice_rnn_estimation.py estimates the RNN neural network (this model had poor results and is not fully debugged).
- choiceModels.py contains the PyTorch classes written for the project: RNN and Neural Network Dataset classes and Model architectures


#### Using Outside Data

The data used in the project is not public, so I have included some additional guidance on using outside datasets with the code. As written, the ChoiceDataset module is designed for the RestaurantChoice dataset and is not particularly portable: for a new dataset, it is best to write your own module that extends the PyTorch Dataset class and provides a `getitem` and a `len` method (see PyTorch [Documentation](https://pytorch.org/docs/stable/torchvision/datasets.html) or [this tutorial](https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel).

For a dataset with n_x covariates for each choice and n_y items that the user chooses between for each session in the dataset, `getitem` should return: 1. (n_y,n_x) tensor of covariates for each of the items for the choice session requested (rows corresponding to missing items are all zeros) 2. (n_y,1) tensor of binary outcomes for each of the items for the choice session (only one value is equal to one for the item selected by the user). 3. If using a neural network module with an embedding layer, then this returns an (n_y,2) tensor which helps in mapping user and item-specific embeddings to the choice session and preventing missing items from affecting embeddings during training. The first column is equal to the user ID for the choice session. The second column is equal to the item ids for each of the n_y items in the dataset. For rows corresponding to items that are missing in a choice session, assign a specific user and item ID, which has embedding fixed to zero in the embedding layer.  

`len` returns how many choice sessions there are in the dataset.

#### Tutorial Using TravelMode Dataset

The travelmode_tutorial Jupyter notebook contains a simple tutorial: it shows how to write a simple Dataset class and estimate the Multinomial Logit model on a toy dataset from the R package AER. 
