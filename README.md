# Introduction to Machine Learning Coursework 2

by Hudson Yeo, Ling Yu Choi, Monika Jotautaite and Grzegorz Sarapata

## Running the code

Please download all files and place them in the same folder. 

## Part 1

### Linear Layer

To initialise a single linear layer, run the commands

```
layer = LinearLayer(n_in=10, n_out=4)
```

where n_in and n_out specify the input and output sizes of the layer.
The LinearLayer class is a building block for multi-layerd networks.

### Activation Functions

To implement a specfific activation function run one of the lines
```
relu = ReluLayer()
sigmoid = SigmoidLayer()
identity = Identity()
```
These are designed to work along the LinearLayer class when constructing a multi-layerd network.

### Multi Layered network

To initialise a network composed of stacked Linear Layers and various Activation Functions run the code
```
net = MultiLayerNetwork(input_dim=4, neurons=[16, 3], activations=["relu", "identity"])
```
where input_dim specifies the size of the input, neurons is a list containing consecutive layers' sizes
and activations indicates which activation functions should be used. The length of neurons will determine
the number of linear layers in the network, and the last element of the list corresponds to the size of the
network's output.

To obtain network's prediction for input X run
```
net.forward(X)
```
To perform backward pass of a gradient
```
net.backward(grad)
```
To update parameters of the network's layers using currently stored gradients
```
net.update_params(learning_rate)
```
where learning_rate specifies the magnitude of gradient descent

### Preprocessor

A data set can be scaled using the Preprocessor class initialised by
```
preprocessor = Preprocessor(data, scale=[0, 1])
```
where data is the the data set we wish to process and scale indicates the range of values to which
we want to scale the data

Scaling and rescaling of the data can be done using the methods
```
preprocessor.apply(data)
preprocessor.revert(data)
```

### Trainer

Trainer class is used to trained a given network. To initilialise an instance run
```
trainer = Trainer(
    network=net,
    batch_size=20,
    nb_epoch=50,
    learning_rate=0.01,
    loss_fun="cross_entropy",
    shuffle_flag=True,
    )
```
where network specifies the network to train,
batch_size is the size of a training batch,
nb_epoch is the number of training epochs,
learning_rate specifies the magnitude of gradient descent steps,
loss_fun indicates which loss function is to be used,
shuffle_flag indicates whether the data is to be shuffled when the network is trained.

Trainer class methods:
for given input and target datasets the network can be trained using 
```
trainer.train(input_dataset, target_dataset)
```
and a loss metric can be calculated using
```
trainer.eval_loss(input_dataset, target_dataset)
```


## Part 2

First, please specify the output label and dataset path (splitting it into x and y).
```
output_label = "median_house_value"
data = pd.read_csv("housing.csv")
x_train = data.loc[:, data.columns != output_label].iloc[:, :]
y_train = data.loc[:, [output_label]]
```

### Preprocessor

To initialise the regressor, run the following

```
regressor = Regressor(x_train, hidden=[50,70], nb_epoch=10)
```

where x_train is the dataset you would like to train, with a list of hidden layers sizes (the example above [50,70] refers to a first hidden layer with 50 neurons and a second hidden layer with 70 neurons), and 10 epochs.

### Fit

To fit the regessor, run the following: 

```
regressor.fit(x_train, y_train)
```

### Score

To calculate the error of the regressor, run the following:
```
error = regressor.score(x_train, y_train)
```
This gives all the error metrics, including RMSE, residual mean and variance, and R squared.
If you would like to plot the histogram of residuals, use plot_hist=True in the function.

### Save and load median_house_value

To save the trained regressor:
```
save_regressor(regressor)
```

You can load the saved model, using:
```
load_regressor()
```

### Hyperparameter search

To utilise the hyperparameter search for hidden layers, please run the following.
The default is to search amongst 3 values 50, 100, 150 neurons across both 1 and 2 hidden layers.
```
results_list, output_params = RegressorHyperParameterSearch(data, output_label)
```
The results_list is a list of all the combinations of hyperparameters that were tested and their associated RMSE values. The output_params is a dictionary which contains the parameters of the best model with the lowest RMSE.

To run the bigger hyperparameter search using the GPU (as detailed in our report), please visit the link below:

https://colab.research.google.com/drive/1Wwk2QsKyAn8x6N4huBQmvalVNJJql4oW?usp=sharing

### RMSE Plots from Hyperparameter Search

To plot the RMSE heatmaps and graphs, run the following:

```
show_loss_graphs()
```

Please note this will require the file result.json, in the directory. The result.json file we used for our graphs in the report has been uploaded in the git repository. Do note that due to the file result.json being created on Google Colab, library version differences may occur and lead to some errors depending on the machine you are using. Regardless, the result.json file we have uploaded should work without any issues.


