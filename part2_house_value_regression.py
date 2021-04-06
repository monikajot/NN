import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement as comb

class Net(nn.Module):

    def __init__(self, dim_0=2, hidden=[200], dim_output=1, dropout=0.5):
        super(Net, self).__init__()

        layers = []
        dims = np.concatenate(([dim_0], hidden))
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        self.linear_out = nn.Linear(dims[-1], dim_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.net(x)
        x = self.dropout(x)
        x = self.linear_out(x)
        return x


class Regressor():

    def __init__(self, x, nb_epoch=30, hidden=[200], lr=0.0025):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        self.scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.lr = lr
        self.model = Net(dim_0=self.input_size, hidden=hidden, dim_output=self.output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, amsgrad=True)
        self.loss = nn.MSELoss()

        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y=None, training=False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} -- Preprocessed input array of size
                (batch_size, input_size).
            - {torch.tensor} -- Preprocessed target array of size
                (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None

        # Binary encode
        if training:

            lb = LabelBinarizer()
            x = x.join(pd.DataFrame(lb.fit_transform(x["ocean_proximity"]),
                                    columns=lb.classes_,
                                    index=x.index))

            x.drop('ocean_proximity', axis=1, inplace=True)
            self.lb = lb

        else:
            x = x.join(pd.DataFrame(self.lb.transform(x["ocean_proximity"]),
                                    columns=self.lb.classes_,
                                    index=x.index))
            x.drop('ocean_proximity', axis=1, inplace=True)

        # fill NA
        for column in x.columns:
            if x.loc[:, column].isnull().sum() != 0:
                mu = x.loc[:, column].mean(skipna=True)
                x.loc[:, column].fillna(mu, inplace=True)

        # get min max scaling parameters from training data
        if training:
            self.scaler.fit(x)

        # apply scaling to x

        x = self.scaler.transform(x)
        if training:
            if y is not None:
                y = y.to_numpy()
                # find scale for labels in y
                if training:

                    y=self.y_scaler.fit_transform(y)
                else:
                    y=self.y_scaler.transform(y)

                y_tensor = torch.from_numpy(y).float()

        x_tensor = torch.from_numpy(x).float()

        return x_tensor, (y_tensor if y is not None else None)


    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################

    def fit(self, X, y, batch_size=50, split=0.7, threshold=5, printout=False):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        loss_train_list = []
        loss_val_list = []
        min_loss = np.inf

        X, y = self._preprocessor(X, y=y, training=True)  # Do not forget

        # split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=split, random_state=42)

        # get batches
        batch_indices = np.arange(len(X_train))
        np.random.seed(42)
        np.random.shuffle(batch_indices)
        batch_indices = batch_indices[:-(len(X_train) % batch_size)].reshape(-1, batch_size)

        for epoch in range(self.nb_epoch):
            running_loss = 0.0

            self.model.train()
            for ib, batch in enumerate(batch_indices):

                # sample batch from X,Y to get train_x,train_y
                # if someone could make this into a proper batch processing method, that'll be great
                X_batch = X_train[batch].clone().detach().float()
                y_batch = y_train[batch].clone().detach().float()

                self.optimizer.zero_grad()  # Zero Grad

                y_pred = self.model(X_batch)  # Forward through the model

                loss = self.loss(y_pred, y_batch)  # Calculate loss

                loss.backward()  # Backprop

                self.optimizer.step()  # step

                if printout:
                    running_loss += loss.item()

            if printout:
                loss_train_list.append(running_loss / batch_indices.shape[0])

            self.model.eval()

            y_pred_val = self.model(X_val)
            score = self.loss(y_pred_val, y_val)
            loss_val_list.append(score.item())

            #early stopping
            if score <= min_loss:
                # if yes then set to min_loss and set c=0
                min_loss = score
                c = 0
                # save best state
                best_state = self.model.state_dict()

            else:
                c += 1
            #if c > 1:
                # if loss has not gone down, slowly decrease learning rate
                #self.lr *= 0.95
                #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, amsgrad=True)
            if c == threshold:
                # stopping early if have threshold (default 3) consecutive epochs with worsening loss
                break

        if printout:
            fig, ax = plt.subplots()
            ax.set_title('Epoch Losses - Training vs Validation')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE')
            ax.set_xticks(np.arange(0, len(loss_val_list), 2))
            ax.set_xticklabels(np.arange(1, len(loss_val_list)+1, 2))
            ax.grid(True)
            ax.plot(np.arange(len(loss_val_list)), loss_train_list, label='Training Set', zorder=2)
            ax.plot(np.arange(len(loss_val_list)), loss_val_list, label='Validation Set', zorder=2)
            ax.scatter(x=np.argmin(loss_val_list), y=np.min(loss_val_list), color='r', zorder=3, label='Best validation loss')
            ax.legend(loc='upper right')
            plt.show()
        # load the best state
        self.model.load_state_dict(best_state)
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training=False)  # Do not forget
        pred = self.model(X.float()).detach().numpy()
        return self.y_scaler.inverse_transform(pred)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y, ret_residuals=False,plot_hist=False):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        pred = self.predict(x)
        residuals = pred - y.to_numpy()
        rmse = np.sqrt(np.mean(residuals**2))
        mean = np.mean(residuals)
        var = np.var(residuals)
        r2 = r2_score(y_true=y.to_numpy(), y_pred=pred)
        if plot_hist:
            plt.hist(residuals, 50)
            plt.title('Histogram of residuals')
            plt.xlabel('Residuals')
            plt.ylabel('Frequency')
            plt.text(200000,2000, f'\u03BC= {mean.round(0)},\n \u03C3={round(math.sqrt(var))}')
            plt.show()

        return {
            'rmse': round(rmse, 2),
            'residual_mean': round(mean, 2),
            'residual_var': round(var, 2),
            'R2': round(r2, 3),
            }, residuals if ret_residuals else None

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch(data, output_label, nb_epoch=5):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.

    Returns:
        The function should return your optimised hyper-parameters.

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    # set of params to iterate from
    params = {'hidden_layers': [1, 2],
              'neurons': [50, 100, 150]}

    # initialise our results dictionary
    output_params = {'neurons': None, 'error': np.inf}

    # split data into X and y
    X = data.loc[:, data.columns != output_label].iloc[:, :]
    y = data.loc[:, [output_label]]

    results_list = []

    # create a set of neuron combinations
    neurons_list = set(comb(params['neurons'], 2))
    neurons_list.update([neuron[::-1] for neuron in neurons_list])
    neurons_list = list(neurons_list)
    # add in single layer neurons
    for neuron in params['neurons']:
        neurons_list.append([neuron])

    for i in neurons_list:

        layers = list(i)
        results = cross_validation(data, layers, output_label, nb_epoch=nb_epoch)
        results_list.append([layers, results.rmse.mean()])

        # if cross validated error is smaller than the one in dictionary,
        # update dictionary with current model
        if results.rmse.mean() < output_params["error"]:
            output_params['neurons'] = layers
            output_params['error'] = results.rmse.mean()
            regressor = Regressor(X, nb_epoch=5, hidden=layers)
            regressor.fit(X, y)
            save_regressor(regressor)

    return results_list, output_params  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def cross_validation(data, layers, output_label, nb_epoch=10, k_folds=10, seed=42):

    X = data.loc[:, data.columns != output_label].iloc[:, :]
    y = data.loc[:, [output_label]]

    # initialise a k fold split
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    kfold.get_n_splits(X, y)

    # create dataframe with columns for each error measure
    results_df = pd.DataFrame(columns=['rmse', 'residual_mean', 'residual_var', 'R2'])

    for train_index, test_index in kfold.split(X, y):
        X_train_fold = X.iloc[train_index]
        y_train_fold = y.iloc[train_index]
        X_test_fold = X.iloc[test_index]
        y_test_fold = y.iloc[test_index]

        regressor = Regressor(X_train_fold, nb_epoch=nb_epoch, hidden=layers)
        regressor.fit(X_train_fold, y_train_fold)

        results = regressor.score(X_test_fold, y_test_fold)
        results_df = results_df.append(pd.DataFrame.from_dict(results[:-1]), ignore_index=True)

    return results_df

def example_main():
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")


    # Spliting input and output
    x_train = data.loc[:, data.columns != output_label].iloc[:, :]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset.
    # You probably want to separate some held-out data
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, hidden=[50,50], nb_epoch=10)
    regressor.fit(x_train, y_train)
    #save_regressor(regressor)

    # Error
    #regressor = load_regressor()
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))
    print(regressor.model)


def show_loss_graphs(path='result.json'):
    import json
    result=json.load(open(path,"r"))
    rmse1=[]
    rmse_neurons=[]
    rmse2=np.zeros([16,16])

    for r in result:
        if len(r[0])==1:
            rmse_neurons.append(r[0][0])
            rmse1.append(r[-1])
        else:
            hidden1=int((r[0][0]-30)/15)
            hidden2=int((r[0][1]-30)/15)
            rmse2[hidden1,hidden2]=r[-1]

    plt.figure(figsize=[12,10])
    plt.imshow(rmse2,cmap='viridis_r')
    plt.xticks(list(range(16)),list(range(30,270,15)))
    plt.yticks(list(range(16)),list(range(30,270,15)))
    plt.title('RMSE by hidden layer neurons, for 2 hidden layer networks')
    plt.xlabel('Neurons in second layer')
    plt.ylabel('Neurons in first layer')
    cbar=plt.colorbar()

    plt.savefig('2D.png')
    plt.show()

    plt.figure(figsize=[8,8])
    plt.title('RMSE by hidden layer neurons, for 1 hidden layer networks')
    plt.xlabel('Neurons in hidden layer')
    plt.ylabel('RMSE')
    plt.plot(rmse_neurons,rmse1)
    plt.savefig('1D.png')
    plt.show()



if __name__ == "__main__":
    example_main()
    # output_label = "median_house_value"
    # data = pd.read_csv("housing.csv")
    # results_list, output_params = RegressorHyperParameterSearch(data, output_label)
