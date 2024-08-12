import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

# loading the dataframe with moore's law data

df = pd.read_csv("data/moores_law_data.csv")

# retrieving the years and transistors per microprocessor info 

def get_data():
    global x
    global y
    x = np.arange(39)
    y = df["Transistors per microprocessor"]

    y_log = np.log(y)

    y_log = y_log.to_numpy()
    
    x = torch.from_numpy(x).type(torch.float64).unsqueeze(dim=1)
    y_log = torch.from_numpy(y_log).type(torch.float64)

    return x, y_log 

# making train/test division:

def train_test_div(x, y):
    div = int(0.8 * len(x))
    x_train, y_train, x_test, y_test = x[:div], y[:div], x[div:], y[div:]
    return x_train, y_train, x_test, y_test

# data visualization 

def data_vis(x_train, y_train, x_test, y_test, pred = None):
    plt.scatter(x_train, y_train, c = "b", label = "Train data", s = 10)
    plt.scatter(x_test, y_test, c = "r", label = "Test data", s = 10)

    if pred is not None:
        plt.scatter(x_test, pred, c = "g", label = "Prediction", s = 10)

    plt.xlabel("Year (0 - 1971 and 38 - 2021)")
    plt.ylabel("Transistors per microprocessor [log]")
    plt.legend()
    plt.show()

# creating linear regression model: 
# Y = A*X + B
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.angular_coef = nn.Parameter(torch.rand(1, dtype = torch.float), requires_grad = True)
        self.linear_coef = nn.Parameter(torch.rand(1, dtype = torch.float), requires_grad = True)

    def __str__(self):
        return f"{self.state_dict()}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.angular_coef * x + self.linear_coef)
    
# training the model: 
# plotting the loss curve for testing as well
def model_train_test(model, x_train, y_train, x_test, y_test):
    # defining a loss and an optimizer: 

    # MAE
    loss_function = nn.L1Loss() 

    # SGD
    optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.01, momentum = 0.9)

    epochs = 150

    # empty loss lists to track values
    train_loss_values = []
    test_loss_values = []
    # list to count the epochs
    epoch_count = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_pred = model(x_train)
        loss = loss_function(train_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Testing
        model.eval()

        with torch.inference_mode(): 
            test_pred = model(x_test)
            loss_test = loss_function(test_pred, y_test)

            # print out what's happening after each epoch
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(loss_test.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {loss_test} ")


    # print the loss curves:
    plt.plot(epoch_count, train_loss_values, label = "Train loss curve")
    plt.plot(epoch_count, test_loss_values, label = "Test loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()    

def inference(model, x_train, y_train, x_test, y_test):
    model.eval()

    with torch.inference_mode():
        y_pred = model(x_test)

    data_vis(x_train, y_train, x_test, y_test, y_pred)

def exp_plot(model):
    params = list(model.parameters())
    y_pred_linear = params[0] * x + params[1]
    y_pred_exp = np.exp(y_pred_linear.detach().numpy())

    plt.plot(x, y_pred_exp, c = "b", label = "Prediction")
    plt.plot(x, y, c = "r", label = "True label")
    plt.xlabel("Year (0 - 1971 and 38 - 2021)")
    plt.ylabel("Transistors per microprocessor")
    plt.legend()
    plt.show()

def main():
    torch.manual_seed(0)

    # get the data from the csv file
    x, y_log = get_data()
    # train/test split
    x_train, y_train, x_test, y_test = train_test_div(x, y_log)
    # data vis
    data_vis(x_train, y_train, x_test, y_test)
    # model instantiation
    model = LinearRegressionModel()
    # model training and testing loop
    model_train_test(model, x_train, y_train, x_test, y_test)
    # making predictions
    inference(model, x_train, y_train, x_test, y_test)
    # printing final parameters values
    print(model)
    # plotting the exp relation
    exp_plot(model)

main()


