import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = nn.Linear(28 * 28, 64)
        self.activation_input = nn.ReLU()
        self.dense_layer1 = nn.Linear(64, 64)
        self.activation1 = nn.ReLU()
        self.output_layer = nn.Linear(64, 10)
        self.activation_output = nn.Softmax(dim=0)
        self.double()

    def forward(self, x):
        x = self.activation_input(self.input_layer(x))
        x = self.activation1(self.dense_layer1(x))
        x = self.activation_output(self.output_layer(x))
        return x


def get_predictions(output):
    return np.argmax(output, 1)


def get_accuracy(predictions, output):
    return np.sum(predictions == output) / output.size * 100


def print_prediction(output, ground_truth):
    output = output.detach().numpy()
    ground_truth = ground_truth.detach().numpy()
    predictions = get_predictions(output)
    accuracy = get_accuracy(predictions, ground_truth)
    print("Predictions: ", predictions.T)
    print("Accuracy: ", accuracy, "%")
    print()


if __name__ == '__main__':
    # Read test data
    test_data = pd.read_csv(".\data_folder\mnist_test.csv").to_numpy()
    X_test = torch.from_numpy(test_data[:, 1:] / 255).to(float)
    y_test = torch.from_numpy(test_data[:, 0])

    load_model = True

    # Load model
    if load_model:
        model = Network()
        model.load_state_dict(torch.load(".\saved_models\TorchMNIST_MaxEpochs"))
        model.eval()

        # Print test accuracy
        print("Test accuracy:")
        print_prediction(model(X_test), y_test)

    else:   # Train and save model

        # Read data
        train_data = pd.read_csv(".\data_folder\mnist_train.csv").to_numpy()
        X = torch.from_numpy(train_data[:, 1:] / 255).to(float)
        y = torch.from_numpy(train_data[:, 0])
        one_hot_y = F.one_hot(torch.from_numpy(train_data[:, 0])).to(float)

        # Create model
        model = Network()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), 0.0005)

        # Train model
        n_epochs = 100
        batch_size = 32

        for epoch in range(n_epochs):

            # Save test prediction accuracy (used for preventing over fitting on the training data)
            predictions_before = get_predictions(model(X_test).detach().numpy())
            accuracy_before = get_accuracy(predictions_before, y_test.detach().numpy())

            # Print prediction accuracy
            print("Epoch: ", epoch)
            print_prediction(model(X), y)

            # Train
            for i in range(0, len(train_data), batch_size):
                # Extract a batch
                X_batch = X[i:i + batch_size]
                y_batch = one_hot_y[i:i + batch_size]

                # Train on batch
                y_predicted = model(X_batch)
                loss = loss_function(y_predicted, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Stop training when model starts to over fit training data
            predictions_after = get_predictions(model(X_test).detach().numpy())
            accuracy_after = get_accuracy(predictions_after, y_test.detach().numpy())
            print(accuracy_before, accuracy_after)
            if accuracy_before > accuracy_after:
                break

        # Print final prediction accuracy
        print("Epoch: ", n_epochs)
        print_prediction(model(X), y)

        # Print final test accuracy
        print("Test accuracy:")
        print_prediction(model(X_test), y_test)

        # Save model
        torch.save(model.state_dict(), ".\saved_models\TorchMNIST_MaxEpochs")

"""

# ↑↑↑ REMOVE COMMENT TO USE ↑↑↑
# GUI for testing MNIST model by drawing the numbers yourself (written by ChatGPT)


from tkinter import *
import tkinter as tk
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Create the main window
root = Tk()
root.title("Draw a digit")

# Create a canvas to draw on
canvas_width = 280
canvas_height = 280
canvas = Canvas(root, width=canvas_width, height=canvas_height, bg='black')
canvas.pack()

# Create a PIL image to draw on
img = Image.new('L', (canvas_width, canvas_height), 0)
draw = ImageDraw.Draw(img)


# Define the event functions
def paint(event):
    # Get the x,y coordinates of the mouse
    x, y = event.x, event.y

    # Draw a circle with radius r
    r = 10
    canvas.create_oval(x - r, y - r, x + r, y + r, fill='white', outline='')

    # Draw on the PIL image
    draw.ellipse((x - r, y - r, x + r, y + r), fill=255)


def clear_canvas():
    # Clear the canvas
    canvas.delete('all')

    # Clear the PIL image
    draw.rectangle((0, 0, canvas_width, canvas_height), fill=0)


def predict_digit():
    # Resize the image to 28x28 pixels
    img_resized = img.resize((28, 28))

    # Display the image using imshow
    #plt.imshow(img_resized)
    #plt.show()

    # Convert the image to a numpy array
    img_array = np.array(img_resized)

    # Normalize the pixel values
    img_array = img_array / 255

    # Flatten the image array
    img_flat = img_array.flatten()

    # Make the prediction using your trained neural network model
    output = model(torch.from_numpy(img_flat))
    predicted_digit = np.argmax(output.detach().numpy())

    # Show the predicted digit
    print("Prediction", "Predicted digit: {}".format(predicted_digit))


# Add buttons to clear the canvas and predict the digit
clear_button = Button(root, text="Clear", command=clear_canvas)
clear_button.pack(side=LEFT, padx=10)

predict_button = Button(root, text="Predict", command=predict_digit)
predict_button.pack(side=RIGHT, padx=10)

# Bind the mouse events
canvas.bind('<B1-Motion>', paint)

# Run the main loop
root.mainloop()

"""
