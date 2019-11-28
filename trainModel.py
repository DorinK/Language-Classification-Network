import torch.utils.data as data
import torch.nn.functional as F
from utils import load_train_set, load_validation_set, load_test_set, index_to_language


def train(model, optimizer, train_loader):

    # Declaring training mode.
    model.train()

    sum_loss = 0.0
    for batch_idx, (y, x) in enumerate(train_loader):

        # Reset the gradients from the previous iteration.
        optimizer.zero_grad()

        # Calculating the model's prediction to the current example.
        output = model(x)

        # Calculating the negative log likelihood loss.
        loss = F.nll_loss(output.reshape(1, 6), y)
        sum_loss += loss.item()

        # Back propagation.
        loss.backward()

        # Updating.
        optimizer.step()

    train_loss = sum_loss / len(train_loader)
    train_accuracy = accuracy_on_dataset(model, train_loader)
    return train_loss, train_accuracy


def accuracy_on_dataset(model, dataset):

    good = bad = 0.0
    for batch_idx, (y, x) in enumerate(dataset):

        # Calculating the model's prediction to the current example
        output = model(x)

        # Get the index of the max log-probability.
        prediction = output.data.max(1)[1]

        # Check for each example, if the model's prediction matches the label.
        if prediction == y:
            good += 1
        else:
            bad += 1

    # Calculating the accuracy rate on the validation set.
    return good / (good + bad)


def validation_accuracy(model, valid_loader):

    # Declaring evaluation mode.
    model.eval()

    return accuracy_on_dataset(model, valid_loader)


def test_predictions(model, test_loader):
    import os

    # Clearing the content of the file if it already exists; Otherwise, creating the file.
    if os.path.exists("./test.pred"):
        os.remove("./test.pred")
    f = open("./test.pred", "a+")

    # For each example we find calculate the model prediction
    for batch_idx, (y, x) in enumerate(test_loader):

        # Calculating the model's prediction to the current example
        output = model(x)

        # Get the index of the max log-probability
        prediction = output.data.max(1)[1]

        # Write to the file
        f.write("{0}\n".format(index_to_language(prediction.item())))

    # Close the file.
    f.close()


def train_model(model, optimizer):

    # Loading the data sets of each section respectively.
    data_set, num_features, num_classes = load_train_set('bigrams')
    valid_set = load_validation_set('bigrams')
    test_set = load_test_set()

    # Creating the torch loaders for each section respectively.
    train_loader = data.DataLoader(data_set, batch_size=1, shuffle=True, num_workers=20, pin_memory=True)
    valid_loader = data.DataLoader(valid_set, batch_size=1, shuffle=True, num_workers=20, pin_memory=True)
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=20, pin_memory=True)

    # After each training of the model we check how much the model has learned, on the validation set.
    for epoch in range(1, 10 + 1):
        train_loss, train_acc = train(model, optimizer, train_loader)
        valid_acc = validation_accuracy(model, valid_loader)
        print(epoch, train_loss, train_acc, valid_acc)

    # Calculating the predictions of the model to the test examples and writing each one to the file.
    test_predictions(model, test_loader)
