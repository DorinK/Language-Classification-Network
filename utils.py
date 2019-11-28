from collections import Counter
from torch import zeros


# Reading the data from the requested file.
def read_data(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        data = []
        for line in file:
            label, text = line.strip().lower().split("\t", 1)
            data.append((label, text))
    return data


# Splitting the data into bigrams.
def text_to_bigrams(text):
    return ["%s%s" % (c1, c2) for c1, c2 in zip(text, text[1:])]


# Splitting the data into unigrams.
def text_to_unigrams(text):
    return ["%s" % c1 for c1 in text]


# Replacing the labels of all examples in the data set into the respective ID of each and the features into vector.
def update_dataset(dataset):
    for index in range(len(dataset)):
        label = labels.index(dataset[index][0], 0, len(labels)) if dataset[index][0] in labels else dataset[index][0]
        feats = dataset[index][1]
        features_vec = zeros(len(features))
        for feature in feats:
            if feature in features:
                features_vec[features.index(feature, 0, len(features))] += 1
        dataset[index] = (label, features_vec)
    return dataset


# Loading the training set data according to the requested representation of the features and pulling out the common
# features on the training set.
def load_train_set(representation):
    train = [(l, text_to_bigrams(t)) for l, t in read_data("train")] if representation == 'bigrams' else [
        (l, text_to_unigrams(t)) for l, t in read_data("train")]
    num_desired_features = 700 if representation == 'bigrams' else 90

    fc = Counter()
    for l, feats in train:
        fc.update(feats)

    # 700 most common bigrams/unigrams(following representation) in the training set.
    vocab = set([x for x, c in fc.most_common(num_desired_features)])

    # Label strings to IDs.
    L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in train]))))}

    # Feature strings to IDs.
    F2I = {f: i for i, f in enumerate(list(sorted(vocab)))}

    global labels, features
    features = list(F2I.keys())
    labels = list(L2I.keys())

    # Replacing the labels of all examples in the data set into the respective ID of each.
    new_train = update_dataset(train)

    return new_train, len(features), len(labels)


# Loading the validation set data according to the requested representation of the features.
def load_validation_set(representation):
    dev = [(l, text_to_bigrams(t)) for l, t in read_data("dev")] if representation == 'bigrams' else [
        (l, text_to_unigrams(t)) for l, t in read_data("dev")]
    return update_dataset(dev)


# Loading the test set data.
def load_test_set():
    test = [(l, text_to_bigrams(t)) for l, t in read_data("test")]
    return update_dataset(test)


# Replacing the ID of the label with the respective label(=language).
def index_to_language(pred):
    language = labels[pred]
    return language
