import os
import sys
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from joblib import dump
from pitch_class_profiling import PitchClassProfiler
from preprocess_text import chunk_wav

DATA_PATH = "data.json"
TEST_PATH = "test.json"


def save_pitch(dataset_path):
    """Extracts pitch class from music dataset and saves them into a json file along with genre labels.
    :param dataset_path (str): Path to dataset
    :param json_path (str): Path to json file used to save pitchs
    :return:
    """

    # dictionary to store mapping, labels, and pitch classes
    data = {"mapping": [], "labels": [], "pitch": [], "order": []}

    # loop through all chord sub-folder
    # for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
    dirpath = f"chunks/{dataset_path}"
    filenames = os.listdir(dirpath)

    print(f"dirpath: {dirpath}\nfilenames: {filenames}")
    semantic_label = dirpath.split("/")[-1]
    print(f"semantic_label: {semantic_label}")

    # save chord label (i.e., sub-folder name) in the mapping
    semantic_label = dirpath.split("/")[-1]
    data["mapping"].append(semantic_label)
    print("\nProcessing: {}".format(semantic_label))

    # process all audio files in chord sub-dir
    i = 0
    for f in filenames:

        # load audio file
        file_path = os.path.join(dirpath, f)
        file_name = file_path.split("/")[-1]
        file_name2 = file_name.split(".")[0]
        data["order"].append(file_name2)
        # process all segments of audio file

        ptc = PitchClassProfiler(file_path)
        data["pitch"].append(ptc.get_profile())
        data["labels"].append(i - 1)
        print("{}, segment:{}".format(file_path, 1))

    # Sorting the dictionary
    n = len(data["order"])
    for i in range(n - 1):

        # Last i elements are already in place
        for j in range(0, n - i - 1):

            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if int(data["order"][j].split("_")[1]) > int(
                data["order"][j + 1].split("_")[1]
            ):
                data["order"][j], data["order"][j + 1] = (
                    data["order"][j + 1],
                    data["order"][j],
                )
                data["pitch"][j], data["pitch"][j + 1] = (
                    data["pitch"][j + 1],
                    data["pitch"][j],
                )
                data["labels"][j], data["labels"][j + 1] = (
                    data["labels"][j + 1],
                    data["labels"][j],
                )

    # save pitch classes to json file
    if os.path.exists(TEST_PATH):
        os.remove(TEST_PATH)
        
    with open(TEST_PATH, "w") as fp:
        json.dump(data, fp, indent=4)


def load_data(data_path):
    """Loads training dataset from json file.
    :param data_path (str): Path to json file containing data
    :return X (ndarray): Inputs
    :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["pitch"])
    y = np.array(data["labels"])
    z = np.array(data["mapping"])
    return X, y, z


def train_ui():
    song_name = "temp.wav"
    chunk_wav(song_name)
    save_pitch(song_name[:-4])

    X, y, z = load_data(DATA_PATH)

    model_svc_lin = SVC(kernel="linear")
    model_svc_lin.fit(X, y)

    X_test, y_test, z_test = load_data(TEST_PATH)

    y_pred_svm_lin = model_svc_lin.predict(X_test)

    print("\nSVM linear: ")

    for i in range(len(X_test)):
        print(z[y_pred_svm_lin[i]], end=" ")

    print("\n")

    return z[y_pred_svm_lin]


def train(song_name):
    chunk_wav(song_name)
    save_pitch(song_name[:-4])

    X, y, z = load_data(DATA_PATH)

    model_knn = KNeighborsClassifier(n_neighbors=3)
    model_ada = AdaBoostClassifier(n_estimators=200, learning_rate=2)
    model_dt = DecisionTreeClassifier()
    model_svc_lin = SVC(kernel="linear")
    model_svc_rbf = SVC(kernel="rbf")

    model_knn.fit(X, y)
    model_ada.fit(X, y)
    model_dt.fit(X, y)
    model_svc_lin.fit(X, y)
    model_svc_rbf.fit(X, y)

    X_test, y_test, z_test = load_data(TEST_PATH)

    y_pred_knn = model_knn.predict(X_test)
    y_pred_ada = model_ada.predict(X_test)
    y_pred_dt = model_dt.predict(X_test)
    y_pred_svm_lin = model_svc_lin.predict(X_test)
    y_pred_svm_rbf = model_svc_rbf.predict(X_test)

    print("KNN: ")
    for i in range(len(X_test)):
        print(z[y_pred_knn[i]], end=" ")

    print("\nAdaboost: ")
    for i in range(len(X_test)):
        print(z[y_pred_ada[i]], end=" ")

    print("\nDecision tree: ")
    for i in range(len(X_test)):
        print(z[y_pred_dt[i]], end=" ")

    print("\nSVM rbf: ")
    for i in range(len(X_test)):
        print(z[y_pred_svm_rbf[i]], end=" ")

    print("\nSVM linear: ")
    for i in range(len(X_test)):
        print(z[y_pred_svm_lin[i]], end=" ")

    print("\n")

    return z[y_pred_svm_lin]


def main():
    try:
        song_name = sys.argv[-1]
        song_name = song_name.split("/")[-1]
        train(song_name)
    except:
        print("Please provide a correct song name")


if __name__ == "__main__":
    main()
