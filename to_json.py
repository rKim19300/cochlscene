import json
import os
import librosa
import math
import numpy as np

SAMPLE_RATE = 44100
DURATION = 10 # 10 seconds
SAMPLES_PER_FILE = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=4096, hop_length=4096, num_segments=1):

    # Dict to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    # Loop through all classes
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # Ensure we are not at root
        if dirpath is not dataset_path:

            # Save semantic label
            dirpath_components = dirpath.split("/")[-1].split("\\")
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print(f"\nProcessing {semantic_label}")

            num_samples_per_segement = int(SAMPLES_PER_FILE / num_segments)
            expected_num_mfcc_vectors_per_segement = math.ceil(num_samples_per_segement / hop_length)

            # Process files for a specific class
            for f in filenames:

                # Load audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # Process segments extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segement * s
                    finish_sample = start_sample + num_samples_per_segement

                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                                 sr=sr,
                                                 n_fft=n_fft,
                                                 n_mfcc=n_mfcc,
                                                 hop_length=hop_length)
                    mfcc = mfcc.T

                    # Store mfcc for segment if it has the expected length
                    if len(mfcc) == expected_num_mfcc_vectors_per_segement:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print(f"{file_path}, segment:{s}")

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


def load_data(train_path, val_path, test_path):
    """
    Loads training set data from json files

        :param train_path (str): path to json file containing train data
        :param val_path (str): path to json file containing validation data
        :param test_path (str): path to json file containing test data

        :return X_train (ndarray): Inputs
        :return y_train (ndarray): Targets
        :return X_val (ndarray): Inputs
        :return y_val (ndarray): Targets
        :return X_test (ndarray): Inputs
        :return y_test (ndarray): Targets
    """

    # Get the train data
    with open(train_path, "r") as fp:
        data = json.load(fp)

    X_train = np.array(data["mfcc"])
    y_train = np.array(data["labels"])

    # Get the validation data
    with open(val_path, "r") as fp:
        data = json.load(fp)

    X_val = np.array(data["mfcc"])
    y_val = np.array(data["labels"])

    # Get the test data
    with open(test_path, "r") as fp:
        data = json.load(fp)

    X_test = np.array(data["mfcc"])
    y_test = np.array(data["labels"])

    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    train_path = "CochlScene/CochlScene/Train"
    val_path = "CochlScene/CochlScene/Val"
    test_path = "CochlScene/CochlScene/Test"
    train_path_json = "json_data/train.json"
    val_path_json = "json_data/val.json"
    test_path_json = "json_data/test.json"

    # Convert Train to json
    # save_mfcc(train_path, train_path_json)

    # Convert Val to json
    #save_mfcc(val_path, val_path_json)

    # Convert Test to json
    #save_mfcc(test_path, test_path_json)



if __name__ == "__main__":
    main()