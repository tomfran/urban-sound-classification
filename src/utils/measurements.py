import numpy as np
from ..data import Dataset

def show_accuracy_loss(net, scaling="scaled", test_dataset_path="../data/processed/extended"):
    loss = []
    accuracy = []

    for fold in [5, 7, 8, 9, 10]:
        td = Dataset(dataset_path=f"{test_dataset_path}/test_{fold}_{scaling}.csv", test_size=0)
        x_test, y_test = td.get_splits()
        results = net.model.evaluate(x_test, y_test, batch_size=128)
        loss.append(results[0])
        accuracy.append(results[1])

    print("\nAccuracy:")
    print(f"\tMean: {np.mean(accuracy)} \n\tStandard deviation: {np.std(accuracy)}")

    print("\nLoss:")
    print(f"\tMean: {np.mean(loss)} \n\tStandard deviation: {np.std(loss)}")
