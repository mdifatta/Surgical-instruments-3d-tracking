import pandas as pd
from sklearn.utils import shuffle


def main():
    # TODO: change paths
    # read entire csv
    df = pd.read_csv('../../data/targets/prova.csv', sep=';')
    # shuffle rows
    df = shuffle(df)
    # count rows
    train_len = int((2 / 3) * len(df.index))

    # split train and test set
    train, test = df[:train_len], df[train_len:]

    # create new csv files
    train.to_csv('../../data/targets/prova_train.csv', sep=';', index=False)
    test.to_csv('../../data/targets/prova_test.csv', sep=';', index=False)


if __name__ == '__main__':
    main()
