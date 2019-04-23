import pandas as pd
from sklearn.utils import shuffle


def main():
    # read entire csv
    df = pd.read_csv('../../data/targets/targets.csv', sep=';')
    # shuffle rows
    df = shuffle(df)
    # count rows
    train_len = int((2 / 3) * len(df.index))

    valid_len = int((3 / 4) * train_len)

    # split train and test set
    train, test = df[:train_len], df[train_len:]

    # split validation set
    train, valid = train[:valid_len], train[valid_len:]

    # create new csv files
    train.to_csv('../../data/targets/train.csv', sep=';', index=False)
    test.to_csv('../../data/targets/test.csv', sep=';', index=False)
    valid.to_csv('../../data/targets/valid.csv', sep=';', index=False)


if __name__ == '__main__':
    main()
