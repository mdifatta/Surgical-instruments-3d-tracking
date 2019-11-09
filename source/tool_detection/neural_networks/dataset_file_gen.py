import pandas as pd
from sklearn.utils import shuffle


def main():
    # read entire csv
    df = pd.read_csv('../../../data/targets/targets.csv', sep=';')
    # shuffle rows
    df = shuffle(df)
    # count rows
    train_len = int(.9 * len(df.index))

    valid_len = int(.1 * train_len)

    # split train and test set
    train, test = df[:train_len], df[train_len:]

    # split validation set
    valid, train = train[:valid_len], train[valid_len:]

    # create new csv files
    train.to_csv('../../../data/targets/train.csv', sep=';', index=False)
    test.to_csv('../../../data/targets/test.csv', sep=';', index=False)
    valid.to_csv('../../../data/targets/valid.csv', sep=';', index=False)

    # dataset's balance
    print('Training set: %d samples VALID, %d samples INVALID' %
          (train[train.valid == 1]['valid'].count(), train[train.valid == 0]['valid'].count()))

    print('Test set: %d samples VALID, %d samples INVALID' %
          (test[test.valid == 1]['valid'].count(), test[test.valid == 0]['valid'].count()))

    print('Validation set: %d samples VALID, %d samples INVALID' %
          (valid[valid.valid == 1]['valid'].count(), valid[valid.valid == 0]['valid'].count()))


if __name__ == '__main__':
    main()
