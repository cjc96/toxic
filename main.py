from data_loader import *

if __name__ == '__main__':
    train = load_train_data('data/train.csv')
    test = load_test_data('data/test.csv', 'data/test_labels.csv')
