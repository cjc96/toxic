from data_loader import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


class ToxicJudge(object):
    def __init__(self, data_path):
        train = load_train_data(data_path + '/train.csv', is_df=False)
        test = load_test_data(data_path + '/test.csv',
                              data_path + '/test_labels.csv', is_df=False)

        train_text, self.train_class = train[1], train[2]
        val_text, self.val_class = train[4], train[5]
        test_text, self.test_class = test[1], test[2]

        total = train_text + val_text
        self.word_generator = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            stop_words='english',
            ngram_range=(1, 1),
            max_features=1000)
        self.word_generator.fit(total)
        train_word_features = self.word_generator.transform(train_text)
        val_word_features = self.word_generator.transform(val_text)
        test_word_features = self.word_generator.transform(test_text)

        self.train_features = train_word_features
        self.val_features = val_word_features
        self.test_features = test_word_features

    def customInput(self, comment, name='rf'):
        input_s = self.word_generator.transform([comment])
        if name == 'rf':
            return self.rf.predict(input_s)
        elif name == 'dt':
            return self.rf.predict(input_s)
        else:
            return [lr.predict(input_s) for lr in self.lrs]

    def lr_process1(self):
        for col in range(6):
            train_target = self.train_class[:, col]
            val_target = self.val_class[:, col]
            valList = []
            trainList = []
            for C in [0.1, 0.3, 0.5, 1, 2]:
                lr = LogisticRegression(C=C, solver='sag')
                lr.fit(self.train_features, train_target)
                valList.append(lr.score(self.val_features, val_target))
                trainList.append(lr.score(self.train_features, train_target))
            plt.plot([0.1, 0.3, 0.5, 1, 2], valList, label='val')
            plt.plot([0.1, 0.3, 0.5, 1, 2], trainList, label='train')
            plt.legend()
            plt.show()
            print(valList)

    def lr_process2(self):
        self.lrs = []
        predictList = []
        Clist = [.5, .3, .3, .1, .5, .8]
        for col in range(6):
            train_target = self.train_class[:, col]
            lr = LogisticRegression(C=Clist[col], solver='sag')
            lr.fit(self.train_features, train_target)
            predictList.append(lr.predict(self.test_features))
            self.lrs.append(lr)
        predictList = np.asarray(predictList).T
        eq = 0
        for idx in range(predictList.shape[0]):
            if np.array_equal(predictList[idx, :], self.test_class[idx, :]):
                eq += 1
        return eq/predictList.shape[0]

    def dt_process(self):
        self.dt = DecisionTreeClassifier(
            max_depth=300, min_samples_split=10, min_samples_leaf=10, max_leaf_nodes=160)
        self.dt.fit(self.train_features, self.train_class)
        return self.dt.score(self.test_features, self.test_class)

    def rf_process(self):
        self.rf = RandomForestClassifier(n_estimators=10, max_depth=300,
                                         min_samples_split=10, min_samples_leaf=10, max_leaf_nodes=170)
        self.rf.fit(self.train_features, self.train_class)
        return self.rf.score(self.test_features, self.test_class)
