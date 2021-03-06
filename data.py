import pandas as pd
import numpy as np

from const import DATA_PATH

class DataProvider:
    """ Abstract class. Provide data for train and validate one entity """

    def get_train(self, ):
        raise NotImplementedError()

    def get_validate(self):
        raise NotImplementedError()

    def get_test(self):
        raise NotImplementedError()
        
    def get_features_names(self):
        raise NotImplementedError()

    def get_features_count(self):
        raise NotImplementedError()

class RandomSeq:
    """ Generate random(shuffled) sequence and return it partly """
    def __init__(self, high_value):
        self.high_value = high_value
        self.rndseq = self.generate()

    def generate(self):
        a = np.arange(self.high_value)
        np.random.shuffle(a)
        return a

    def get(self, length):
        if len(self.rndseq) < length:
            self.rndseq = self.generate()
        r = self.rndseq[:length]
        self.rndseq = self.rndseq[length:]
        return r

class TwoSigmaDataProvider(DataProvider):
    """ Provide data from twosigma kaggle competion """

    def __init__(self, path=DATA_PATH, train=5000, validate=1000, test=1000):
        # size for one pack returned by get_train etc.
        self.train_size = train
        self.validate_size = validate
        self.test_size = test
        # read the data
        self.df = pd.read_hdf(path)
        # columns
        self.non_features = ['id', 'timestamp', 'y']
        self.feature_cols = [c for c in self.df.columns if c not in self.non_features]
        self.y_col = 'y'
        # data prepere
        self.df = self.fillnan(self.df)
        self.separate(self.df)
        
    def fillnan(self, df):
        self.df_mean = self.df.median(axis=0)
        return df.fillna(self.df_mean)

    def separate(self, df):
        """ separate df data set on train, validate, test """
        shffl = np.arange(df.shape[0])
        np.random.shuffle(shffl)
        size = int(df.shape[0]/3)
        print(df.shape[0], size)
        self.train = df.iloc[shffl[:size]].reset_index(drop=True)
        self.validate = df.iloc[shffl[size:2*size]].reset_index(drop=True)
        self.test = df.iloc[shffl[2*size:]].reset_index(drop=True)
        # Random sequences for get random subsets
        self.random_seq = {'train': RandomSeq(len(self.train)),
                           'validate': RandomSeq(len(self.validate)),
                           'test': RandomSeq(len(self.test))}
        
    def get_random(self, df, size, rnd_seq_name):
        rnd_indxs = self.random_seq[rnd_seq_name].get(size)
        res = self.train.iloc[rnd_indxs, :]
        return {'x': res[self.feature_cols], 'y': (res[self.y_col] > 0).astype(int)}
        
    def get_train(self):
        return self.get_random(self.train, self.train_size, 'train')

    def get_validate(self):
        return self.get_random(self.validate, self.validate_size, 'validate')

    def get_test(self):
        return self.get_random(self.test, self.test_size, 'test')
    
    def get_features_names(self):
        return self.feature_cols

    def get_features_count(self):
        return len(self.feature_cols)


        