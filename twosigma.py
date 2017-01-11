import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from genom import Genom, gen_make
from entity import Observer, Predictor, Entity
from const import *


class LogRegressPredictor(Predictor):
    """
    Predictor wth logistic regression

    """
    def __init__(self, genom=None):
        super().__init__(genom)
        #self.classifier = self.create_classifier()

    def _create_classifier(self):
        lreg = self.genom.genom['logreg_l_reg'].values[0]
        c = self.genom.genom['logreg_c'].values[0]
        solver = self.genom.genom['logreg_solver'].values[0]
        if solver != 'liblinear':
            lreg = 'l2'
        return LogisticRegression(penalty=lreg, C=c, solver=solver)
        
    def generate_genom(self):
        return Genom([gen_make('logreg_l_reg', 'sampled', ['l1','l2'], size=1), 
                      gen_make('logreg_c', 'sampled', [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000], size=1),
                      gen_make('logreg_solver', 'sampled', ['newton-cg', 'lbfgs', 'liblinear', 'sag'], size=1)
                     ])


class TwoSigmaObserver(Observer):
    """
    Features selector from pandas\numpy feature array

    """
    def __init__(self, genom=None,feature_count=FEATURE_COUNT):
        self.size = feature_count
        super().__init__(genom)
        self.indexs = np.nonzero(self.get_genom().genom['ts_features'].values)[0]
        # if no one features select random one
        if len(self.indexs) == 0:
            rnd = np.random.randint(self.size)
            self.indexs = [rnd]
            self.get_genom().genom['ts_features']._data[rnd] = 1
            
    def generate_genom(self):
        return Genom([gen_make('ts_features', 'bernuli', size=self.size, p=0.05)])

    def process(self, observation):
        """
        Select only feature with 1 in bernuli gen ts_features
        """
        if type(observation) == pd.DataFrame:
            return observation.iloc[:,self.indexs]
        if type(observation) == np.ndarray:
            return observation[:,self.indexs]
        raise ValueError("Not supported type {}".format(type(observation)))

class TSEntity(Entity):
    observer_class = TwoSigmaObserver
    predictor_class = LogRegressPredictor

    def __init__(self, genom=None):
        super().__init__(genom)


def test():
    print("read file")
    with pd.HDFStore("../input/train.h5", "r") as train:
        # Note that the "train" dataframe is the only dataframe in the file
        df = train.get("train")
    print("fill NA")
    df = df.fillna(0)
    e1 = TSEntity()   
    e2 = TSEntity()   
    print(e1.get_genom())
    print(e2.get_genom())
    e1.fit(df.iloc[:1000, :], df.iloc[:1000, :]['y'] > 0)
    e1.action(df.iloc[1000:2000, :])
    e3 = e1.rep(e2)
    print(e3.get_genom())
    e3.fit(df.iloc[:1000, :], df.iloc[:1000, :]['y'] > 0)
    e3.action(df.iloc[1000:2000, :])
    
if __name__=="__main__":
    test()