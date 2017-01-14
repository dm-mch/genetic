import numpy as np
import pandas as pd
from genom import Genom
from const import FEATURE_COUNT

class Fenotype:
    """
    Abstract class
    Contain genoms and build some from it
    
    """

    def __init__(self, genom=None):
        self.genom = self.select_gens(genom) or self.generate_genom()

    def select_gens(self, genom):
        """
        Select from genom only gens specified for this fenotype

        """
        if genom is None:
            return None
        my_genom = Genom({k:v for k,v in genom.items() if k in self.possible_gens()})
        assert len(self.possible_gens()) == len(my_genom)
        #print("select_gens:", my_genom)
        return my_genom
    
    def generate_genom(self):
        """
        Generate random genom
        """
        raise NotImplementedError()
                
    def get_genom(self):
        return self.genom

    def possible_gens(self):
        """
        get list of possible gens

        """
        raise NotImplementedError()

    
class Predictor(Fenotype):
    """
    Train and Prediction(action) on preprocessed features
    """
    def __init__(self, genom=None):
        super().__init__(genom)
        self.classifier = self._create_classifier()
        
    def _create_classifier(self):
        """
        Return classifier that support method fit and predict
        """
        raise NotImplementedError()
        
    def fit(self, x, y):
        self.classifier.fit(x,y)
        
    def action(self, x):
        return self.classifier.predict(x)

class Observer(Fenotype):
    """
    Features preprocess and select
    """
    def process(self, observation):
        raise NotImplementedError()
    

class Entity(Fenotype):
    """
    Entity with genom

    """
    # should be redefined in subclass
    observer_class = Observer
    predictor_class = Predictor

    def __init__(self, genom=None, feature_count=FEATURE_COUNT):
        self.observer = self.observer_class(genom, feature_count=feature_count)
        self.predictor = self.predictor_class(genom)
        super().__init__(genom)
        
    def generate_genom(self):
        """
        Cocatinate together predctor and observer genom
        """
        genom = Genom([])
        genom.merge(self.observer.get_genom())
        genom.merge(self.predictor.get_genom())
        return genom

    def possible_gens(self):
        return self.observer.possible_gens() + self.predictor.possible_gens()
    
    def rep(self, other, eps = 0.01):
        """
        Create one new entity from two
        Crossingover and mutation
        
        """
        g1 = self.get_genom()
        g2 = other.get_genom()
        new_g = g1.cross(g2)
        new_g.mutation(eps)
        return self.__class__(new_g)

    def fit(self, x, y):
        """
        Process features(x) and fit on its
        """
        return self.predictor.fit(self.observer.process(x), y)
    
    def action(self, x):
        """
        Process features and make an action
        """
        return self.predictor.action(self.observer.process(x))
