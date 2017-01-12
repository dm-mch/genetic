import numpy as np
import pandas as pd
import copy

def gen_make(name, type, *argv, **argw):
    gen_types = {
        "normal": GenNormal,
        "uniform": GenUniform,
        "bernuli": GenBernuli,
        "sampled": GenSampled
    }
    if type not in gen_types.keys():
        raise Exception("Not supported type of gen {}".format(type))
    return gen_types[type](name, *argv, **argw)
    
class Gen:
    """
    Abstract class for gen
    Contain vector data of value of gen
    Different Gen classes for different distribution of value in data vector

    """
    def __init__(self, name, data=None, dtype=np.float32, range=None, size=None, **argw):
        # self.type = None # will define in subclass
        self.name = name
        self._dtype = dtype
        self._range = range
        self._size = size
        if data is None:
            assert self._size is not None
            self.data = self.generate()
        else:
            self.data = np.array(data, dtype=dtype)
            if range is not None:
                print("clip data range ", range)
                self.data = self.data.clip(range[0], range[1])
            if size is not None:
                assert size == len(data)
        self._size = len(self.data)
            
    def generate(self):
        """
        Generate initial data for gen
        """
        raise NotImplementedError("Should implemented in subclass")
        
    def mutation(self, eps = 0.01, inplace=True):
        e = np.random.rand(self._size)
        res = np.array(self.data, copy=True)
        for i in range(self._size):
            if e[i] <= eps:
                res[i] = self.mutvalue(self.data[i])
        if inplace:
            self.data = res
        return res
    
    def mutvalue(self, value):
        """
        Generate mutation for one value
        """        
        raise NotImplementedError("Should implemented in subclass")
        
    def copy(self):
        return copy.deepcopy(self)
    
    @property
    def values(self):
        return self.data
    
    def __str__(self):
        return "Gen: {self.name}, type: {self.type} size: {self._size}, {self.values}".format(self=self)

class GenNormal(Gen):
    """
    Gen of normal distrubuted values

    """
    def __init__(self, name, mean=0, std=0.1, dtype=np.float32, **argw):
        self.type = 'Normal'
        self._mean = mean
        self._std = std
        assert dtype == np.float32
        super().__init__(name, dtype=dtype,**argw)
        
    def generate(self):
        r = np.random.normal(self._mean, self._std, self._size)
        if self._range is not None:
            r = r.clip(self._range[0],self._range[1])
        return r
    
    def mutvalue(self, value):
        v = np.random.normal(value, self._std/10., 1)   
        if self._range is not None:
            v = v.clip(self._range[0],self._range[1])
        return v[0]

        
class GenUniform(Gen):
    """
    Gen of uniform distributed value from range

    """
    def __init__(self, name, **argw):
        self.type = 'Uniform'
        super().__init__(name, **argw)
        
    def generate(self):
        assert self._range is not None
        assert len(self._range) == 2
        if self._dtype in (np.int8, np.int16, np.int32, np.int64):
            r = np.random.randint(self._range[0],self._range[1] + 1, self._size)
        else:
            r = np.random.uniform(self._range[0], self._range[1], self._size)
        return r  
        
    def mutvalue(self, value):
        return np.random.normal(value, (self._range[1] - self._range[0])/10., 1).clip(self._range[0],self._range[1]).astype(self._dtype)[0]

        
class  GenBernuli(Gen):
    """
    Gen of 0 and 1 from bernuli distribution

    """
    def __init__(self, name, p=0.5, dtype=np.int8, **argw):
        self.type = 'Bernuli'
        self._p = p
        assert dtype == np.int8
        super().__init__(name, dtype=dtype,  range=(0,1), **argw)
        
    def generate(self):
        return np.random.binomial(1, self._p, self._size)

    def mutvalue(self, value):
        return np.random.binomial(1, self._p, 1)[0]    
    
class GenSampled(Gen):
    """
    Gen of samples (enumerated values from sample)
    """
    def __init__(self, name, samples, probs=None, dtype=np.int8, **argw):
        self.type = 'Sampled'
        self._samples = samples
        assert dtype == np.int8
        if probs is None:
            self._probs = [1./len(samples)] * len(samples)
        else:
            assert len(probs) == len(samples)
            self._probs = probs
        super().__init__(name, dtype=dtype, range=(0,len(samples)-1), **argw)
        
     
    def generate(self):
        return np.random.choice(np.arange(len(self._samples), dtype=np.int8), self._size, p=self._probs)
    
    def mutvalue(self, value):
        return np.random.choice(np.arange(len(self._samples), dtype=np.int8), 1, p=self._probs)[0]
    
    @property
    def values(self):
        return [self._samples[d] for d in self.data]

def cross(v1, v2):
    """
    Crossingover for vector v1 and v2
    return new vector
    """
    assert len(v1) == len(v2)
    l = len(v1)
    # Сколькими частями будем обмениваться
    parts = int(min(l, max(2,round(np.random.normal(l/2, l/3, 1)[0]))))
    # Будет ли обменяна часть
    cross = np.random.choice([1, 0], parts)
    # Вычисляем длины частей
    lengths = np.array([round(l/parts)] * parts)
    if lengths[0] * parts != l:
        diff = int(lengths[0] * parts - l)
        inc = np.sign(diff)
        for i in range(diff):
            lengths[i] -= inc
    # резлуьтат
    v = np.array(v1)
    # делаем перестановки
    start = 0
    for i,l in enumerate(lengths):
        if cross[i] == 1:
            v[start:start+l] = v2[start:start+l]
        start += l
    return v

class Genom(dict):
    """
    Set of named gens
    Support crossingover op and mutation for all gens in genom
    
    """
    def __init__(self, gens, *argv, **argw):
        if isinstance(gens, list):
            super().__init__({g.name:g for g in gens},*argv, **argw)
        else:
            super().__init__(gens,*argv, **argw)

        for v in self.values():
            assert isinstance(v, Gen) 
        
    def mutation(self, eps=0.01):
        for gen in self.values():
            gen.mutation(eps)

    def _check(self, other):
        """
        Check comatibility of two genom
        Now only equaled supported
        """
        assert len(self) == len(other)
        other_keys = other.keys()
        for k in self.keys():
            assert k in other_keys 
            
    def cross(self, other):
        self._check(other)
        # self class constructor
        new = self.__class__([])
        for name,gen in self.items():
            new[name] = gen.copy()
            new[name].data = cross(new[name].data, other[name].data)
        return new
    
    def merge(self, other):
        self.update(other)
    
    def __str__(self):
        s = '{count} gens:\n'.format(count = len(self))
        for gen in self.values():
            s += str(gen) + '\n'
        return s

