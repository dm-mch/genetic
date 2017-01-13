import numpy as np
import pandas as pd
import threading

from worker import TrainWorker, PopulateWorker, EvalQueuePack
from fenotypes import TSEntity
from data import TwoSigmaDataProvider, DataProvider
from const import GOOD_FRACTION, WOKERS_COUNT, MUTATION_EPSILON

def populate(population, data_provider, population_size=None):
    """
    Get population [list of EvalQueuePack with entity and score]
    and make new population

    """
    # select bests entity
    best_size =  int(len(population) * GOOD_FRACTION)
    bests_entity = np.array([p.entity for p in  sorted(population, key=lambda k: k.score, reverse=True)[:best_size]])
    # new population size
    population_size = population_size or len(population)
            
    new_population = []
    for i in range(population_size - best_size):    
        # select 2 random entity from bests
        pair = bests_entity[np.random.choice(np.arange(best_size, dtype=np.int32), size = 2, replace = False)] 
        # generate new one from pair
        new_entity = pair[0].rep(pair[1], eps = MUTATION_EPSILON)
        # add to population
        new_population.append(EvalQueuePack(train=data_provider.get_train(),
                                            validate= data_provider.get_validate(),
                                            entity=new_entity))
    # all bests add to new population
    for entity in bests_entity:
        new_population.append(EvalQueuePack(train=data_provider.get_train(),
                                            validate= data_provider.get_validate(),
                                            entity=entity))
    return new_population


def evolution(data_provider, generations=100, population_size=500):
    def population_callback(population):
        return populate(population, data_provider, population_size)

    population_worker = PopulateWorker(population_callback, id = "Populator", dequeue_size=population_size, generations=generations)
    train_workers = [TrainWorker(id="Train%d"%i, result_queue=population_worker.queue) for i in range(WOKERS_COUNT)]
    # param for queue train workers
    population_worker.add_params(workers=train_workers)

    first_generation = [EvalQueuePack(train=data_provider.get_train(),
                                      validate= data_provider.get_validate(),
                                      entity=TSEntity())
                                      for e in range(population_size * 2)]
    # queue first generation
    population_worker.balanced_enqueue(first_generation, [w.queue for w in train_workers])
    for w in train_workers:
        print(w.id, "queue size ", w.queue.qsize())

    for w in train_workers: w.start()
    population_worker.start()

    population_worker.join()
    print("Evolution finished")

def main():
    print("Create data provider...")
    dp = TwoSigmaDataProvider()
    print("Starting evolution...")
    evolution(dp)

if __name__ == "__main__":
    main()



