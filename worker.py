import threading
import queue
import numpy as np
import sklearn.metrics as metrics

from const import GENERATIONS_COUNT

class StoppableThread(threading.Thread):
    """
    A thread that has a 'stop' event.
    """

    def __init__(self):
        super().__init__()
        self._stop_evt = threading.Event()

    def stop(self):
        """ Stop the thread"""
        self._stop_evt.set()

    def stopped(self):
        """
        Returns:
            bool: whether the thread is stopped or not
        """
        return self._stop_evt.is_set()

    def queue_put_stoppable(self, q, obj):
        """ Put obj to queue, but will give up when the thread is stopped"""
        while not self.stopped():
            try:
                q.put(obj, timeout=5)
                break
            except queue.Full:
                pass

    def queue_get_stoppable(self, q, count=1):
        """ Take obj from queue, but will give up when the thread is stopped"""
        buf = []
        while not self.stopped():
            try:
                buf.append(q.get(timeout=5))
            except queue.Empty:
                pass
            if len(buf) == count:
                return buf[0] if count == 1 else buf 


class QueueWorker(StoppableThread):
    """ Worker who dequeue batch from self.queue and do task func(QueueWorker, batch, **argw)  """
    def __init__(self, func, id=None, dequeue_size=1, maxsize=0, **argw):
        self.queue = queue.Queue(maxsize)
        self.id = id
        self._func = func
        self._argw = argw
        self._dequeue_size = dequeue_size
        super().__init__()

    def add_params(self, **argw):
        self._argw.update(argw)

    def run(self):
        print("Worker {} started".format(self.id))
        while not self.stopped():
            batch = self.queue_get_stoppable(self.queue, count=self._dequeue_size)
            self._func(self, batch, **self._argw)
        print("Worker {} finished".format(self.id))

class EvalQueuePack:
    def __init__(self, **argw):
        self.train = argw.get('train', None)
        self.validate = argw.get('validate', None)
        self.entity = argw.get('entity', None)
        self.score = argw.get('score', None)


class TrainWorker(QueueWorker):
    """ Get EvalQueuePack from self.queue, train and score, than put data to result queue """
    def __init__(self, score=metrics.accuracy_score, *argv, **argw):
        self.score = score
        super().__init__(TrainWorker.train_and_score, *argv, **argw)   

    @staticmethod
    def train_and_score(self, queue_pack, result_queue=None):
        assert isinstance(queue_pack, EvalQueuePack)
        qp = queue_pack
        qp.entity.fit(qp.train['x'], qp.train['y'])
        #print("self.score:",self.score)
        qp.score = self.score(qp.validate['y'], qp.entity.action(qp.validate['x']))
        # No more needed
        qp.train = None
        qp.validate = None
        # put to result queue
        self.queue_put_stoppable(result_queue, qp)


class PopulateWorker(QueueWorker):
    """ """
    def __init__(self, populate_callback, *argv, generations=GENERATIONS_COUNT, **argw):
        self.populate = populate_callback
        self.result_population = None
        self.generations = generations
        self.generation_count = 0
        super().__init__(PopulateWorker.populate_and_enqueue, *argv, **argw)   

    @staticmethod
    def populate_and_enqueue(self, queue_packs, workers=None):
        """ 
        @param queue_packs list of EvalQueuePack with entity and score 
        @param workers list of TrainWorker for enqueue new population for train

        """
        if self.generation_count >= self.generations:
            #finish. stop workers and self
            print(self.generation_count, "evaluated... stoping all workers")
            # save result population
            self.result_population = queue_packs
            # stop all workers
            for w in workers:
                w.stop()
            self.stop()
        else:
            print(self.generation_count, "Bests score:", 
                 ["%.3f"%p.score for p in sorted(queue_packs, key=lambda k: k.score, reverse=True)[:10]])
            # get old population sort\score and generate new        
            population = self.populate(queue_packs)
            # enqueue new population
            self.balanced_enqueue(population, [w.queue for w in workers])
            self.generation_count += 1

    def balanced_enqueue(self, queue_packs, queues):
        qsizes = [q.qsize() for q in queues]
        maxq = np.max(qsizes)
        meanq = np.mean(qsizes)
        if maxq > 1.5 * meanq:
            print("Balancing queue. Sizes ", qsizes)
            queues.remove(queues[np.argmax(qsizes)])

        max_queue = int(len(queue_packs)/len(queues))
        queue_index = 0
        for i, pack in enumerate(queue_packs):
            self.queue_put_stoppable(queues[queue_index], pack)
            if i>0 and i%max_queue == 0 and queue_index < len(queues) - 1:
                queue_index+=1


def test():
    import numpy as np

    def put_plus_one(worker, batch, queue=None):
        if batch is None: return
        worker.queue_put_stoppable(queue, batch+1)
        print("{} worker put {}. Self queue size {}".format(worker.id, batch+1, worker.queue.qsize()))

    def sum_and_balance(worker, batch, workers=None, max=1000):
        if batch is None: return
        r = np.sum(batch)
        for w in workers:
            worker.queue_put_stoppable(w.queue, r)
        print("{} worker put {}. Self queue size {}".format(worker.id, r, worker.queue.qsize()))
        if r >= max:
            print("Sum", r, "reached")
            for w in workers:
                w.stop()
            worker.stop()

    sum_worker = QueueWorker(sum_and_balance, "Summator0", dequeue_size=8, max=10000)
    plus_one_workers = [QueueWorker(put_plus_one, "PlusOne%d"%i, queue=sum_worker.queue) for i in range(8)]
    sum_worker.add_params(workers=plus_one_workers)

    for w in plus_one_workers:
        w.queue.put(1)
        print(w.id, w.queue.qsize())

    sum_worker.start()
    for w in plus_one_workers: w.start()

if __name__ == '__main__':
    test()