import ray

ray.init()


@ray.remote
class Counter:
    def __init__(self):
        self.data = []

    def inc(self):
        self.data.append(1)

    def get(self):
        return self.data


@ray.remote
def x(counter):
    ray.get(counter.inc.remote())


counter = Counter.remote()
task = []
for i in range(4):
    task.append(x.remote(counter))
ray.wait(task)
print(ray.get(counter.get.remote()))
