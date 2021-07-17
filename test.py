import gym
from multiprocessing import Process


def f(x):
    while True:
        x = x**2
    print(result)
    return result

process = Process(target=f, args=(2, ))
process.start()
try:
    process.join()
except KeyboardInterrupt:
    process.kill()
    print("KeyboardInterrupt, process killed")