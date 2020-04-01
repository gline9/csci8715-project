import pandas as pd
import numpy as np
from pandas import DataFrame as df
from heapq import nlargest
from random import randrange
from sklearn import preprocessing

df = pd.DataFrame()

values = df.values()
normalized_values = preprocessing.MinMaxScaler().fit_transform(values)
df = pd.DataFrame(normalized_values, columns=['lat', 'lon'])

# Should be even
POPULATION_SIZE = 1000
HALF_POPULATION_SIZE = POPULATION_SIZE / 2

RANDOM_PATHS = 1000

def main():
    print("hello")
    pass

def fitness_function(network):
    total_distance = 0
    for _ in range(0, RANDOM_PATHS):
        endpoints = df.sample(2)
        total_distance = total_distance + network.distance_between((endpoints[0].lat, endpoints[0].lon),(endpoints[1].lat, endpoints[1].lon))

    return -total_distance

class Population():
    def __init__(self, num_individuals):
        self._population = []

    def iteration(self, fitness_function):

        scores = {individual:fitness_function(individual) for individual in self._population}
        half_biggest = nlargest(HALF_POPULATION_SIZE, scores, key=scores.get)
        self._population = list(half_biggest)

        for _ in range(0, HALF_POPULATION_SIZE):
            a = self._population[randrange(0, HALF_POPULATION_SIZE)]
            b = self._population[randrange(0, HALF_POPULATION_SIZE)]
            c = a.merge_with(b)
            self._population.append(c)

class Individual():
    def __init__(self, genetics):
        self._genetics = genetics

    def merge_with(self, genes):
        mixing = np.random.choice([True, False], genes.length)
        new_genes = np.where(mixing, x=self._genetics, y=genes._genetics)
        return Individual(new_genes)


class Segment():
    def __init__(self, start, end):
        self._start = start
        self._end = end

class Location():
    def __init__(self, lat, lon):
        self._lat = lat
        self._lon = lon

if __name__ == "__main__":
    main()
