import pandas as pd
import numpy as np
from pandas import DataFrame as df
from heapq import nlargest, nsmallest
from random import randrange, randint, random
from sklearn import preprocessing
from math import inf, sqrt
from matplotlib import pyplot as plt
from time import time

df = pd.DataFrame()

# create random population before we get data
for _ in range(0, 5000):
    df = df.append([[random(), random()]])

values = df.values
normalized_values = preprocessing.MinMaxScaler().fit_transform(values)
df = pd.DataFrame(normalized_values, columns=['lat', 'lon'])

# Should be even
POPULATION_SIZE = 100
HALF_POPULATION_SIZE = int(POPULATION_SIZE / 2)

ITERATIONS = 100

MUTATION_RATE = 0.005

RANDOM_PATHS = 100

NUMBER_OF_STATIONS = 6
NUMBER_OF_ROUTE_GENES = 30
NUMBER_OF_ROUTES = 8

WALKING_SPEED = 1
NETWORK_SPEED = 15

def main():
    print("Creating population")
    population = Population(POPULATION_SIZE)
    for i in range(0, ITERATIONS):
        print("Running iteration " + str(i))
        population.iteration(fitness_function)

    population.plot_best(fitness_function)
    plt.show()

def fitness_function(individual):
    # start_time = time()
    total_distance = 0
    for _ in range(0, RANDOM_PATHS):
        # endpoints = df.sample(2)
        total_distance = total_distance + individual.to_network().distance_between(Location(random(), random()),Location(random(), random()))

    # print("fitness function time " + str(time() - start_time))

    return -total_distance

class Population():
    def __init__(self, num_individuals):
        self._population = []
        for _ in range(0, num_individuals):
            self._population.append(Individual())

    def iteration(self, fitness_function):

        scores = {}
        raw_scores = []
        total_score = 0
        for i in range(0, len(self._population)):
            individual = self._population[i]
            score = fitness_function(individual)
            scores[individual] = score
            total_score += score
            raw_scores.append(score)

        print("average score " + str(total_score / POPULATION_SIZE))
        print("max score " + str(max(raw_scores)))

        half_biggest = nlargest(HALF_POPULATION_SIZE, scores, key=scores.get)
        self._population = list(half_biggest)

        for _ in range(0, HALF_POPULATION_SIZE):
            a = self._population[randrange(0, HALF_POPULATION_SIZE)]
            b = self._population[randrange(0, HALF_POPULATION_SIZE)]
            c = a.merge_with(b)
            self._population.append(c)

    def plot_best(self, fitness_function):
        scores = {}
        raw_scores = []
        total_score = 0
        for i in range(0, len(self._population)):
            individual = self._population[i]
            score = fitness_function(individual)
            scores[individual] = score
            total_score += score
            raw_scores.append(score)

        biggest = list(nlargest(1, scores, key=scores.get))[0]

        biggest.to_network().plot()


class Individual():
    def __init__(self, genetics=None):
        self._genetics = genetics
        self._network = None

        if self._genetics is None:
            self._genetics = []

            for _ in range(0, NUMBER_OF_STATIONS):
                self._genetics.append(LocationGene())

            for _ in range(0, NUMBER_OF_ROUTES):
                self._genetics.append(SegmentGene())

    def merge_with(self, individual):
        mixing = np.random.choice([True, False], len(individual._genetics))
        new_genes = np.where(mixing, self._genetics, individual._genetics)
        for i in range(0, len(new_genes)):
            if random() < MUTATION_RATE:
                new_genes[i] = new_genes[i].mutate()

        return Individual(new_genes)

    def to_network(self):
        if None != self._network:
            return self._network

        locations = []
        for i in range(0, NUMBER_OF_STATIONS):
            locations.append(self._genetics[i].to_location())

        routes = []
        for i in range(0, NUMBER_OF_ROUTES):
            routes.append(self._genetics[NUMBER_OF_STATIONS + i].to_segment())

        self._network = Network(locations, routes)
        return self._network

class LocationGene():

    def __init__(self, lat=None, lon=None):
        self._lat = lat if lat is not None else random()
        self._lon = lon if lon is not None else random()

    def to_location(self):
        return Location(self._lat, self._lon)

    def mutate(self):
        lat_lon_decider = randint(0, 1)
        new_coordinate = random()
        if lat_lon_decider == 0:
            return LocationGene(new_coordinate, self._lon)
        else:
            return LocationGene(self._lat, new_coordinate)

class SegmentGene():

    def __init__(self, start_loc=None, end_loc=None):
        self._start_loc = start_loc if None != start_loc else LocationGene()
        self._end_loc = end_loc if None != end_loc else LocationGene()

    def to_segment(self):
        return Segment(self._start_loc.to_location(), self._end_loc.to_location())

    def mutate(self):
        start_stop_decider = randint(0, 1)
        if start_stop_decider == 0:
            return SegmentGene(self._start_loc.mutate(), self._end_loc)
        else:
            return SegmentGene(self._start_loc, self._end_loc.mutate())

class Network():
    def __init__(self, locations, segments):
        self._orig_locations = list(locations)
        self._locations = locations
        self._neighbors = {}
        for location in self._locations:
            self._neighbors[location] = []

        num_segments_used = 0
        for segment in segments:
            if num_segments_used >= NUMBER_OF_ROUTES:
                break

            neighbor_list = self._neighbors[segment.get_start(self._locations)]
            end = segment.get_end(self._locations)
            if neighbor_list.count(end) > 0:
                continue

            neighbor_list.append(end)
            num_segments_used = num_segments_used + 1

        for location in self._neighbors.keys():
            if len(self._neighbors[location]) == 0:
                self._locations.remove(location)

        self._segments = segments

    def distance_between(self, start, stop):
        # most accurate is to compare every pair of points but that takes too long as it multiplies the problem by v^2
        shortest_start = None
        shortest_start_dist = inf
        for location in self._locations:
            dist = start.distance_to(location)
            if dist < shortest_start_dist:
                shortest_start = location
                shortest_start_dist = dist

        shortest_start_idx = self._locations.index(shortest_start)

        shortest_stop = None
        shortest_stop_dist = inf
        for location in self._locations:
            dist = start.distance_to(location)
            if dist < shortest_stop_dist:
                shortest_stop = location
                shortest_stop_dist = dist

        shortest_stop_idx = self._locations.index(shortest_stop)

        path_dist = self.shortest_path_dist(shortest_start_idx, shortest_stop_idx)

        travel_time = self.travel_time(start, shortest_start_idx, path_dist, shortest_stop_idx, stop)

        return travel_time

    def travel_time(self, start, start_idx, path_dist, stop_idx, stop):
        network_time = path_dist / NETWORK_SPEED

        to_start_time = start.distance_to(self._locations[start_idx]) / WALKING_SPEED
        to_stop_time = stop.distance_to(self._locations[stop_idx]) / WALKING_SPEED

        return network_time + to_start_time + to_stop_time

    def shortest_path_dist(self, start_idx, stop_idx):
        Qdist = {}
        dist = {}
        prev = {}

        for v in self._locations:
            dist[v] = inf
            Qdist[v] = inf
            prev[v] = None

        Qdist[self._locations[start_idx]] = 0
        dist[self._locations[start_idx]] = 0

        while len(Qdist) != 0:
            smallest = list(nsmallest(1, Qdist, key=Qdist.get))[0]

            if smallest == self._locations[stop_idx]:
                return dist[smallest]

            Qdist.pop(smallest)

            for neighbor in self._neighbors[smallest]:
                temp = smallest.distance_to(neighbor) + dist[smallest]
                if temp < dist[neighbor]:
                    dist[neighbor] = temp
                    if neighbor in Qdist:
                        Qdist[neighbor] = temp
                    prev[neighbor] = smallest

        return inf

    def plot(self):
        for location in self._orig_locations:
            plt.plot(location.get_lat(), location.get_lon(), location.get_lat(), location.get_lon(), marker='o', color='red')

        for segment in self._segments:
            start = segment.get_start(self._orig_locations)
            end = segment.get_end(self._orig_locations)

            plt.plot([start.get_lat(), end.get_lat()], [start.get_lon(), end.get_lon()], marker='o', color='red')
            plt.plot([segment._start.get_lat(), segment._end.get_lat()], [segment._start.get_lon(), segment._end.get_lon()], marker='o', color='green')

class Segment():
    def __init__(self, start, end):
        self._start = start
        self._end = end

    def get_start(self, locations):
        return self._closest_point(locations, self._start)

    def get_end(self, locations):
        return self._closest_point(locations, self._end)

    def _closest_point(self, locations, point):
        closest_so_far = None
        closest_distance = inf

        for location in locations:
            distance = location.distance_to(point)
            if distance < closest_distance:
                closest_distance = distance
                closest_so_far = location

        return closest_so_far

class Location():
    def __init__(self, lat, lon):
        self._lat = lat
        self._lon = lon

    def distance_to(self, other):
        return sqrt((self._lat - other._lat) ** 2 + (self._lon - other._lon) ** 2)

    def get_lat(self):
        return self._lat

    def get_lon(self):
        return self._lon

if __name__ == "__main__":
    main()
