from __future__ import print_function

import gym
import gym.wrappers

import multiprocessing
import neat
import numpy as np
import os
import pickle
import random
import time


env = gym.make('BipedalWalker-v2')

print("action space: {0!r}".format(env.action_space))
print("observation space: {0!r}".format(env.observation_space))


class BipedalBotGenome(neat.DefaultGenome):
    def __init__(self, key):
        super().__init__(key)
        self.discount = None

    def configure_new(self, config):
        super().configure_new(config)
        self.discount = 0.01 + 0.98 * random.random()

    def configure_crossover(self, genome1, genome2, config):
        super().configure_crossover(genome1, genome2, config)
        self.discount = random.choice((genome1.discount, genome2.discount))

    def mutate(self, config):
        super().mutate(config)
        self.discount += random.gauss(0.0, 0.05)
        self.discount = max(0.01, min(0.99, self.discount))

    def distance(self, other, config):
        dist = super().distance(other, config)
        disc_diff = abs(self.discount - other.discount)
        return dist + disc_diff


class PooledErrorCompute(object):
    def __init__(self):
        self.pool = multiprocessing.Pool()
        self.test_episodes = []

        self.min_reward = -200
        self.max_reward = 200

        self.episode_score = []
        self.episode_length = []

    def evaluate_fitness(self, genomes, config):
        for genome in genomes:
            observation = env.reset()

            net = neat.nn.FeedForwardNetwork.create(genome, config)

            frames = 0
            total_fitness = 0
            reward = 0
            fitness = 0
            for k in range(5):
                while 1:

                    # Feed the latest observation into the neural net
                    output = net.activate(observation)
                    output = np.clip(output, -1, 1)

                    observation, reward, done, info = env.step(np.array(output))
                    fitness += reward
                    frames += 1

		    #print(frames)
                    if done:
                        total_fitness += fitness
                        env.reset()
                        break

            genome.fitness = total_fitness / 5
            print('Genome fitness = {}'.format(genome.fitness))

def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(BipedalBotGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    ec = PooledErrorCompute()
    pop.run(ec.evaluate_fitness, 1)
    winner = pop.statistics.best_genome()
    del pop

    winning_net = neat.nn.FeedForwardNetwork.create(winner, config)

    env = gym.wrappers.Monitor(env, 'results', force=True)

    # stats = neat.StatisticsReporter()
    # pop.add_reporter(stats)
    # pop.add_reporter(neat.StdOutReporter(True))
    # Checkpoint every 25 generations or 900 seconds.
    # pop.add_reporter(neat.Checkpointer(25, 900))

    streak = 0
    episode = 0
    best_reward = -200
    while streak < 100:
        fitness = 0
        frames = 0
        reward = 0
        observation = env.reset()
        while 1:
            inputs = observation

            # active neurons
            output = winningnet.serial_activate(inputs)
            output = np.clip(output, -1, 1)
            # print(output)
            observation, reward, done, info = env.step(np.array(output))

            fitness += reward

            frames += 1

            if done or frames > 2000:
                if fitness >= 100:
                    print(fitness)
                    print ('streak: ', streak)
                    streak += 1
                else:
                    print(fitness)
                    print('streak: ', streak)
                    streak = 0
                break

        episode += 1
        if fitness > best_reward:
            best_reward = fitness

        #except KeyboardInterrupt:
        #    print("User break.")
        #    break

    env.close()


if __name__ == '__main__':
    run()
