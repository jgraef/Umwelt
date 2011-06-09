#!/usr/bin/python3.1 -OO

from environment import Environment, Agent, Wall, FoodPatch
from video import Video
from age import Genome, Descriptor
from time import asctime, time
from sys import argv
import cProfile

DESCRIPTOR = Descriptor(alphabet = "ACGT",  # We use a biology-like alphabet
                        # Neurons:
                        devices = ["ACAA",  # Threshold activation function
                                   "ACAC",  # Sigmoidal (exp) activation function
                                   "ACAG"], # Linear activation function
                        terminal = "TT",    # Terminal marker
                        parameter = "GG",   # Parameter marker
                        # Mutation possibilities:
                        possibilities = {"char_delete":          0.010,
                                         "char_insert":          0.025,
                                         "char_replace":         0.025,
                                         "frag_delete":          0.005,
                                         "frag_move":            0.005,
                                         "frag_copy":            0.010,
                                         "device_insert":        0.025,
                                         "chromosome_delete":    0.001,
                                         "chromosome_copy":      0.001,
                                         "chromosome_crossover": 0.001})

class Controller:
    def __init__(self, file):
        print("Artificial Life Experiment 0.1")
        print()

        print("Initializing AGE descriptor")
        self.desc = DESCRIPTOR
        self.run = 0
        if (file==None):
            self.reset()
        else:
            print("Loading state: "+file)
            self.env = Environment()
            comments = self.env.load(file, self.desc)
            print(*comments, sep = '\n')

        print("Initializing Video")
        self.video = Video(self, self.env)
        #self.video.record("record/night_%d.zip"%self.run)

    def reset(self):
        self.run += 1
        print("Resetting")
        
        print("Creating environment")
        self.env = Environment()
        #self.env.add_wallbox()
        self.seperation_wall = Wall((-40.0, -40.0), (40.0, 40.0))

    def seperate(self):
        try:
            self.env.walls.remove(self.seperation_wall)
        except ValueError:
            self.env.walls.append(self.seperation_wall)

    def spawn(self, n = 1):
        print("Spawning %d random agents"%n)
        
        for i in range(n):
            g = Genome(desc = self.desc)
            g.add_randomly((2, 5), (100, 500))
            for j in range(5):
                g.mutate()
            a = Agent(self.env.random_pos(), self.env.random_angle(), g)
            self.env.agents.append(a)

    def clone(self, agent):
        genome = Genome(desc = self.desc, chromosomes = [c for c in agent.genome.chromosomes])
        self.env.agents.append(Agent(self.env.random_pos(agent.pos, 5.0), self.env.random_angle(), genome, agent.generation))

    def mainloop(self):
        print("Entering mainloop")
        try:
            while (self.video.handle_event()):
                if (len(self.env.agents)<5):
                    self.spawn(50)
                if (len(self.env.food_patches)==0):
                    for i in range(10):
                        self.env.food_patches.append(FoodPatch(self.env.random_pos()))
        except KeyboardInterrupt:
            pass
        print("Quit mainloop")

__all__ = ["Controller"]


if (__name__=="__main__"):
    try:
        file = argv[1]
    except IndexError:
        file = None

    c = Controller(file)
    c.mainloop()
    #cProfile.run("c.mainloop()")
