from environment import Environment, Agent, Wall
from video import Video
from age import Genome, Descriptor
from time import asctime, time
from sys import argv
import cProfile


class Controller:
    def __init__(self, file):
        print("Artificial Life Experiment 0.1")
        print()

        print("Initializing AGE descriptor")
        self.desc = Descriptor(alphabet = "ACGT",  # We use a biology-like alphabet
                               # Neurons:
                               devices = ["ACAA",  # Threshold activation function
                                          "ACAC",  # Sigmoidal (exp) activation function
                                          "ACAG"], # Linear activation function
                               terminal = "TT",    # Terminal marker
                               parameter = "GG",   # Parameter marker
                               # Mutation possibilities:
                               possibilities = {"char_delete":          0.020,
                                                "char_insert":          0.050,
                                                "char_replace":         0.050,
                                                "frag_delete":          0.010,
                                                "frag_move":            0.010,
                                                "frag_copy":            0.025,
                                                "device_insert":        0.050,
                                                "chromosome_delete":    0.001,
                                                "chromosome_copy":      0.001,
                                                "chromosome_crossover": 0.001})
        self.run = 0
        if (file==None):
            self.reset()
        else:
            print("Loading state: "+file)
            self.env = Environment()
            self.env.load(file)

    def reset(self):
        self.run += 1
        print("Resetting")
        
        print("Creating environment")
        self.env = Environment()
        #self.env.add_wallbox()
        w = (10.0, 10.0)
        self.seperation_wall = Wall(w, (self.env.size[0]-w[0], self.env.size[1]-w[1]))

        print("Initializing Video")
        self.video = Video(self, self.env)
        #self.video.record("record/night_%d.zip"%self.run)

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

    def mainloop(self):
        print("Entering mainloop")
        try:
            while (self.video.handle_event()):
                pass
                if (len(self.env.agents)<5):
                    self.spawn(50)
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
    #c.mainloop()
    cProfile.run("c.mainloop()")
