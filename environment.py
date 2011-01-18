from pycann import Network
from random import random, uniform, triangular
from math import sin, cos, sqrt, hypot, pi, isnan, isinf, modf
from time import time

def ray_circle_intersect(E, d, C, r):
    """ Check for ray/circle intersection
        E: Starting point of ray
        d: Direction vector of ray
        C: Center of circle
        r: Radius of circle """
    f = (E[0]-C[0], E[1]-C[1])
    a = d[0]**2+d[1]**2
    b = 2*f[0]*d[0]+2*f[1]*d[1]
    c = f[0]**2+f[1]**2-r**2
    D = b**2-4*a*c
    if (D<0): # no intersection
        return False
    else:
        D = sqrt(D)
        denom = 2*a
        return ((-b+D)/denom,
                (-b-D)/denom)

def circle_circle_intersect(pa, ra, pb, rb):
    return ((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2<=(ra+rb)**2)

def ray_ray_intersect(P0, D0, P1, D1):
    # normal vector
    perp = lambda v: (v[1], -v[0])
    # dot product
    dot = lambda v, u: v[0]*u[0]+v[1]*u[1]
    # vector subtraction
    vecsub = lambda v, u: (v[0]-u[0], v[1]-u[1])
    
    if (dot(perp(D1), D0)==0):
        # rays are parallel
        return None
    else:
        # rays intersect
        denominator = dot(perp(D1),D0)
        s = dot(perp(D1), vecsub(P1, P0))/denominator
        t = dot(perp(D0), vecsub(P1, P0))/denominator
        return s, t

def in_range(x, m = 0.0, M = 1.0):
    if (x<m):
        return m
    elif (x>M):
        return M
    else:
        return x

def random_or_fixed(v, rand = uniform):
    if (type(v)==int):
        v = float(v)
    if (type(v)==tuple and len(v)==2):
        return rand(*v)
    elif (type(v)==float):
        return v
    else:
        raise TypeError("Invalid type: "+str(type(v)))


class Brain(Network):
    """ Implements a pycann network created from a genome. """

    FIXED_INPUTS = 10 # (Health: 1, Tactile: 2, Vision: 6, Carry: 1)
    FIXED_OUTPUTS = 7 # (Forward: 1, Turn: 2, Eat: 1, Mate: 1, Color: 1, Carry: 1)
    
    def __init__(self, genome):
        # mapping for different activation functions
        # FIXME Linear neurons don't work. I think in recurrent networks some
        # voltages exceede until they become infinite
        activation_functions = {"ACAA": "SIGMOID_STEP",
                                "ACAC": "SIGMOID_EXP",
                                "ACAG": "LINEAR"}
        # init network
        num_neurons = len(genome.devices)
        Network.__init__(self, self.FIXED_INPUTS, max((0, num_neurons-self.FIXED_INPUTS-self.FIXED_OUTPUTS)), self.FIXED_OUTPUTS)
        self.set_learning_rate(0.02)
        self.set_gamma(0,  0.1)
        self.set_gamma(1,  0.01)
        self.set_gamma(2,  0.01)
        self.set_gamma(3, -0.01)

        # configure neurons
        for i in range(num_neurons):
            di = genome.devices[i]

            # activation function
            self.set_activation_function(i, activation_functions[di.device])
            #self.set_activation_function(i, "SIGMOID_STEP")

            # threshold
            try:
                t = di.parameters[0][1]
            except IndexError:
                t = 1.0
            self.set_threshold(i, t)

            # connect
            mc = None
            for j in range(num_neurons):
                dj = genome.devices[j]

                # connection weights
                # Terminal 0: output
                # Terminal 1: input
                try:
                    w = -1.0+4.0*genome.terminal_score(di.terminals[0], dj.terminals[1])
                except IndexError:
                    w = 0.0
                # NOTE: this sets weight of connection from neuron i to neuron j
                self.set_weight(i, j, w)

                # modularity connection
                # Terminal 2: modularity connection (connected to output terminal of another neuron)
                try:
                    w = -1.0+4.0*genome.terminal_score(di.terminals[2], dj.terminals[0])
                    if (mc==None or w>mc[1]):
                        mc = (j, w)
                except IndexError:
                    pass

            if (mc!=None):
                self.set_mod_connection(i, *mc)

    def process(self, inputs):
        self.set_inputs(*inputs)
        self.step()
        o = self.get_outputs()
        return o


class Agent:
    """ Class representing an autonomous agent in an environment
        An agent consists of environmental information (position, angle, size
        and color), a health value, a genome and a neural network. """
    
    def __init__(self, pos, angle, genome):
        self.pos = pos
        self.angle = angle
        self.genome = genome
        self.health = 500.0
        self.size = 0.0
        self.brain = None
        self.last_action = (0.0, 0.0, False, False, 0.0)
        self.last_perception = tuple((0.0 for i in range(10)))
        self.carry = None
        self.decode_genome()

    def __repr__(self):
        return "<Agent pos=%s angle=%f health=%f size=%f action=%s>"%(repr(self.pos), self.angle, self.health, self.size, repr(self.last_action))

    def decode_genome(self):
        # TODO decode genome: size, color
        self.genome.parse()
        self.brain = Brain(self.genome)
        self.size = 1.0
        self.memory_usage = self.brain.memory_usage+len(self.genome)


    def get_color(self):
        return (in_range(0.001*self.health),
                0.0,
                self.last_action[4])

    def get_action(self, tsolid, tfood, vleft, vright):
        # Perception: Health,
        #             Tactile sensor,
        #             Vision left,
        #             Vision right,
        #             Carry
        # Action:     Forward movement,
        #             Turning movement,
        #             Eating,
        #             Mating,
        #             Color,
        #             Carry
        TURN_RATE = 0.125*pi
        SPEED = 0.5
        perception = (0.001*self.health,
                      1.0 if (tsolid) else 0.0,
                      1.0 if (tfood) else 0.0,
                      1.0 if (self.carry!=None) else 0.0)+\
                      vleft+vright
        self.last_perception = perception
        action = self.brain.process(perception)
        action = (action[0]*SPEED,
                  (action[1]-action[2])*TURN_RATE,
                  action[3]>0.5,
                  action[4]>0.5,
                  in_range(action[5]),
                  action[6]>0.5)
        self.last_action = action
        return action


class FoodPatch:
    """ Class for food patches. Food patches supply health to the agents in
        a environment. The energy they supply is limited and decays. """
    
    def __init__(self, pos, size = (4.0, 10.0), energy = (150.0, 500.0), decay = (1.0, 2.0)):
        self.pos = pos
        self.size = random_or_fixed(size)
        self.energy = random_or_fixed(energy)
        self.decay = random_or_fixed(decay)
        self.max_energy = self.energy

    def __repr__(self):
        return "<FoodPatch pos=%s size=%f energy=%f decay=%f>"%(repr(self.pos), self.size, self.energy, self.decay)

    def get_color(self):
        return (0.0, in_range(self.energy/self.max_energy, 0.1), 0.0)

class Wall:
    """ Class for walls. Agents can't move through them. """

    def __init__(self, a, b, harm = 10.0, color = (1.0, 0.0, 0.0)):
        self.a = a
        self.b = b
        self.d = (b[0]-a[0], b[1]-a[1])
        self.harm = harm
        self.color = color

    def get_color(self):
        return self.color

    def __repr__(self):
        return "<Wall A=%s B=%s harm=%f color=%s>"%(self.a, self.b, self.harm, self.color)

class Environment:
    """ Class for environments. An environment is a closed space of agents and
        food patches. It coordinates movement and other actions between agents.
        """
    
    def __init__(self, size = (300.0, 200.0), food_rate = 0.25, vision_percent = 0.10):
        self.size = size
        food_rate = modf(food_rate)
        self.food_rate = (int(food_rate[1]), round(1.0/food_rate[0]) if food_rate[0]!=0 else None)
        self.agents = []
        self.food_patches = []
        self.walls = []
        self.t = 0
        self.comp_t = 0
        self.vision_range = vision_percent*hypot(*self.size)
        self.memory_usage = 0

    def step(self, ct_factor = 0.0):
        """ Do one time step in simulation. Time is measured in these steps. """

        t_start = time()
        self.memory_usage = 0
        
        # decay food patches
        for f in self.food_patches:
            f.energy -= f.decay
            if (f.energy<=0.0):
                self.food_patches.remove(f)
            
        # create food patches
        def add_foodpatch():
            offset = (0.5*self.size[0]+0.01*self.t,
                      0.5*self.size[1])
            p = ((triangular(-1.0, 1.0)*0.2*self.size[0]+offset[0])%self.size[0],
                 (triangular(-1.0, 1.0)*0.2*self.size[1]+offset[1])%self.size[0])
            self.food_patches.append(FoodPatch(p))
        for i in range(int(self.food_rate[0])):
            add_foodpatch()
        if (self.food_rate[1]!=None and self.t%self.food_rate[1]==0):
            add_foodpatch()

        # update agents
        for a in self.agents:
            # count memory usage (ANN and AGE)
            self.memory_usage += a.memory_usage
            
            # constant lose of health
            a.health -= 0.10*a.size

            # check if agent is close to another agent
            ab = self.is_in_agent(a)
            # check if agent is close to food
            f = self.is_in_foodpatch(a)
            # check if agent collides with wall
            collision = self.is_in_wall(a)
            if (collision):
                w, cv = collision
                a.health -= a.size*w.harm
                a.pos = ((a.pos[0]+cv[0]), (a.pos[1]+cv[1]))
            
            # action
            action = a.get_action(ab!=None or collision!=None,
                                  f!=None,
                                  self.vision(a, +0.01*pi),
                                  self.vision(a, -0.01*pi))
            a.pos = ((a.pos[0]+action[0]/a.size*sin(a.angle))%self.size[0],
                     (a.pos[1]+action[0]/a.size*cos(a.angle))%self.size[1])
            a.angle = (a.angle+action[1])%(2.0*pi)
            if (isnan(a.pos[0]) or isnan(a.pos[1]) or isnan(a.angle)):
                print("Agent #%d"%self.agents.index(a))
                print(a)
                raise ValueError("Agent includes a NaN value: "+repr(a))
            # loose energy for moving forward and turning
            if (action[0]):
                a.health -= 0.1 if a.carry==None else 0.2
            if (action[1]):
                a.health -= 0.05

            # pick up food
            if (f!=None and f.energy>0.0 and action[5] and a.carry==None):
                amount = min((f.energy, 5.0*a.size))
                a.carry = amount
                f.energy -= amount
            # lay down food to existing food patch
            elif (f!=None and not action[5] and a.carry!=None):
                f.energy += a.carry
                a.carry = None
            # lay down food
            elif (f==None and not action[5] and a.carry!=None):
                self.food_patches.append(FoodPatch(a.pos, 1.5, a.carry, 0))
                a.carry = None
                  
            # eat food
            if (f!=None and action[2]):
                amount = min((f.energy, 2.5*a.size))
                a.health += amount
                f.energy -= amount

            # action with other agent
            if (ab!=None):
                # mate
                if (action[3] and a.health>100.0 and ab.health>100.0):
                    self.mate(a, ab)
                # eat
                # let them taste flesh
                # to get hunters increase health gain factor when eating flesh
                # and/or decrease health gain factor when eathing food (or
                # decrease food rate so there aren't enough food patches)
                if (action[2]):
                    amount = min((ab.health, 3.0*a.size))
                    a.health += 0.9*amount
                    ab.health -= amount

            # remove agents with 0 or less health
            # (turn them into food patches)
            if (a.health<=0.0):
                self.agents.remove(a)
                self.food_patches.append(FoodPatch(a.pos, a.size, (5.0, 10.0)))

        # increment time
        self.t += 1

        # subtract computation time from health
        self.comp_t = time()-t_start
        if (ct_factor!=0.0):
            for a in self.agents:
                a.health -= ct_factor*self.comp_t

    def vision(self, a, angle = 0.0):
        """ Returns the visual perception of an agent """
        angle = (a.angle+angle)%(2*pi)
        d = (sin(angle), cos(angle))
        ar = a.size
        nearest = (None, None)
        # collide vision ray with other agents and food patches
        for b in self.agents+self.food_patches:
            if (a!=b and not circle_circle_intersect(a.pos, self.vision_range, b.pos, b.size)):
                t = ray_circle_intersect(a.pos, d, b.pos, b.size)
                if (t):
                    t = t[0] if (t[0]<t[1] and t[0]>ar) else t[1]
                    # t is the distance to the intersection point
                    if (t>=ar and t<=self.vision_range and (nearest[0]==None or t<nearest[0])):
                        nearest = (t, b)
        # collide vision ray with walls
        for w in self.walls:
            t = ray_ray_intersect(w.a, w.d, a.pos, d)
            if (t):
                tw, ta = t
                if (tw>0.0 and tw<1.0 and ta>=ar and ta<=self.vision_range and (nearest[0]==None or ta<nearest[0])):
                    nearest = (ta, w)
        if (nearest[1]!=None):
            c = nearest[1].get_color()
            if (type(c)!=tuple and len(c)!=3):
                print(repr(c))
            return c
        else:
            return (1.0, 1.0, 1.0)

    def mate(self, a, b):
        """ Performs the action of mating agent a and b in an environment.
            This does spawn a child agent between agent a and b with a
            crossed over genome. """
        pos = ((a.pos[0]+b.pos[0])/2,
               (a.pos[1]+b.pos[1])/2)
        genome = a.genome.crossover(b.genome)
        genome.mutate()
        c = Agent(pos, self.random_angle(), genome)
        # each parent gives 25% of its health to the child
        c.health = 0.25*(a.health+b.health)
        a.health *= 0.5
        b.health *= 0.5
        self.agents.append(c)

    def is_in_foodpatch(self, a):
        """ Checks whether an agent is at a foodpatch """
        for f in self.food_patches:
            if (circle_circle_intersect(a.pos, a.size, f.pos, f.size)):
                return f
        return None

    def is_in_agent(self, a):
        """ Checks whether an agent is near another agent """
        for b in filter(lambda c: c!=a, self.agents):
            if (circle_circle_intersect(a.pos, a.size, b.pos, b.size)):
                return b
        return None

    def is_in_wall(self, a):
        """ Checks whether an agent collides with a wall """
        for w in self.walls:
            t = ray_circle_intersect(w.a, w.d, a.pos, a.size)
            if (t):
                if ((t[0]>0.0 and t[0]<1.0) or (t[1]>0.0 and t[1]<1.0)):
                    t = 0.5*(t[0]+t[1])
                    # Collision vector (from collision point to agent position)
                    cv = (a.pos[0]-w.a[0]-w.d[0]*t,
                          a.pos[1]-w.a[1]-w.d[1]*t)
                    return w, cv
        return None

    def random_pos(self):
        """ Returns a random position in the environment. """
        return (random()*self.size[0], random()*self.size[1])

    def random_angle(self):
        """ Returns a random angle. """
        return 2*pi*random()

    def save(self, path):
        f = open(path, "wt")
        print(self, file = f)
        for a in self.agents:
            print(a, file = f)
            for c in a.genome.chromosomes:
                print(c, file = f)
        for fp in self.food_patches:
            print(fp, file = f)
        for w in self.walls:
            print(w, file = f)
        f.close()

    def load(self, path):
        pass # TODO

    def add_wallbox(self, A = (1.0, 1.0), C = None):
        if (C==None):
            C = (self.size[0]-A[0],
                 self.size[1]-A[1])
        B = (A[0], C[1])
        D = (C[0], A[1])
        self.walls.append(Wall(A, B))
        self.walls.append(Wall(B, C))
        self.walls.append(Wall(C, D))
        self.walls.append(Wall(D, A))

    def __repr__(self):
        return "<Environment t=%d size=%s food_rate=%s vision_range=%f>"%(self.t, repr(self.size), repr(self.food_rate), self.vision_range)

__all__ = ["Agent", "FoodPatch", "Environment", "Wall"]
