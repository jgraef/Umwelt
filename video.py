import pygame
from math import sin, cos, isnan, pi
from zipfile import ZipFile, ZIP_DEFLATED
from time import asctime
import subprocess


EVENT_RENDER = pygame.USEREVENT
EVENT_SIMULATE = pygame.USEREVENT+1
OUTLINE_WIDTH = 2
FRAMERATE = 10
SIMRATE = 20
FFMPEG_BIN = "/usr/bin/ffmpeg"
SCREENSHOT_DIR = "media/screenshots/"
RECORD_DIR = "media/records/"
STATE_DIR = "states/"

class Video:
    def __init__(self, controller, env):
        pygame.init()
        self.controller = controller
        self.rec = None
        self.pause = False
        # Whether to render eyes
        # 0 = don't render
        # 1 = render vision rays
        # 2 = render percepted colors
        self.render_eyes = 0

        self.env = env
        self.env_scale = 800.0/env.size[1]
        env_rect = pygame.Rect(0, 0, int(env.size[0]*self.env_scale), int(env.size[1]*self.env_scale))

        popgraph_rect = pygame.Rect(0, env_rect.height, env_rect.width, 150)
        self.popgraph_queue = []
        self.popgraph_t = 0
        
        w = env_rect.width
        h = env_rect.height+popgraph_rect.height
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Artificial Life - "+asctime())
        
        self.env_surf = self.screen.subsurface(env_rect)
        self.popgraph_surf = self.screen.subsurface(popgraph_rect)

        self.font = pygame.font.SysFont("freemono", 12, True)
        pygame.time.set_timer(EVENT_RENDER, int(1000/FRAMERATE))
        pygame.time.set_timer(EVENT_SIMULATE, int(1000/SIMRATE))

    def __del__(self):
        self.record()

    def record(self, filename = None):
        if (self.rec!=None):
            self.rec["zip"].close()
            self.rec = None
            
        if (filename!=None):
            z = ZipFile(filename, "w")
            z.comment = ("fps=%d; simrate=%d; w=%d; h=%d"%(FRAMERATE, SIMRATE, self.screen.get_width(), self.screen.get_height())).encode()
            self.rec = {"last_t": 0,
                        "zip": z,
                        "tmp": "/tmp/al_tmp.jpg"}

    def record_frame(self):
        pygame.image.save(self.screen, self.rec["tmp"])
        self.rec["zip"].write(self.rec["tmp"], str(self.env.t).zfill(8)+".jpg")

    def render(self):
        # background color: #C1BC6C
        self.screen.fill((0xC1, 0xBC, 0x6C))

        self.render_env()
        self.render_popgraph()

        if (self.rec!=None):
            if (self.env.t>self.rec["last_t"]):
                self.record_frame()
            # draw record symbol
            p = (self.screen.get_width()-20, 20)
            pygame.draw.circle(self.screen, (255, 0, 0), p, 15, 0)
            pygame.draw.circle(self.screen, (0, 0, 0), p, 15, 2)

        pygame.display.flip()

    def render_popgraph(self):
        def draw_text(text, pos):
            self.popgraph_surf.blit(self.font.render(text, True, (0, 0, 0)), pos)

        w, h = self.popgraph_surf.get_size()
        pygame.draw.line(self.popgraph_surf, (0, 0, 0), (0, 0), (w, 0))

        if (self.env.t>self.popgraph_t):
            # append new population size
            self.popgraph_queue.append(len(self.env.agents))
            if (len(self.popgraph_queue)>=w):
                self.popgraph_queue.pop(0)
            self.popgraph_t = self.env.t

        h_scale = 1.0 # max(self.popgraph_queue)/h

        # graph graph
        lasty = 0
        for x in range(len(self.popgraph_queue)):
            y = self.popgraph_queue[x]
            pygame.draw.line(self.popgraph_surf, (0, 0, 0), (x, int(h_scale*(h-lasty))), (x, int(h_scale*(h-y))))
            lasty = y

        # render some information
        draw_text("Time:   %d (%f s)"%(self.env.t, self.env.comp_t), (5, 5))
        draw_text("Agents: %d"%(len(self.env.agents)), (5, 20))
        draw_text("Food:   %d"%(len(self.env.food_patches)), (5, 35))
        draw_text("Memory: %d kB"%(0.0009765625*self.env.memory_usage), (5, 50))


    def render_env(self):
        def convert_color(c):
            return tuple(map(lambda x: int(255*x), c))
        def convert_vec(v):
            if (type(v)==tuple):
                return tuple(map(lambda x: int(x*self.env_scale), v))
            elif (type(v)==float or type(v)==int):
                return int(v*self.env_scale)

        # render food patches
        for f in self.env.food_patches:
            pygame.draw.circle(self.env_surf,
                               convert_color(f.get_color()),
                               convert_vec(f.pos),
                               convert_vec(f.size))
            pygame.draw.circle(self.env_surf,
                               (0, 0, 0),
                               convert_vec(f.pos),
                               convert_vec(f.size),
                               OUTLINE_WIDTH)

        # render agents
        for a in self.env.agents:
            p = convert_vec(a.pos)
            c = convert_color(a.get_color())
            pygame.draw.circle(self.env_surf,
                               c,
                               p,
                               convert_vec(a.size))
            pygame.draw.circle(self.env_surf,
                               (0, 0, 0),
                               p,
                               convert_vec(a.size),
                               OUTLINE_WIDTH)
            pygame.draw.line(self.env_surf,
                             (200, 200, 200),
                             convert_vec(a.pos),
                             convert_vec((a.pos[0]+a.size*sin(a.angle), a.pos[1]+a.size*cos(a.angle))),
                             OUTLINE_WIDTH)
            if (self.render_eyes==1):
                rl = self.env_scale*self.env.vision_range
                cl = convert_color(a.last_perception[4:7])
                cr = convert_color(a.last_perception[7:10])
                al = a.angle+0.01*pi
                ar = a.angle-0.01*pi
                pygame.draw.line(self.env_surf,
                                 cl,
                                 p,
                                 (int(p[0]+rl*sin(al)), int(p[1]+rl*cos(al))),
                                 1)
                pygame.draw.line(self.env_surf,
                                 cr,
                                 p,
                                 (int(p[0]+rl*sin(ar)), int(p[1]+rl*cos(ar))),
                                 1)
            elif (self.render_eyes==2):
                ex, ey = 4.0*cos(a.angle), 4.0*sin(a.angle)
                cl = convert_color(a.last_perception[4:7])
                cr = convert_color(a.last_perception[7:10])
                pygame.draw.circle(self.env_surf,
                                   cl,
                                   (int(p[0]+ex), int(p[1]-ey)),
                                   4)
                pygame.draw.circle(self.env_surf,
                                   cr,
                                   (int(p[0]-ex), int(p[1]+ey)),
                                   4)

        # render walls
        for w in self.env.walls:
            pygame.draw.line(self.screen, convert_color(w.get_color()), convert_vec(w.a), convert_vec(w.b), 2)
    
    def handle_event(self):
        e = pygame.event.wait()
        if (e.type==pygame.QUIT):
            return False
        elif (e.type==EVENT_RENDER or e.type==pygame.VIDEOEXPOSE):
            self.render()
        elif (e.type==EVENT_SIMULATE and not self.pause):
            self.env.step()
        elif (e.type==pygame.KEYUP):
            if (e.key==pygame.K_F9):
                # Make a screenshot
                path = SCREENSHOT_DIR+asctime()+".png"
                print("Screenshot: "+path)
                self.screenshot(path)
            elif (e.key==pygame.K_F10):
                # Start/Stop recording (JPEGs in a ZIP archive)
                if (self.rec==None):
                    path = RECORD_DIR+asctime()+".zip"
                    print("Start recording: "+path)
                    self.record(path)
                else:
                    print("Stop recording")
                    self.record()
            elif (e.key==pygame.K_F11):
                # Save state
                path = STATE_DIR+asctime()+".txt"
                print("Save state: "+path)
                self.env.save(path)
            elif (e.key==pygame.K_F12):
                self.controller.spawn(10)
            elif (e.key==pygame.K_SPACE):
                self.pause = not self.pause
            elif (e.key==pygame.K_e):
                self.render_eyes = (self.render_eyes+1)%3
            elif (e.key==pygame.K_s):
                self.controller.seperate()
                
        pygame.event.clear(e.type)
        return True

    def screenshot(self, filename):
        pygame.image.save(self.screen, filename)

__all__ = ["Video"]
