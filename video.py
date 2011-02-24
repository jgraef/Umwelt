import pygame
from math import sin, cos, isnan, pi
from zipfile import ZipFile, ZIP_DEFLATED
from time import asctime
from environment import Agent, FoodPatch
import ossaudiodev
from utils import *


EVENT_RENDER = pygame.USEREVENT
EVENT_SIMULATE = pygame.USEREVENT+1
OUTLINE_WIDTH = 2
FRAMERATE = 10
SIMRATE = 20
SCREENSHOT_DIR = "media/screenshots/"
RECORD_DIR = "media/records/"
STATE_DIR = "states/"
SCROLL_SPEED = 0.1
AUDIO_SAMPLERATE = 8000


class Video:
    def __init__(self, controller, env, resolution = (800, 600)):
        pygame.init()
        self.controller = controller
        self.rec = None
        self.pause = False
        # Whether to render eyes
        # 0 = don't render
        # 1 = render vision rays
        # 2 = render percepted colors
        self.render_eyes = False
        # scrolling state
        self.scroll = None
        # selection
        self.selection_rect = None
        self.selection = []
        # hint system
        self.show_hints = False

        self.env = env
        self.env_scale = 4.0
        self.env_frame = pygame.Rect(0, 0, 1, 1)
        self.env_frame.center = (0, 0)

        self.popgraph_queue = []
        self.popgraph_t = 0
        
        self.set_resolution(resolution)
        pygame.display.set_caption("Artificial Life - "+asctime())
        pygame.display.set_icon(pygame.image.load("icon.png"))
        
        self.font = pygame.font.SysFont("freemono", 12, True)
        pygame.time.set_timer(EVENT_RENDER, int(1000/FRAMERATE))
        pygame.time.set_timer(EVENT_SIMULATE, int(1000/SIMRATE))

        # audio
        #TODO

    def __del__(self):
        self.audiodev.close()
        self.record()

    def set_resolution(self, resolution):
        env_rect = pygame.Rect(0, 0, resolution[0], resolution[1]-150)
        self.env_frame.w = env_rect.w/self.env_scale
        self.env_frame.h = env_rect.h/self.env_scale
        popgraph_rect = pygame.Rect(0, env_rect.height, env_rect.width, 150)
        self.screen = pygame.display.set_mode(resolution, pygame.RESIZABLE)
        self.env_surf = self.screen.subsurface(env_rect)
        self.popgraph_surf = self.screen.subsurface(popgraph_rect)


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
        self.render_env()
        self.render_popgraph()

        if (self.rec!=None):
            if (self.env.t>self.rec["last_t"]):
                self.record_frame()

        pygame.display.flip()

    def render_popgraph(self):
        def draw_text(text, pos, color = (0, 0, 0)):
            self.popgraph_surf.blit(self.font.render(text, True, color), pos)

        # background
        self.popgraph_surf.fill((255, 255, 255))

        w, h = self.popgraph_surf.get_size()
        pygame.draw.line(self.popgraph_surf, (0, 0, 0), (0, 0), (w, 0))

        if (self.env.t>self.popgraph_t):
            # append new population size
            self.popgraph_queue.append(len(self.env.agents))
            if (len(self.popgraph_queue)>=w):
                self.popgraph_queue.pop(0)
            self.popgraph_t = self.env.t

        h_scale = h/max(self.popgraph_queue)

        # graph graph
        lasty = 0
        for x in range(len(self.popgraph_queue)):
            y = self.popgraph_queue[x]
            pygame.draw.line(self.popgraph_surf, (0, 0, 0), (x, h-int(h_scale*lasty)), (x, h-int(h_scale*y)))
            lasty = y

        # render some information
        draw_text("Time:   %d (%f s)"%(self.env.t, self.env.comp_t), (5, 5))
        draw_text("Agents: %d"%(len(self.env.agents)), (5, 20))
        draw_text("Food:   %d"%(len(self.env.food_patches)), (5, 35))
        draw_text("Memory: %d kB"%(0.0009765625*self.env.memory_usage), (5, 50))
        if (self.rec!=None):
            draw_text("REC", (w-100, 5), (255, 0, 0))

    def render_env(self):
        def convert_color(c):
            return tuple(map(lambda x: int(255*x), c))
        def coord_string(p):
            v = (abs(p[1]),
                 "N" if (p[1]<0.0) else "S",
                 abs(p[0]),
                 "W" if (p[0]<0.0) else "E")
            return "%.2f%s %.2f%s"%v

        w, h = self.env_surf.get_size()

        # scroll
        if (self.scroll!=None):
            scroll_to = pygame.mouse.get_pos()
            self.env_frame.left += SCROLL_SPEED*(scroll_to[0]-self.scroll[0])
            self.env_frame.top  += SCROLL_SPEED*(scroll_to[1]-self.scroll[1])

        # background color: #C1BC6C
        self.env_surf.fill((0xC1, 0xBC, 0x6C))

        # render food patches
        for f in filter(self.in_frame, self.env.food_patches):
            pygame.draw.circle(self.env_surf,
                               convert_color(f.get_color()),
                               self.convert_vec(f.pos),
                               self.convert_vec(f.size))
            pygame.draw.circle(self.env_surf,
                               (255, 0, 0) if (f in self.selection) else (0, 0, 0),
                               self.convert_vec(f.pos),
                               self.convert_vec(f.size),
                               OUTLINE_WIDTH)

        # render agents
        for a in self.env.agents:
            p = self.convert_vec(a.pos)
            if (self.show_hints):
                pygame.draw.line(self.env_surf, (119, 0, 0), (w//2, h//2), p, 1)

            if (self.in_frame(a)):
                c = convert_color(a.get_color())
                pygame.draw.circle(self.env_surf,
                                   c,
                                   p,
                                   self.convert_vec(a.size))
                pygame.draw.circle(self.env_surf,
                                   (255, 0, 0) if (a in self.selection) else (0, 0, 0),
                                   p,
                                   self.convert_vec(a.size),
                                   OUTLINE_WIDTH)
                pygame.draw.line(self.env_surf,
                                 (200, 200, 200),
                                 p,
                                 self.convert_vec((a.pos[0]+a.size*sin(a.angle), a.pos[1]+a.size*cos(a.angle))),
                                 OUTLINE_WIDTH)
                if (self.render_eyes):
                    rl = self.env_scale*self.env.vision_range
                    cl = convert_color(a.last_perception[5:8])
                    cr = convert_color(a.last_perception[8:11])
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

        # render walls
        for w in self.env.walls:
            pygame.draw.line(self.env_surf, convert_color(w.get_color()), self.convert_vec(w.a), self.convert_vec(w.b), 2)

        # draw selection rectangle
        if (self.selection_rect!=None):
            pygame.draw.rect(self.env_surf, (190, 190, 190), self.selection_rect, 1)

        # render coordinates
        self.env_surf.blit(self.font.render(coord_string(self.env_frame.center), True, (0, 0, 0)), (5, 5))

    def play_sound(self):
        pass # TODO
    
    def handle_event(self):
        e = pygame.event.wait()
        if (e.type==pygame.QUIT):
            return False
        elif (e.type==EVENT_RENDER or e.type==pygame.VIDEOEXPOSE):
            self.render()
            pygame.event.clear(e.type)
        elif (e.type==EVENT_SIMULATE and not self.pause):
            self.env.step()
            pygame.event.clear(e.type)
        elif (e.type==pygame.VIDEORESIZE):
            self.set_resolution(e.size)
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
                pygame.display.toggle_fullscreen()
            elif (e.key==pygame.K_s):
                self.controller.spawn(10)
            elif (e.key==pygame.K_SPACE):
                self.pause = not self.pause
            elif (e.key==pygame.K_e):
                self.render_eyes = not self.render_eyes
            elif (e.key==pygame.K_c):
                for o in self.selection:
                    if (type(o)==Agent):
                        self.controller.clone(o)
            elif (e.key==pygame.K_UP):
                self.env_frame.top -= 10.0
            elif (e.key==pygame.K_DOWN):
                self.env_frame.top += 10.0
            elif (e.key==pygame.K_LEFT):
                self.env_frame.left -= 10.0
            elif (e.key==pygame.K_RIGHT):
                self.env_frame.left += 10.0
            elif (e.key==pygame.K_PLUS or e.key==pygame.K_KP_PLUS):
                self.zoom(+1)
            elif (e.key==pygame.K_MINUS or e.key==pygame.K_KP_MINUS):
                self.zoom(-1)
            elif (e.key==pygame.K_DELETE):
                for o in self.selection:
                    try:
                        if (type(o)==Agent):
                            self.env.agents.remove(o)
                        elif (type(o)==FoodPatch):
                            self.env.food_patches.remove(o)
                    except ValueError:
                        pass
            elif (e.key==pygame.K_TAB):
                self.show_hints = not self.show_hints
        elif (e.type==pygame.MOUSEBUTTONDOWN):
            if (e.button==1):
                self.selection_rect = pygame.Rect(e.pos, (0, 0))
            elif (e.button==3):
                self.scroll = e.pos
            elif (e.button==4):
                self.zoom(+1)
            elif (e.button==5):
                self.zoom(-1)
        elif (e.type==pygame.MOUSEBUTTONUP):
            if (e.button==1 and self.selection_rect!=None):
                self.selection = list(filter(lambda o: self.selection_rect.collidepoint(self.convert_vec(o.pos)), self.env.agents+self.env.food_patches))
                self.selection_rect = None
            elif (e.button==3):
                self.scroll = None
        elif (e.type==pygame.MOUSEMOTION):
            if (self.selection_rect!=None): # FIXME
                self.selection_rect.width  = abs(self.selection_rect.left-e.pos[0])
                self.selection_rect.height = abs(self.selection_rect.top-e.pos[1])
                self.selection_rect.left   = min((self.selection_rect.left, e.pos[0]))
                self.selection_rect.top    = min((self.selection_rect.top, e.pos[1]))
                
        return True

    def zoom(self, pm):
        self.env_scale += pm*0.2

        m = 3.0 # TODO limit? (not 3.0)
        M = None
        if (m!=None and self.env_scale<m): 
            self.env_scale = m
        elif (M!=None and self.env_scale>M):
            self.env_scale = M

        w, h = self.env_surf.get_size()
        self.env_frame.w = w/self.env_scale
        self.env_frame.h = h/self.env_scale

    def screenshot(self, filename):
        pygame.image.save(self.screen, filename)

    def convert_vec(self, v):
        if (type(v)==tuple):
            return tuple(map(lambda x, o: int((x-o)*self.env_scale), v, self.env_frame))
        elif (type(v)==float or type(v)==int):
            return int(v*self.env_scale)
    
    def in_frame(self, a_or_f):
        return self.env_frame.collidepoint(a_or_f.pos)


__all__ = ["Video"]
