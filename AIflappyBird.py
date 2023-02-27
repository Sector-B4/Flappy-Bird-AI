from pygame.locals import *
import pygame
import os
import sys
import neat
import time
import random

pygame.font.init()

GEN = 0

WIN_WIDTH = 500
WIN_HEIGHT = 766

BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))]
PIPE = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
GROUND = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
BG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))

STAT_FONT = pygame.font.SysFont("Comicsans", 50)
GEN_FONT = pygame.font.SysFont("Comicsans", 30)


class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROL_VEL = 20
    ANIMATION_T = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1

        d = self.vel*self.tick_count + 1.5*self.tick_count**2

        if d >= 16:
            d = 16

        if d < 0:
            d -= 2

        self.y = self.y + d

        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
            else:
                if self.tilt > -90:
                    self.tilt -= self.ROL_VEL

    def draw(self, win):
        self.img_count += 1

        if self.img_count < self.ANIMATION_T:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_T*2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_T*3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_T*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_T*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_T*2

        rotated_img = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_img.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_img, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Pipe:
    GAP = 200
    VEL = 5

    def __init__(self, x):
        self. x =x
        self.height = 0
        self.gap = 100
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE, False, True)
        self.PIP_BOTTOM = PIPE
        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIP_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIP_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        bp = bird_mask.overlap(bottom_mask, bottom_offset)
        tp = bird_mask.overlap(top_mask, top_offset)

        if tp or bp:
            return True
        return False

class Ground:
    VEL = 5
    WIDTH = GROUND.get_width()
    IMG = GROUND

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))

def draw_win(win, birds, pipes, ground, score, gen):
    win.blit(BG, (0, 0))
    for p in pipes:
        p.draw(win)

    txt = STAT_FONT.render("Score: " + str(score), 1,(255,255,255))
    win.blit(txt, (WIN_WIDTH - 150 - txt.get_width(), 90))

    txt = GEN_FONT.render("Generation: " + str(gen), 1,(255,255,255))
    win.blit(txt, (10, 10))

    ground.draw(win)
    for bird in birds:
        bird.draw(win)
    pygame.display.update()

def main(genomes, config):
    global GEN
    GEN += 1
    nets = []
    ge = []
    birds = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        g.fitness = 0
        ge.append(g)

    ground = Ground(700)
    pipes = [Pipe(700)]
    score = 0

    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()



    run = True
    while run and len(birds) > 0:
        clock.tick(32)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.05

            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

            if output[0] > 0.5:
                bird.jump()

        ground.move()

        rem = []
        add_pipe = False
        for p in pipes:
            p.move()
            for x, bird in enumerate(birds):
                if p.collide(bird):
                    #ge[x].fitness -= 1
                    nets.pop(x)
                    ge.pop(x)
                    birds.pop(x)

            if p.x + p.PIPE_TOP.get_width() < 0:
                rem.append(p)

            if not p.passed and p.x < bird.x:
                    p.passed = True
                    add_pipe = True

        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 0.5
            pipes.append(Pipe(600))
        for r in rem:
            pipes.remove(r)

        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() -10 >= 700 or bird.y < -50:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        draw_win(win, birds, pipes, ground, score, GEN)



def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    pp = neat.Population(config)

    pp.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pp.add_reporter(stats)

    winner = pp.run(main, 50)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run(config_path)



