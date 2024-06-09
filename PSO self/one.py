import numpy as np

class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.pbest = position
        self.pbest_fitness = float('inf')

    def update_velocity(self, gbest, w, c1, c2):
        r1 = np.random.rand()
        r2 = np.random.rand()
        self.velocity = w * self.velocity + c1 * r1 * (self.pbest - self.position) + c2 * r2 * (gbest - self.position)

    def update_position(self):
        self.position += self.velocity

    def evaluate_fitness(self, fitness_function):
        self.fitness = fitness_function(self.position)

        if self.fitness < self.pbest_fitness:
            self.pbest = self.position
            self.pbest_fitness = self.fitness

class PSO:
    def __init__(self, fitness_function, num_particles, dim, minx, maxx, w, c1, c2):
        self.fitness_function = fitness_function
        self.num_particles = num_particles
        self.dim = dim
        self.minx = minx
        self.maxx = maxx
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.particles = []
        self.gbest = None
        self.gbest_fitness = float('inf')

    def initialize(self):
        for i in range(self.num_particles):
            position = np.random.uniform(self.minx, self.maxx, self.dim)
            velocity = np.zeros(self.dim)
            particle = Particle(position, velocity)
            self.particles.append(particle)

    def optimize(self, max_iter):
        for i in range(max_iter):
            for particle in self.particles:
                particle.evaluate_fitness(self.fitness_function)

                if particle.fitness < self.gbest_fitness:
                    self.gbest = particle.position
                    self.gbest_fitness = particle.fitness

            for particle in self.particles:
                particle.update_velocity(self.gbest, self.w, self.c1, self.c2)
                particle.update_position()

    def get_best_solution(self):
        return self.gbest

def rastrigin(x):
    n = len(x)
    return 10*n + sum([xi**2 - 10*np.cos(2*np.pi*xi) for xi in x])
def spherical(x):
    n = len(x)
    return 10*n + sum([xi**2 for xi in x])

if __name__ == '__main__':
    pso = PSO(rastrigin, 30, 2, -5.12, 5.12, 0.5, 1, 2)
    pso.initialize()
    pso.optimize(100)
    print(pso.get_best_solution())