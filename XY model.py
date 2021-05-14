import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

# XY Model

particle_width = 40
particle_height = particle_width
type = ti.f32
num_particle = particle_width * particle_height
end = np.zeros((num_particle, 2))
centre = np.zeros((num_particle, 2))

dx = 1 / particle_width
dy = 1 / particle_height

old_hamiltonian = ti.field(type, shape=())
new_hamiltonian = ti.field(type, shape=())
temperature = 4.0

dim = 2

a = ti.Vector(dim, type, shape=(particle_width, particle_height))

b = ti.Vector(dim, type, shape=(particle_width, particle_height))
B = ti.Vector(dim, type, shape=())

magnetic_field = 1000
J = 0.1

@ti.func
def round(i):
    result = i
    if i < 0:
        result = particle_width + i
    elif i >= particle_width:
        result = i - particle_width
    return result

@ti.func
def sub_hamiltonian(i, j, vec):
    rst = 0.0
    rst += (-B[None].dot(vec[i, j]))
    rst += -J * vec[i, j].dot(vec[round(i - 1), j])
    rst += -J * vec[i, j].dot(vec[round(i + 1), j])
    rst += -J * vec[i, j].dot(vec[i, round(j + 1)])
    rst += -J * vec[i, j].dot(vec[i, round(j - 1)])

    return rst

@ti.func
def Hamiltonian(vec) -> type:
    result = 0.0
    for i in range(particle_width):
        for j in range(particle_height):
            result += sub_hamiltonian(i, j, vec)
    return result

@ti.func
def normalize(vec):
    for i in range(particle_width):
        for j in range(particle_height):
            vec[i, j] = vec[i, j].normalized()

@ti.kernel
def init():
    for i in range(particle_width):
        for j in range(particle_height):
            a[i, j] = ti.Vector([ti.random(type) - 0.5, ti.random(type) - 0.5])
    normalize(a)
    old_hamiltonian[None] = Hamiltonian(a)
    B[None] = ti.Vector([magnetic_field, 0])


@ti.func
def create_new_form():
    for i in range(particle_width):
        for j in range(particle_height):
            b[i, j] = a[i, j] + 0.2 * ti.Vector([ti.random(type) - 0.5, ti.random(type) - 0.5])
    normalize(b)
    new_hamiltonian[None] = Hamiltonian(b)


@ti.kernel
def step() -> ti.i32:
    create_new_form()
    rst = 0
    if (ti.random(type) < ti.exp(-(new_hamiltonian[None] - old_hamiltonian[None]) / temperature)):
        # if (new_hamiltonian[None] < old_hamiltonian[None]):
        for i in range(particle_width):
            for j in range(particle_height):
                a[i, j] = b[i, j]
        print("T={}  ".format(temperature), end="")
        print(new_hamiltonian)
        old_hamiltonian[None] = new_hamiltonian[None]
        create_new_form()
        rst = 1
    return rst


if __name__ == "__main__":
    init()
    gui = ti.GUI('XY Model', (1000, 1000), background_color=0xFFFFFF)
    frame = 0
    for i in range(particle_width):
        for j in range(particle_height):
            centre[i * particle_width + j, ...] = np.array([i * dx, j * dy]) + dx / 2

    while True:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                break
            if gui.event.key == 'w': temperature += 0.1
            if gui.event.key == 's': temperature -= 0.1
        if (step() == 1):
            end = centre + a.to_numpy().reshape(num_particle, 2) / (3 * particle_width)
            gui.lines(centre, end, radius=1.5, color=0xFF0000)
            end = centre - a.to_numpy().reshape(num_particle, 2) / (3 * particle_width)

            gui.lines(centre, end, radius=1.5, color=0x0000FF)

            gui.show()
