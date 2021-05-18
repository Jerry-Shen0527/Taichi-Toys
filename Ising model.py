from numpy.core.fromnumeric import shape
import taichi as ti
import numpy as np
from taichi.lang.ops import sqrt

ti.init(arch=ti.gpu)

# XY Model

particle_width = 20
particle_height = particle_width
type = ti.f32
num_particle = particle_width * particle_height
end = np.zeros((num_particle, 2))
centre = np.zeros((num_particle, 2))

dx = 1 / particle_width
dy = 1 / particle_height

old_hamiltonian = ti.field(type, shape=())
new_hamiltonian = ti.field(type, shape=())
Magnitude = ti.field(type, shape=())
temperature = ti.field(type, shape=())

dim = 2

a = ti.Vector(dim, type, shape=(particle_width, particle_height))

b = ti.Vector(dim, type, shape=(particle_width, particle_height))
B = ti.Vector(dim, type, shape=())
index = ti.Vector(dim, type, shape=())

magnetic_field = 1
J = 1

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
def M(vec) -> type:
    result = 0.0
    for i in range(particle_width):
        for j in range(particle_height):
            result += vec[i,j][1]
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
            a[i, j] = ti.Vector([0.0,ti.random(type)-0.5])
            b[i, j] = a[i, j]
    normalize(a)
    B[None] = ti.Vector([0.0,magnetic_field])
    old_hamiltonian[None] = Hamiltonian(a)
    Magnitude[None]=M(a)



@ti.func
def create_new_form():
    for i in range(particle_width):
        for j in range(particle_height):
            b[i, j] = a[i, j]
            if(ti.random(type) >0.98):
                b[i, j] = a[i, j]*(-1)
    normalize(b)
    new_hamiltonian[None] = Hamiltonian(b)
# @ti.func
# def create_new_form(i,j):
#     value=ti.random(type) - 0.5
#     if value!=0:
#         b[i, j] = a[i, j]*value/abs(value)
#     new_hamiltonian[None] = Hamiltonian(b)

@ti.kernel
def step() -> ti.i32:
    create_new_form()
    # create_new_form(index[None][0],index[None][1])
    index[None][0]=round(index[None][0]+1)
    if(index[None][0]==0):
        index[None][1]=round(index[None][1]+1)
    rst = 0
    # print(ti.exp(-(new_hamiltonian[None] - old_hamiltonian[None]) / temperature))
    if (ti.random(type) < ti.exp(-(new_hamiltonian[None] - old_hamiltonian[None]) / temperature)):
        # if (new_hamiltonian[None] < old_hamiltonian[None]):
        for i in range(particle_width):
            for j in range(particle_height):
                a[i, j] = b[i, j]
        old_hamiltonian[None] = new_hamiltonian[None]
        Magnitude[None]=M(a)
        rst = 1
    else:
        rst=0


    return rst


if __name__ == "__main__":

    temperature[None]=50
    init()
    gui = ti.GUI('XY Model', (1000, 1000), background_color=0xFFFFFF)
    frame = 0
    for i in range(particle_width):
        for j in range(particle_height):
            centre[i * particle_width + j, ...] = np.array([i * dx, j * dy]) + dx / 2
    av=0.0
    count=0

    shapecount=0

    while True:
        av=count*av+Magnitude[None]
        count+=1
        av=av/count

        
        # if gui.get_event(ti.GUI.PRESS):
        if shapecount==30:
            # if gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            #     break
            # if gui.event.key == 'w': temperature[None] += 0.1*temperature[None]
            # if gui.event.key == 's': temperature[None] -= 0.1*temperature[None]
            temperature[None] -= 0.1*temperature[None]
            print('{',end='')
            print(temperature,end=",")
            print(av,end='},')
            av=0
            count=0
            shapecount=0

        
        if (step() == 1):
            shapecount=shapecount+1
            rst=a.to_numpy().reshape(num_particle, 2) / (3 * particle_width)
            filter1=rst[:,1]>0
            filter2=rst[:,1]<0
            gui.circles(centre[filter1], color = 0xFF0000, radius = 10*50/particle_width)
            gui.circles(centre[filter2], color = 0x0000FF, radius = 10*50/particle_width)
            gui.show()
        # else:
        #     rst=b.to_numpy().reshape(num_particle, 2) / (3 * particle_width)
        #     filter1=rst[:,1]>0
        #     filter2=rst[:,1]<0
        #     gui.circles(centre[filter1], color = 0xFFFF00, radius = 10*50/particle_width)
        #     gui.circles(centre[filter2], color = 0x00FFFF, radius = 10*50/particle_width)
        #     gui.show()
        # print(temperature,end=",")
        # print(av)

            # gui.lines(centre, end, radius=4.0, color=0x0000FF)
            # gui.triangles(centre+end_rt, end, centre+end_rt_neg, color = 0x000000)

            

