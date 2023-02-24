import taichi as ti
ti.init(arch=ti.gpu)
import time
import math
import numpy as np

n_size = 1120#
h = 4.0

l_size = 1 / n_size
n_grid = int(n_size/h)
n_grid = n_grid - n_grid % 8
n = 1
while True:
  if 2**(n+1) <= n_grid:
    n += 1
  else:
    n_grid = 2**n
    break

n_grid =256#

n_particles = 1024 * 4 * 2
dt = 1e-3
p_vol, p_rho = (1/n_size * 2.0)**2, 1.0
p_mass = p_vol * p_rho
h = h/n_size#(3.0 * 32.0 * p_vol / (4.0 * n_particles * math.pi)) ** (1.0 / 3.0)
mu = 1.0e-2
B = 5.0#p_rho * 340.29 ** 2 / 7

w = 0.9

chunk = 2**8
print(n_size, n_grid,n_particles,chunk)

x = ti.Vector.field(2, ti.f32)
v = ti.Vector.field(2, ti.f32)
a = ti.Vector.field(2, ti.f32)
p = ti.field(ti.f32)
rho = ti.field(ti.f32)

p_rho0 = ti.field(ti.f32,shape=())

par = ti.field(ti.i32,shape=())

max_num_particles_per_cell = n_particles
pid = ti.field(ti.i32)

#ti.root.dense(ti.ij,4).pointer(ti.ij,n_grid // 8).dense(ti.ij, 8).dynamic(ti.l, max_num_particles_per_cell, 512).place(pid)
pidroot = ti.root.pointer(ti.ij, n_grid // 8).pointer(ti.ij, 8)
pidroot.dynamic(ti.l, max_num_particles_per_cell, chunk).place(pid)
ti.root.dense(ti.l, n_particles).place(x, v, a, p, rho)

support = 1

@ti.kernel
def initialize_particle_grid():
  for p in x:
    ipos = ti.floor(x[p] * n_grid).cast(ti.i32)
    for i in range(-support, support + 1):
      for j in range(-support, support + 1):
        offset = ti.Vector([i, j])
        box_ipos = ipos + offset
        ti.append(pid.parent(), box_ipos, p)

@ti.func
def nWspi(r):
  t = ti.Vector([0.0, 0.0])
  if r.norm() > 0.0 and r.norm() <= h:
    t = (-30.0 / (math.pi * h ** 5))  * ((h - r.norm()) ** 2) * (r / r.norm())
  #2D...-30 / (math.pi * h ** 5), 3D...-45 / (math.pi * h ** 6)
  return  t

@ti.func
def Wpol(r):
  t = 0.0
  if r.norm() >= 0.0 and r.norm() <= h:
    t = (4.0 / (math.pi * h ** 8)) * ((h ** 2 - r.norm() ** 2) ** 3)
  #2D...4 / (math.pi * h ** 8), 3D...315 / (64 * math.pi * h ** 9)
  return t

@ti.func
def n2Wvis(r):
  t = 0.0
  if r.norm() > 0.0 and r.norm() <= h:
    t = (20 / (3 * math.pi * h ** 5))  * (h - r.norm())
  #2D...20 / (3 * math.pi * h ** 5), 3D...45 / (2 * math.pi * h ** 6)
  return t

@ti.kernel
def substep():
  for i in x:
    rho[i] = 0.0
  for i in x:
    ipos = ti.floor(x[i] * n_grid).cast(ti.i32)
    num_particles = ti.length(pid.parent(), ipos)
    for k in range(num_particles):#num_particles
      j = pid[ipos[0], ipos[1], k]
      r = x[i] - x[j]
      if r.norm() <= h:
        rho[i] += p_mass * Wpol(r)
    #p[i] = B * ((rho[i] / p_rho) ** 7 - 1)       
    p[i] = B * (rho[i] - p_rho0[None])
  for i in x:
    fv = ti.Vector([0.0, 0.0])
    fp = ti.Vector([0.0, 0.0])
    ipos = ti.floor(x[i] * n_grid).cast(ti.i32)
    num_particles = ti.length(pid.parent(), ipos)
    for k in range(num_particles):
      j = pid[ipos[0], ipos[1], k]
      if i != j:
        r = x[i] - x[j]
        if r.norm() <= h:
          fv += p_mass * ((v[j] - v[i]) / rho[j]) * n2Wvis(r)
          fp += p_mass * ((p[i] + p[j]) / (2 * rho[j])) * nWspi(r)
    fv = mu * fv
    fp = -1 * fp
    f = fv + fp
    a[i] = f
  for i in x:
    v[i] += (a[i] + ti.Vector([0.0, -9.8])) * dt
    for j in ti.static(range(2)):
      if x[i][j] + v[i][j] * dt - l_size * 3 <= 0.0 or x[i][j] + v[i][j] * dt + l_size * 3 >= 1.0:
        v[i][j] = 0.0
    x[i] += v[i] * dt
    par[None] = ti.length(pid.parent(), ti.Vector([int(n_grid/2), 3]).cast(ti.i32))

@ti.kernel
def setup():
  for i in x:
    n_size2 = int(n_size / 1.5)
    #x[i] = [ti.random() * 0.3 + 0.05, ti.random() * 0.3 + 0.05]
    x[i]=[(i % (n_size2 * w)) / n_size2 + 0.5 - w / 2, int(i / (n_size2  *  w)) / n_size2 + 0.2 - 0.2 / 2]
    v[i] = [0.0, 0.0]
  l = 2 / n_size
  n = int(h / l + 1) + 1
  for i in range(-n,n+1):
    for j in range(-n,n+1):
      p_rho0[None] += p_mass * Wpol(ti.Vector([i*l,j*l]))
setup()

gui_t = True
debug = False

if gui_t == True:
  gui = ti.GUI("SPH", res=512, background_color=0xEEEEEE)
for frame in range(1000):
  time_s = time.perf_counter()
  for s in range(int((1/(60 * 8)) // dt)):
    pidroot.deactivate_all()
    initialize_particle_grid()
    substep()
    if debug == True:
      print(par[None])
      gui.circles(x.to_numpy(), radius=2.0, color=0x068587)
      gui.show()
  #time_e=time.perf_counter()
  if gui_t == True:
    gui.circles(x.to_numpy(), radius=2.0, color=0x068587)
    time_e=time.perf_counter()
    gui.show()
    #gui.show(f'sph_m/{frame:04d}.png')
  if frame == 0:
    pass
  elif frame != 1:
    tim=max(tim,time_e-time_s)
    tim_a = (tim_a + time_e-time_s) / 2
  else:
    tim = time_e-time_s
    tim_a = tim

fps_min = 1 / tim
fps_avg = 1 / tim_a
print(1 / tim)
print(1 / tim_a)

if gui_t == True:
  gui.core.should_close = True
