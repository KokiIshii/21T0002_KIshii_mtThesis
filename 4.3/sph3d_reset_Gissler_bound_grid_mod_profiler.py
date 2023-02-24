import taichi as ti
import numpy as np
import math
import time
import sys

sys_value = sys.argv #["", "grid-division", "hierarchical-data-structure", "leaf-size", "fluid-particles"]
#sys_value = ["", "136", "bitmasked_bitmasked", "2", "11532"]

ti.init(arch=ti.gpu, kernel_profiler=True)

eps = 1e-4
inf = 1e10

pCFL = ti.field(ti.f32, shape=())

res = 720, 720
aspect_ratio = res[0] / res[1]
num_part = int(sys_value[4])
color_buffer = ti.Vector.field(3, ti.f32)

camera_pos = ti.Vector([0.5, 0.3, 3.3])
light_direction = [1.2, 0.3, 0.9]
light_color = [1.0, 1.0, 1.0]

inv_dx = int(sys_value[1])
l_size = 1 / inv_dx

#fluid
p_vol, p_rho = (1 / inv_dx * 2.0)**3, 1.0
p_mass = p_vol * p_rho
h = 4.0
mu = 1.0e-2
B = 5.0

n_grid = int(inv_dx/h)
d_grid = n_grid
n = 1
while True:
  if 2**n < d_grid:
    n += 1
  else:
    d_grid = 2**n
    break

h = h / inv_dx

sphere_radius = 1.0 / inv_dx

particle_x = ti.Vector.field(3, ti.f32)
particle_v = ti.Vector.field(3, ti.f32)
particle_a = ti.Vector.field(3, ti.f32)
particle_p = ti.field(ti.f32)
particle_rho = ti.field(ti.f32)
particle_color = ti.Vector.field(3, ti.f32)

fov = 0.23

w = 0.49
dt = 1e-3
chunk = 2**8
particle_rho0 = ti.field(ti.f32, shape=())
num_particles = ti.field(ti.f32, shape=())

#rigid
rp_vol, rp_rho = (1/inv_dx * 2.0)**3, 1.0
rp_mass = rp_vol * rp_rho

rad = 0.1
rn_particles = 1146
sigma = mu
cs = 1000

rx = ti.Vector.field(3, ti.f32)
rx0 = ti.Vector.field(3, ti.f32)
rv = ti.Vector.field(3, ti.f32)
rp = ti.field(ti.f32)
rc = ti.Vector.field(3, ti.f32)

rrho = ti.field(ti.f32)
rrho0 = 1

Vb = ti.field(ti.f32)

Rx = ti.Vector.field(3, ti.f32, shape=())
Rv = ti.Vector.field(3, ti.f32, shape=())
Rom = ti.Vector.field(3, ti.f32, shape=())
Rf = ti.Vector.field(3, ti.f32, shape=())
Rtau = ti.Vector.field(3, ti.f32, shape=())
Rthe = ti.Vector.field(3, ti.f32, shape=())

#boundary
bound = [[0.25, 0, 0.25], [0.75, 1, 0.75]]
bxx = abs(np.array(bound[1]) - np.array(bound[0])) * [1.1, 1.0, 1.1]
bxxl = (bxx/(l_size*2))
bx = ti.Vector.field(3, ti.f32)
brho = ti.field(ti.f32)
brho0 = 10000
bn_particles = int(bxxl[0]*bxxl[1]*2 + bxxl[1]*bxxl[2]*2 + bxxl[2]*bxxl[0]*2 - bxxl[0]*4 - bxxl[1]*4 - bxxl[2]*4 + 8)
ti.root.dense(ti.l, bn_particles).place(bx,brho)

#data
def data_structure(structure, grid, ggg):
  lis = structure.split("_")
  pidroot = ti.root
  floor = len(lis)

  for i in range(floor):
    if i == 0:
      if lis[0] == "bitmasked":
        pidroot = pidroot.bitmasked(ti.ijk, grid // ggg**(floor-1))
      elif lis[0] == "pointer":
        pidroot = pidroot.pointer(ti.ijk, grid // ggg**(floor-1))
    else:
      if lis[i] == "bitmasked":
        pidroot = pidroot.bitmasked(ti.ijk, ggg)
      elif lis[i] == "pointer":
        pidroot = pidroot.pointer(ti.ijk, ggg)
  return pidroot
  

ti.root.dense(ti.ij, (res[0] // 8, res[1] // 8)).dense(ti.ij, 8).place(color_buffer)

max_num_particles_per_cell = num_part
pid = ti.field(ti.i32)
rpid = ti.field(ti.i32)
ggg = int(sys_value[3])
structure = sys_value[2]

pidroot = data_structure(structure, d_grid, ggg)
pidroot.dynamic(ti.l, max_num_particles_per_cell, chunk).place(pid)

pidroot_r = data_structure(structure, d_grid, ggg)
pidroot_r.dynamic(ti.l, max_num_particles_per_cell, chunk).place(rpid)

bpid = ti.field(ti.i32)
ti.root.pointer(ti.ijk, d_grid // 8).pointer(ti.ijk, 8).dynamic(ti.l, max_num_particles_per_cell, chunk).place(bpid)

ti.root.dense(ti.l, num_part).place(particle_x, particle_v, particle_a, particle_p, particle_rho, particle_color)
ti.root.dense(ti.l, rn_particles).place(rx, rx0, rv, rp, rc, rrho, Vb)

#render
@ti.func
def intersect_sphere(pos, d, center, radius):
  T = pos - center
  A = 1.0
  B = 2.0 * T.dot(d)
  C = T.dot(T) - radius * radius
  delta = B * B - 4.0 * A * C
  dist = inf
  hit_pos = ti.Vector([0.0, 0.0, 0.0])

  if delta > -1e-4:
    delta = ti.max(delta, 0)
    sdelta = ti.sqrt(delta)
    ratio = 0.5 / A
    ret1 = ratio * (-B - sdelta)
    dist = ret1
    hit_pos = pos + d * dist

  return dist, hit_pos

@ti.func
def dda_particle(eye_pos, d):
  hit_pos = ti.Vector([0.0, 0.0, 0.0])
  normal = ti.Vector([0.0, 0.0, 0.0])
  c = ti.Vector([0.0, 0.0, 0.0])

  closest_intersection = inf

  x = ti.Vector([0.0, 0.0, 0.0])
  color = ti.Vector([0.0, 0.0, 0.0])
  for k in range(num_particles[None]):
    if k < num_part:
      x = particle_x[k]
      color = particle_color[k]
    elif k < num_part + rn_particles:
      j = k - num_part
      x = rx[j]
      color = rc[j]
    else:
      j = k - (num_part + rn_particles)
      x = bx[j]
      color = ti.Vector([1.0, 1.0, 1.0])
    dist, _ = intersect_sphere(eye_pos, d, x, sphere_radius)
    if dist < closest_intersection and dist > 0:
      hit_pos = eye_pos + dist * d
      closest_intersection = dist
      normal = (hit_pos - x).normalized()
      c = color
      
  return closest_intersection, normal, c

@ti.kernel
def render():
  for u, v in color_buffer:
    pos = camera_pos
    d = ti.Vector([2 * fov * u / res[1] - fov * aspect_ratio,
                   2 * fov * v / res[1] - fov,
                   -1.0])
    d = d.normalized()
    
    contrib = ti.Vector([0.3, 0.3, 0.3])
    throughput = ti.Vector([1.0, 1.0, 1.0])

    closest, normal, c = dda_particle(pos, d)
    if normal.norm() != 0:
      throughput *= c

      direct = (ti.Vector(light_direction)).normalized()
      dot = direct.dot(normal)
      if dot > 0:
        contrib += throughput * ti.Vector(light_color) * dot

    if closest == inf:
      contrib = ti.Vector([0.7, 0.7, 0.7])
        
    color_buffer[u, v] = contrib

@ti.kernel
def copy(img: ti.ext_arr()):
  for i, j in color_buffer:
    for c in ti.static(range(3)):
      img[i, j, c] = color_buffer[i, j][c]

#kernel
@ti.func
def nWspi(r):
  t = ti.Vector([0.0, 0.0, 0.0])
  rn = r.norm()
  if r.norm() > 0.0 and r.norm() <= h:
    t = (-45 / (math.pi * h ** 6))  * ((h - r.norm()) ** 2) * (r / r.norm())
  #2D...-30 / (math.pi * h ** 5), 3D...-45 / (math.pi * h ** 6)
  return  t

@ti.func
def Wpol(r):
  t = 0.0
  rn = r.norm()
  if r.norm() >= 0.0 and r.norm() <= h:
    t = (315 / (64 * math.pi * h ** 9)) * ((h ** 2 - r.norm() ** 2) ** 3)
  #2D...4 / (math.pi * h ** 8), 3D...315 / (64 * math.pi * h ** 9)
  return t

@ti.func
def n2Wvis(r):
  t = 0.0
  rn = r.norm()
  if r.norm() > 0.0 and r.norm() <= h:
    t = (45 / (math.pi * h ** 6))  * (h - r.norm())
  #2D...20 / (3 * math.pi * h ** 5), 3D...45 / (math.pi * h ** 6)
  return t

@ti.func
def nWvis(r):
  t = ti.Vector([0.0, 0.0, 0.0])
  rn = r.norm()
  if r.norm() > 0.0 and r.norm() <= h:
    t = (15 / (2 * math.pi * h ** 5))  * (-3 * r.norm() / (2 * h) + 2 - h**3 / (2 * r.norm() ** 3)) * r
  #2D...10 / (3 * math.pi * h ** 4), 3D...15 / (2 * math.pi * h ** 5)
  return  t

#main
support = 1
@ti.kernel
def initialize_particle_grid():
  for p in particle_x:
    ipos = ti.floor(particle_x[p] * n_grid).cast(ti.i32)
    for i in range(-support, support + 1):
      for j in range(-support, support + 1):
        for k in range(-support, support + 1):
          offset = ti.Vector([i, j, k])
          box_ipos = ipos + offsetsph3d_reset_Gissler_bound_grid_mod_profiler.py
          ti.append(pid.parent(), box_ipos, p)
  for p in rx:
    ipos = ti.floor(rx[p] * n_grid).cast(ti.i32)
    for i in range(-support, support + 1):
      for j in range(-support, support + 1):
        for k in range(-support, support + 1):
          offset = ti.Vector([i, j, k])
          box_ipos = ipos + offset
          ti.append(rpid.parent(), box_ipos, p)

@ti.kernel
def cal_rho():
  for i in particle_x:
    particle_rho[i] = 0.0
  #fluid_rho
  for i in particle_x:
    ipos = ti.floor(particle_x[i] * n_grid).cast(ti.i32)
    n_particles = ti.length(pid.parent(), ipos)
    for k in range(n_particles):
      j = pid[ipos[0], ipos[1], ipos[2], k]
      r = particle_x[i] - particle_x[j]
      particle_rho[i] += p_mass * Wpol(r)
    n_particles = ti.length(rpid.parent(), ipos)
    for k in range(n_particles):
      j = rpid[ipos[0], ipos[1], ipos[2], k]
      r = particle_x[i] - rx[j]
      particle_rho[i] += (rrho[j] * Vb[j]) * Wpol(r)
    particle_p[i] = B * (particle_rho[i] - particle_rho0[None])

@ti.kernel
def cal_fluid_f():
  #fluid_f
  for i in particle_x:
    fv = ti.Vector([0.0, 0.0, 0.0])
    fp = ti.Vector([0.0, 0.0, 0.0])
    fbv = ti.Vector([0.0, 0.0, 0.0])
    fbp = ti.Vector([0.0, 0.0, 0.0])
    ipos = ti.floor(particle_x[i] * n_grid).cast(ti.i32)
    n_particles = ti.length(pid.parent(), ipos)
    for k in range(n_particles):
      j = pid[ipos[0], ipos[1], ipos[2], k]
      r = particle_x[i] - particle_x[j]
      fv += p_mass * ((particle_v[j] - particle_v[i]) / particle_rho[j]) * n2Wvis(r)
      fp += p_mass * ((particle_p[i] + particle_p[j]) / (2 * particle_rho[j])) * nWspi(r)
    n_particles = ti.length(rpid.parent(), ipos)
    for k in range(n_particles):
      j = rpid[ipos[0], ipos[1], ipos[2], k]
      r = particle_x[i] - rx[j]
      fbv += - p_mass * rrho[j] * (-1 * (sigma * h * cs) / (particle_rho[i] + rrho[j])) * (min((particle_v[i] - rv[j]).dot(particle_x[i] - rx[j]), 0) / ((particle_x[i] - rx[j]).norm()**2 + 0.01 * (h**2))) * nWvis(r)
      fbp += - p_mass * rrho[j] * (particle_p[j] / (particle_rho[i] * rrho[j])) * nWspi(r)
    n_particles = ti.length(bpid.parent(), ipos)
    for k in range(n_particles):
      j = bpid[ipos[0], ipos[1], ipos[2], k]
      r = particle_x[i] - bx[j]
      fbv += - p_mass * brho[j] * (-1 * (sigma * h * cs) / (particle_rho[i] + brho[j])) * (min((particle_v[i] - 0).dot(particle_x[i] - bx[j]), 0) / ((particle_x[i] - bx[j]).norm()**2 + 0.01 * (h**2))) * nWvis(r)
      fbp += - p_mass * brho[j] * (particle_p[j] / (particle_rho[i] * brho[j])) * nWspi(r)
    fv = mu * fv
    fp = -1 * fp
    f = fv + fp + fbp + fbv
    particle_a[i] = f / p_rho

@ti.kernel
def cal_rigid_f():
  #rigid_f
  Rf[None] = [0.0, 0.0, 0.0]
  Rtau[None] = [0.0, 0.0, 0.0]
  for k in range(rn_particles):
    fbv = ti.Vector([0.0, 0.0, 0.0])
    fbp = ti.Vector([0.0, 0.0, 0.0])
    ipos = ti.floor(rx[k] * n_grid).cast(ti.i32)
    num_particles = ti.length(pid.parent(), ipos)
    for i in range(num_particles):
      j = pid[ipos[0], ipos[1], ipos[2], i]
      r = particle_x[j] - rx[k]
      fbv += p_mass * rrho[k] * (-1 * (sigma * h * cs) / (particle_rho[j] + rrho[k])) * (min((particle_v[j] - rv[k]).dot(particle_x[j] - rx[k]), 0) / ((particle_x[j] - rx[k]).norm()**2 + 0.01 * (h**2))) * nWvis(r)
      fbp += p_mass * rrho[k] * (particle_p[j] / (particle_rho[j] *rrho[k])) * nWspi(r)
    Rf[None] += fbv + fbp
    Rtau[None] += (rx[k] - Rx[None]).cross(fbv + fbp)

@ti.kernel
def cal_fluid_xv():
  #fluid_xv
  for i in particle_x:
    particle_v[i] += (particle_a[i] + ti.Vector([0.0, -9.8, 0.0])) * dt
    if particle_x[i][0] + particle_v[i][0] * dt - l_size * (inv_dx/4)  <= 0.0 or particle_x[i][0] + particle_v[i][0] * dt + l_size * (inv_dx/4) >= 1.0:
      particle_v[i][0] = 0.0
    if particle_x[i][1] + particle_v[i][1] * dt - l_size * 2 <= 0.0:# or particle_x[i][1] + particle_v[i][1] * dt >= 1.0:
      particle_v[i][1] = 0.0
    if particle_x[i][2] + particle_v[i][2] * dt - l_size * (inv_dx/4) <= 0.0 or particle_x[i][2] + particle_v[i][2] * dt + l_size * (inv_dx/4) >= 1.0:
      particle_v[i][2] = 0.0
    particle_x[i] += particle_v[i] * dt

@ti.kernel
def cal_rigid_xv():
  #rigid_xv
  Rv[None] += (Rf[None] / (rrho0 * (rad * inv_dx) ** 2) + ti.Vector([0.0, -9.8, 0.0])) * dt#
  if Rx[None][0] + Rv[None][0] * dt - l_size * (inv_dx / 4) - rad <= 0.0 or Rx[None][0] + Rv[None][0] * dt + l_size * (inv_dx / 4) + rad >= 1.0:
    Rv[None][0] = 0.0
  if Rx[None][1] + Rv[None][1] * dt - l_size * 0 - rad <= 0.0: #or Rx[None][1] + Rv[None][1] * dt + l_size * 3 + rad >= 1.0:
    Rv[None][1] = 0.0
  if Rx[None][2] + Rv[None][2] * dt - l_size * (inv_dx / 4) - rad <= 0.0 or Rx[None][2] + Rv[None][2] * dt + l_size * (inv_dx / 4) + rad >= 1.0:
    Rv[None][2] = 0.0
  Rom[None] += (Rtau[None] / (rp_rho * (rad * inv_dx) ** 2)) * dt
  Rx[None] += Rv[None] * dt
  Rthe[None] += Rom[None] * dt
  R = ti.Matrix([[ti.cos(Rthe[None][2]) * ti.cos(Rthe[None][1]), ti.cos(Rthe[None][2]) * ti.sin(Rthe[None][1]) * ti.sin(Rthe[None][0]) - ti.sin(Rthe[None][2]) * ti.cos(Rthe[None][0]), ti.cos(Rthe[None][2]) * ti.sin(Rthe[None][1]) * ti.cos(Rthe[None][0]) + ti.sin(Rthe[None][2]) * ti.sin(Rthe[None][0])],
                 [ti.sin(Rthe[None][2]) * ti.cos(Rthe[None][1]), ti.sin(Rthe[None][2]) * ti.sin(Rthe[None][1]) * ti.sin(Rthe[None][0]) + ti.cos(Rthe[None][2]) * ti.cos(Rthe[None][0]), ti.sin(Rthe[None][2]) * ti.sin(Rthe[None][1]) * ti.cos(Rthe[None][0]) - ti.cos(Rthe[None][2]) * ti.sin(Rthe[None][0])],
                 [-ti.sin(Rthe[None][1]), ti.cos(Rthe[None][1]) * ti.sin(Rthe[None][0]), ti.cos(Rthe[None][1]) * ti.cos(Rthe[None][0])]])
  for k in rx:
    rx[k] = Rx[None] + R @ rx0[k]
    rv[k] = Rv[None] + Rthe[None].cross(rx0[k])

def substep():
  cal_rho()
  cal_fluid_f()
  cal_rigid_f()
  cal_fluid_xv()
  cal_rigid_xv()

@ti.kernel
def dCFL():  
    for i in range(num_part):
      for j in ti.static(range(3)):
        if 1 < particle_v[i][j] * dt / l_size:
          pCFL[None] = 1
  
def main():
  num_particles[None] = num_part + rn_particles
  pCFL[None] = 0
  
  @ti.kernel
  def setup():
    for i in range(num_part):
      n_size = int(inv_dx / 2.0)
      offset = [0.5, 0.4, 0.5]
      particle_x[i] = [(i % (n_size * w)) / n_size + offset[0]- w / 2
                       , int(i / ((n_size * w) ** 2)) / n_size + offset[1]
                       , (int(i / (n_size * w)) % (n_size * w)) / n_size + offset[2] - w / 2]
      particle_v[i] = [0.0, 0.0, 0.0]
      particle_color[i] = [0.5, 0.5, 1.0]
    l = 2 / inv_dx
    n = int(h / l + 1) +1
    for i in range(-n, n+1):
      for j in range(-n, n+1):
        for k in range(-n, n+1):
          particle_rho0[None] += p_mass * Wpol(ti.Vector([i*l, j*l, k*l]))
  setup()

  Rx[None] = [0.5, 0.2, 0.5]
  rx0[0] = [0, 1*rad, 0]
  rx0[rn_particles-1] = [0, -1*rad, 0]
  ph = 0
  for k in range(2, rn_particles):
      hk = 2 * (k-1) / (rn_particles - 1) - 1
      th =  ti.acos(hk)
      ph =  ph + (3.6 / ti.sqrt(rn_particles)) * (1 / ti.sqrt(1 - hk**2))
      rx0[k-1] = [rad * ti.sin(th) * ti.cos(ph), rad * ti.cos(th), rad * ti.sin(th) * ti.sin(ph)]

  @ti.kernel
  def setup_rigid():
    for i in range(rn_particles):
      rx[i] = Rx[None] + rx0[i]
      rc[i] = [1.0, 0.5, 0.5]
    for i in range(rn_particles):
      delta = 0.0
      for j in range(rn_particles):
        if i != j:
          r = rx[i] - rx[j]
          if r.norm() < h:
            delta += Wpol(r)
      Vb[i] = 0.7 / delta
      rrho[i] = 0.0
      for j in range(rn_particles):
        r = rx[i] - rx[j]
        if r.norm() < h:
          rrho[i] += rrho0 * Vb[i] * Wpol(r)
      Vb[i] = rrho0 * Vb[i] / rrho[i]
  setup_rigid()
    
  def setup_boundary():
    bx_l = []
    for i in range(int(bxxl[0])):
        for j in range(int(bxxl[1])):
          for k in range(int(bxxl[2])):
            if not(i == 0 or i == int(bxxl[0]) -1) and not(j == 0 or j == int(bxxl[1]) -1) and not(k == 0 or k == int(bxxl[2]) -1):
              pass
            else:                
              bx_l.append([(bound[0][0]+bound[1][0])/2 - bxx[0]/2 + (i * l_size*2) + l_size,
                           (bound[0][1]+bound[1][1])/2 - bxx[1]/2 + (j * l_size*2) + l_size,
                           (bound[0][2]+bound[1][2])/2 - bxx[2]/2 + (k * l_size*2) + l_size])
    return np.array(bx_l)
  bx.from_numpy(setup_boundary())

  support = 1
  @ti.kernel
  def initialize_bparticle_grid():
    for p in bx:
      ipos = ti.floor(bx[p] * n_grid).cast(ti.i32)
      for i in range(-support, support + 1):
        for j in range(-support, support + 1):
          for k in range(-support, support + 1):
            offset = ti.Vector([i, j, k])
            box_ipos = ipos + offset
            ti.append(bpid.parent(), box_ipos, p)
  initialize_bparticle_grid()

  @ti.kernel
  def setup_boundrho():
    for i in bx:
      delta = 0.0
      ipos = ti.floor(bx[i] * n_grid).cast(ti.i32)
      num_particles = ti.length(bpid.parent(), ipos)
      for k in range(num_particles):
        j = bpid[ipos[0], ipos[1], ipos[2], k]
        if i != j:
          r = bx[i] - bx[j]
          if r.norm() < h:
            delta += Wpol(r)
      V = 0.7 / delta
      brho[i] = 0.0
      for k in range(num_particles):
        j = bpid[ipos[0], ipos[1], ipos[2], k]
        r = bx[i] - bx[j]
        if r.norm() < h:
          brho[i] += brho0 * V * Wpol(r)
  setup_boundrho()

  rendering = False
  render_record = False
  CFL = False

  if rendering==True:
    gui = ti.GUI('3DSPH',res)
  for frame in range(1000):
    time_s = time.perf_counter()
    pidroot.deactivate_all()
    pidroot_r.deactivate_all()
    initialize_particle_grid()
    substep()
    time_e = time.perf_counter()
    if rendering == True:
      render()
      img = np.zeros((res[0], res[1], 3), dtype = np.float32)
      copy(img)
      gui.set_image(img)
      if render_record == False:
        gui.show()
      else:
        gui.show(f'sph3d_m/{frame:04d}.png')
    if frame == 0:
      pass
    elif frame != 1:
      tim=max(tim,time_e-time_s)
      tim_a = (tim_a + time_e-time_s) / 2
    else:
      tim = time_e-time_s
      tim_a = tim
    if CFL == True:
      dCFL()
  print("tim_a", tim_a)
  if CFL == True:
    print(pCFL[None])

if __name__ == '__main__':
  main()
  ti.kernel_profiler_print()
