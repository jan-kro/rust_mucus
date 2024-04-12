import numpy as np
import matplotlib.pyplot as plt
import rust_mucus
import mucus_rust as mc
from time import time

# set up test system
cfg_path = "/net/storage/janmak98/masterthesis/output/mesh_talk/configs/cfg_mesh_tracer_6a.toml"
cfg = mc.Config.from_toml(cfg_path)
top = mc.Topology(cfg)

positions = mc.utils.load_trajectory(cfg, frame=12345)[0]

# test dist_dir
def get_dist_dir(positions, box_length, n_beads):
    r_left = np.tile(positions, (n_beads, 1, 1))
    r_right = np.reshape(np.repeat(positions, n_beads, 0), (n_beads, n_beads, 3))
    directions = r_left - r_right
    directions -= box_length*np.round(directions/box_length)
    distances = np.linalg.norm(directions, axis=2)
    return distances, directions

def get_dist_dir_2(positions, box_length, n_beads, cutoff_pbc, idx_table, bonds):
    r_left = np.tile(positions, (n_beads, 1, 1))
    r_right = np.reshape(np.repeat(positions, n_beads, 0), (n_beads, n_beads, 3))
    directions = r_left - r_right
    directions -= box_length*np.round(directions/box_length)
    distances = np.linalg.norm(directions, axis=2)
    distances += 2*cutoff_pbc*np.eye(n_beads) # add cutoff to disregard same atoms
    L_box = distances < cutoff_pbc # NOTE the "<" is important, because if it was "<=" the diagonal values would be included
    directions = directions[L_box]
    distances  = distances[L_box]
    idx_interactions = idx_table[:, L_box].T     
    L_nn = list(())
    for idx in idx_interactions:
        if np.any(np.logical_and(idx[0] == bonds[:, 0], idx[1] == bonds[:, 1])):
            L_nn.append(True)
        else:
            L_nn.append(False)
    return distances, directions, idx_interactions, L_nn

def get_force_bonded(distances, directions, idx_interactions, L_nn, top):
        forces = np.zeros_like(top.positions)
        idxs = idx_interactions[L_nn]
        distances = distances[L_nn].reshape(-1,1)
        directions = directions[L_nn]
        force_constants = top.force_constant_nn[top.get_tags(idxs[:,0], idxs[:,1])].reshape(-1,1)
        r0 = top.r0_bond[top.get_tags(idxs[:,0], idxs[:,1])].reshape(-1,1)
        forces_temp = 2*force_constants*(1-r0/distances)*directions 
        for i, force in zip(idxs[:, 0], forces_temp):
            forces[i, :] += force
        return forces
    
def get_force_lj_cutoff(distances, directions, idx_interactions, top, cfg):
        forces = np.zeros_like(top.positions)
        L_lj = distances < cfg.cutoff_LJ
        idxs = idx_interactions[L_lj]
        distances = distances[L_lj].reshape(-1, 1)
        directions = directions[L_lj]
        epsilon = top.epsilon_lj[top.get_tags(idxs[:,0], idxs[:,1])].reshape(-1,1)
        sigma = top.sigma_lj[top.get_tags(idxs[:,0], idxs[:,1])].reshape(-1,1)
        forces_temp = 4*epsilon*(-12*sigma**12/distances**14 + 6*sigma**7/distances**8)*directions 
        for i, force in zip(idxs[:, 0], forces_temp):
            forces[i, :] += force
        return forces
    
def get_force_debye(distances, directions, idx_interactions, L_nn, B_debye, top, cfg):
        forces = np.zeros_like(top.positions)
        L_nb = np.logical_not(L_nn)
        idxs = idx_interactions[L_nb]
        distances = distances[L_nb].reshape(-1, 1)
        directions = directions[L_nb]
        q2 = top.q_bead[idxs[:,0]]*top.q_bead[idxs[:,1]]
        forces_temp = -q2*cfg.lB_debye*(1+B_debye*distances)*np.exp(-B_debye*distances)*directions/distances**3
        for i, force in zip(idxs[:, 0], forces_temp):
            forces[i, :] += force
        return forces

n_atoms = cfg.n_particles
l_box = cfg.lbox


# get distances and directions with the python function
tdp = time()
dist_old, dir_old = get_dist_dir(positions, l_box, n_atoms)
tdp = time() - tdp
# print(positions)


distances = np.zeros((n_atoms, n_atoms))
directions = np.zeros((n_atoms, n_atoms, 3))
tdr = time()
rust_mucus.get_dist_dir(positions, distances, directions, l_box, n_atoms)
tdr = time() - tdr
#print(distances)
# print(positions)
# print("")

# for i in range(3):
#     print(dir_old[:,:,i])
#     print(directions[:,:,i])
#     print("")

DELTA = 1e-10
n_wrong = 0

for i in range(n_atoms):
    for j in range(n_atoms):
        if i != j:
            d0 = distances[i, j] - dist_old[i, j]
            d1 = directions[i, j, 0] - dir_old[i, j, 0]
            d2 = directions[i, j, 1] - dir_old[i, j, 1]
            d3 = directions[i, j, 2] - dir_old[i, j, 2]
            if d0 > DELTA or d1 > DELTA or d2 > DELTA or d3 > DELTA:
                n_wrong += 1
                print("Wrong entry: ", i, j)
                print("distances: ", distances[i, j], dist_old[i, j])
                print("directions: ", directions[i, j, :], dir_old[i, j, :])
                print("")
                print(positions)
                print("")
                for k in range(3):
                    print(dir_old[:,:,k])
                    print(directions[:,:,k])
                    print("")
print("Number of atoms: ", n_atoms)

print("\n TEST DISTANCES AND DIRECTIONS \n")

print("Number of wrong entries: ", n_wrong)
print(f"Time for rust function:   {(tdr):.6f}")
print(f"Time for python function: {(tdp):.6f}")
print(f"Speedup (py/rs):          {(tdp/tdr)*100:.2f} %")
# steps = 100
# couple_consts = np.array([1.0])
# traj = np.zeros((steps, 1, 1))
# dt = 0.001
# mass = 12.
# k_harm = 17.0
# x0 = 1.0
# len_chunk = steps

tags = np.array(top.tags, dtype=np.uint)

# create bond table
bond_list = top.bonds
bond_table = np.zeros((n_atoms, n_atoms), dtype=bool)
for ij in bond_list:
    bond_table[ij[0], ij[1]] = True

# create charge list
charges_all = np.array(top.q_bead)
charges = np.zeros(top.ntags)
for i, charge in zip(tags, charges_all):
    charges[i] = charge

lB_debye = cfg.lB_debye
B_debye = np.sqrt(cfg.c_S)*cfg.r0_nm/10
sigmas_lj = top.sigma_lj
epsilons_lj = top.epsilon_lj
force_constants = top.force_constant_nn
bond_lengths = top.r0_bond
cutoff2 = cfg.cutoff_pbc**2
n_particles = n_atoms
n_dim = 3
use_force_bonded = True
use_force_lj = True
use_force_deb = True

# calculate bond forces with rust function
force_total_rust = np.zeros((n_atoms, 3))

tfr = time()
rust_mucus.get_forces(positions, 
                      tags, 
                      bond_table, 
                      force_constants, 
                      bond_lengths, 
                      sigmas_lj,
                      epsilons_lj,      
                      charges,          
                      lB_debye,         
                      B_debye,          
                      force_total_rust, 
                      l_box, 
                      cutoff2, 
                      n_particles, 
                      n_dim, 
                      use_force_bonded, 
                      use_force_lj, 
                      use_force_deb)   
tfr = time() - tfr

# define index table
idx_table = np.zeros((2, cfg.n_particles, cfg.n_particles), dtype=int)
for i in range(cfg.n_particles):
    for j in range(cfg.n_particles):
        idx_table[0, i, j] = i
        idx_table[1, i, j] = j

force_total = np.zeros((n_atoms, 3))
# calculate bond forces with python function

tfp = time()
distances, directions, idx_interactions, L_nn = get_dist_dir_2(positions, l_box, n_particles, cfg.cutoff_pbc, idx_table, top.bonds)
force_total += get_force_bonded(distances, directions, idx_interactions, L_nn, top)
force_total += get_force_lj_cutoff(distances, directions, idx_interactions, top, cfg)
force_total += get_force_debye(distances, directions, idx_interactions, L_nn, B_debye, top, cfg)
tfp = time() - tfp

# check if forces are equal
count_false = 0
for force_r, force_p in zip(force_total_rust, force_total):
    for i in range(3):
        if np.abs(force_r[i] - force_p[i]) > DELTA:
            count_false += 1
            # print("Forces are not equal")
            # print(force_r)
            # print(force_p)

print("\n TEST FORCES \n")

print(f"Number of false entries: {count_false}")

print(f"Time for rust function:   {tfr:.6f}")
print(f"Time for python function: {tfp:.6f}")
print(f"Speedup (py/rs):          {(tfp/tfr)*100:.2f} %")

# print("")
# print(force_total_rust)
# print("")
# print(force_total)

# a = np.random.randn(10000, 10000)
# b = np.random.randn(10000, 10000)

# ta = time()
# a = np.zeros_like(a)
# ta = time() - ta

# tb = time()
# b.fill(0)
# tb = time() - tb

# print(f"\nTime for a: {ta:.6f}")
# print(f"Time for b: {tb:.6f}")
# print(f"Speedup (a/b): {(ta/tb)*100:.2f} %")
# print(b.dtype)