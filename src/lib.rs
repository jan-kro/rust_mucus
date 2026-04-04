use std::mem::Discriminant;

use numpy::npyffi::npy_float16;
use numpy::PyArray3;
use pyo3::ffi::Py_TPFLAGS_LIST_SUBCLASS;
use pyo3::prelude::*;
use pyo3::types::*;
use pyo3::types::PyList;
use numpy::{PyArray1,PyArray2,IntoPyArray};
use ndarray::prelude::*;
use ndarray::{Array,ArrayView2,Array2, NdIndex, Dim};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> String {
    // if let b_unwrapped = b.extract::<usize>()
    (a + b + 1).to_string()
}


#[pyfunction]
fn get_dist_dir(positions: &PyArray2<f64>, distances: &PyArray2<f64>, directions: &PyArray3<f64>, l_box: f64, n_particles: usize) {
    // make mutable
    let positions      = unsafe { positions.as_array_mut() };
    let mut distances  = unsafe { distances.as_array_mut() };
    let mut directions = unsafe { directions.as_array_mut() };

    let mut dist: f64;
    //let mut dir: Array2<f64> = Array::zeros(2);
    
    for i in 0..n_particles {
        for j in 0..n_particles {
            if i > j {
                for d in 0..3 {
                    dist = positions[[j, d]] - positions[[i, d]];
                    directions[[i, j, d]] =  dist - l_box * (dist / l_box).round();
                    directions[[j, i, d]] = -directions[[i, j, d]];
                }

                distances[[i, j]] = (directions[[i, j, 0]].powi(2) + directions[[i, j, 1]].powi(2) + directions[[i, j, 2]].powi(2)).sqrt();
                distances[[j, i]] = distances[[i, j]]
            }
        }
    }
}

#[pyfunction]
fn get_dist(positions: &PyArray2<f64>, distances: &PyArray2<f64>, l_box: f64, n_particles: usize, n_dim: usize) {
    // make mutable
    let positions      = unsafe { positions.as_array_mut() };
    let mut distances  = unsafe { distances.as_array_mut() };

    let mut dir       = Array::zeros(n_dim);     // distance vector
    let mut dist:       f64;                     // distance
    let mut dist2:      f64;                     // squared distance
    
    for i in 0..n_particles {
        for j in 0..n_particles {
            if i > j {
                dist2 = 0.0;

                // get distance vector
                for d in 0..n_dim {
                    dir[d] = positions[[j, d]] - positions[[i, d]];
                    dir[d] -= l_box * (dir[d] / l_box).round();
                    
                    // get squared distance
                    dist2 += dir[d].powi(2);
                }
                
                // get distance 
                dist = dist2.sqrt();
                

                // save distance
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
                
            }
        }
    }
}


#[pyfunction]
fn get_dist_histogram(
    positions:      &PyArray2<f64>,
    counts:         &PyArray1<isize>,
    r_range:        (f64, f64),
    n_bins:         usize, 
    l_box:          f64, 
    n_particles:    usize,
    tag_list:       &PyArray1<usize>,
    tag_pair:       (usize, usize), 
    n_dim:          usize) {
    
    let positions   = unsafe { positions.as_array() };
    let tag_list    = unsafe { tag_list.as_array() };
    let mut counts  = unsafe { counts.as_array_mut() };

    let mut dir       = Array::zeros(n_dim);     // distance vector
    let mut dist:       f64;                     // distance
    let mut dist2:      f64;                     // squared distance

    let (r_min, r_max) = r_range;
    let bin_width = (r_max - r_min) / n_bins as f64;

    // reset counts
    counts.fill(0);
    
    // go through every combination of particles
    for i in 0..n_particles { 
        for j in 0..n_particles {
            
            // check if particles are of type tag_pair.0 = a and tag_pair.1 = b respectively
            let forward_match = tag_list[i] == tag_pair.0 && tag_list[j] == tag_pair.1;
            let reverse_match = tag_list[j] == tag_pair.0 && tag_list[i] == tag_pair.1;

            // do not count i==j
            // do not calculate the same distance twice (because [i,j] and [j,i] will give the same distance)
            // the order the match of particle types does not matter i.e. [a,b] or [b,a]
            if i > j && (forward_match || reverse_match) {
                dist2 = 0.0;

                // get distance vector
                for d in 0..n_dim {
                    dir[d] = positions[[j, d]] - positions[[i, d]];
                    dir[d] -= l_box * (dir[d] / l_box).round();
                    
                    // get squared distance
                    dist2 += dir[d].powi(2);
                }
                
                // get distance 
                dist = dist2.sqrt();
                
                // Bin distances
                if dist >= r_min && dist < r_max {
                    let bin = ((dist - r_min) / bin_width).floor() as usize;
                    if bin < n_bins {
                        // if a==b, then the distance needs to be counted twice
                        if forward_match && reverse_match {
                            counts[bin] += 2;
                        // if a!=b only count the distance once
                        } else {
                            counts[bin] += 1
                        }
                    }
                // this might count double, if r_range == L_box/2
                //TODO delete after thinking about it / if problems arise
                } else if dist == r_max {
                    // Edge case: if value equals r_max, assign to last bin
                    if n_bins > 0 {
                        if forward_match && reverse_match {
                            counts[n_bins - 1] += 2;
                        } else {
                            counts[n_bins - 1] += 1;
                        }
                    }
                }
            }
        }
    }
}


fn force_bonded(positions: ArrayView2<f64>, force_total: &mut Array2<f64>, k_harm: f64) {
    for i in 0..force_total.shape()[0] {
        force_total[[i, 0]] -= k_harm * positions[[0, 0]];
    }
}

//define python function that takes n usize as input and returns a numpy array of size n containing ones
// #[pyfunction]
// fn integrate<'py>(py: Python<'py>, n: usize) -> &'py PyArray1<f64> {
//     let mut ones = Array::ones(n);
//     ones.into_pyarray(py)
// }
#[pyfunction]
fn integrate(traj: &PyArray3<f64>, steps: usize, dt: f64, k_harm: f64, mass: f64, x0: f64, couple_const: &PyArray1<f64>){
    // We need to cast the Python object to a NumPy array to be able to access and write its data.
    let mut traj = unsafe { traj.as_array_mut() };
    // these might be useful later
    let n_particles = traj.shape()[1];
    let n_dim = traj.shape()[2];
    // We make a temporary array to store the total force on each particle, pass this to force functions
    let mut force_total: Array2<f64> = Array2::zeros((n_particles, n_dim));
    let couple_const = unsafe { couple_const.as_array() };
    let k_harm = couple_const[0] + k_harm;
    // iterate over time steps
    // hacky way to get forces
    force_bonded(traj.index_axis(Axis(0), 0), &mut force_total, k_harm);
    for idx_p in 0..n_particles {
        for idx_dim in 0..n_dim {
            traj[[0, idx_p, idx_dim]] = force_total[[idx_p, idx_dim]];
        }
    }
    // for idx_step in 0..steps {
    //     let t = idx_step as f64 * dt;
    //     let x = x0 * (t * k_harm / mass).cos();
    //     // traj[[idx_step, 0, 0]] = x;

    //     force_harmonic(traj.index_axis(Axis(0), idx_step), &mut force_total, k_harm);
    // }
    }
    

/// Takes positions and force field parameters and the force array as pointers to the python arrays
/// and rewrites the force array with the forces on each particle.
/// 
/// This function uses a cell linked list to speed up the calculation of the particle distances.
///
/// # Arguments
///
/// * `positions` - 2D array of positions of shape (n_particles, n_dim)
/// * `tags` - 1D array of atom labels (unsinged integers) of shape (n_particles)
/// * `bond_table` - 2D array of bond table of shape (n_particles, n_particles)
/// * `force_constants` - 2D array of force constants of shape (n_tags, n_tags)
/// * `bond_lengths` - 2D array of bond lengths of shape (n_tags, n_tags)
/// * `sigmas_lj` - 2D array of sigma lj parameters of shape (n_tags, n_tags)
/// * `epsilons_lj` - 2D array of epsilon lj parameters of shape (n_tags, n_tags)
/// * `charges` - 1D array of particle charges of shape (n_tags)
/// * `lB_debye` - debye length
/// * `B_debye` - debye constant
/// * `force_total` - 2D array of forces of shape (n_particles, n_dim)
/// * `l_box` - box length
/// * `cutoff2` - interaction cutoff distance squared
/// * `n_particles` - number of particles
/// * `n_dim` - number spacial dimensions
/// * `use_force_bonded` - boolean to use bonded forces
/// * `use_force_lj` - boolean to use lennard jones forces
/// * `use_force_deb` - boolean to use debye forces
/// * `neighbour_cells_idx` - 2D array of indices of neighboring cells of shape (n_skin_cells, 3)
/// * `head_array` - 3D array of head indices of shape (n_cells, n_cells, n_cells)
/// * `list_array` - 1D array of list indices of shape (n_particles)
/// * `n_cells` - number of cells in each dimension
/// * `n_neighbour_cells` - number of neighbor cells (length of neighbour_cells_idx)
#[pyfunction]
fn get_forces_cell_linked_virial(
    positions:              &PyArray2<f64>, 
    tags:                   &PyArray1<usize>,
    bond_list:              &PyList, //Vec<Vec<usize>>,
    // bond_table:             &PyArray2<bool>,    // size is here (nparticles, nparticles)
    force_constants:        &PyArray2<f64>,     // size is here (ntags, ntags)
    bond_lengths:           &PyArray2<f64>,     // size is here (ntags, ntags)
    sigmas_lj:              &PyArray2<f64>,     // size is here (ntags, ntags)
    epsilons_lj:            &PyArray2<f64>,     // size is here (ntags, ntags)
    charges:                &PyArray1<f64>,     // size is here (ntags)
    lB_debye:               f64,
    B_debye:                f64,
    force_total:            &PyArray2<f64>,
    distances:              &PyArray2<f64>,     // here
    l_box:                  f64,
    cutoff2:                f64,  
    n_particles:            usize,
    n_dim:                  usize,
    write_distances:        bool,               // here
    use_force_bonded:       bool,
    use_force_lj:           bool,
    use_force_deb:          bool,
    neighbour_cells_idx:    &PyArray2<isize>,     // contains the index list of the skin cells of shape (n_skin_cells, 3)
    head_array:             &PyArray3<isize>,     // head array of shape (n_cells,n_cells,n_cells)
    list_array:             &PyArray1<isize>,     // list array of shape (n_particles)
    n_cells:                usize,              // number of cells in each dimension
    n_neighbour_cells:      usize,              // number of neighbor cells
    calc_virial:            bool,
    virial:                 &PyArray2<f64>,
    ) {

    // load arrays 
    let positions       = unsafe { positions.as_array() };
    let tags            = unsafe { tags.as_array() };
    let bond_list: Vec<Vec<usize>> = unsafe { bond_list.extract().unwrap_unchecked() };
    // the bond list is a python list of lists where the list at i contains all the bond partners of i, which may vary
    // so the lists in each entry of bond list may be of varying size
    let force_constants = unsafe { force_constants.as_array() };
    let bond_lengths    = unsafe { bond_lengths.as_array() };
    let sigmas_lj       = unsafe { sigmas_lj.as_array() };
    let epsilons_lj     = unsafe { epsilons_lj.as_array() };
    let charges         = unsafe { charges.as_array() };
    let mut force_total = unsafe { force_total.as_array_mut() };
    let mut distances   = unsafe { distances.as_array_mut() };
    let mut virial      = unsafe { virial.as_array_mut() };

    let neighbour_cells_idx = unsafe { neighbour_cells_idx.as_array() };
    let head_array = unsafe { head_array.as_array() };
    let list_array = unsafe { list_array.as_array() };

    // define variables
    let mut tag_pair:   (usize, usize) = (0, 0); // tag pair of interacting particles
    let mut dir       = Array::zeros(n_dim);     // distance vector
    let mut dist2:      f64;                     // squared distance
    let mut dist:       f64;                     // distance
    let mut force:      f64;                     // force acting on particle i resulting from the interaction with particle j
    let mut bonded:     bool;                    // boolean to check if particles are bonded
    let mut head_idx:   isize;                     // index of head particle of current cell
    let mut cell_idx:   isize;                     // index of head particle of neighboring cell
    let mut i:          usize;                   // index of particle i
    let mut j:          usize;                   // index of particle j
    let mut cell_i:     usize;                   // index i of neighboring cell
    let mut cell_j:     usize;                   // index j of neighboring cell
    let mut cell_k:     usize;                   // index k of neighboring cell

    // Chill code using the numpy crate functions for numpy functionality
    let wca_cutoff = &sigmas_lj * (2.0_f64.powf(1.0/6.0));
    
    // lets compute the bonded interactions before in a simple way such that there is no chance for those to accidentally get cut off
    for part_idx_i in 0..bond_list.len(){
        // the bond partners of particle i are contained in the list bond_list[part_idx_i]
        for b_table_idx_j in 0..bond_list[part_idx_i].len(){
            j = bond_list[part_idx_i][b_table_idx_j];
            // check for the index to not double compute things and so on.
            if part_idx_i < j {
                // reset variables
                dist2 = 0.0;
                tag_pair.0 = tags[part_idx_i];
                tag_pair.1 = tags[j];

                // get distance vector
                for d in 0..n_dim {
                    dir[d] = positions[[j, d]] - positions[[part_idx_i, d]];
                    dir[d] -= l_box * (dir[d] / l_box).round();
        
                    // get squared distance
                    dist2 += dir[d].powi(2);
                }
                
                // calculate distance
                dist = dist2.sqrt();
                // get bondforce
                force = 2.0*force_constants[tag_pair] * (1.0 - bond_lengths[tag_pair]/dist);
            
                for d in 0..n_dim {
                    // dir does not need to be normalized because it happens in force calculation
                    force_total[[part_idx_i, d]] += force * dir[d];
                    force_total[[j, d]] -= force * dir[d];
                }

                if calc_virial {

                    for a in 0..n_dim {
                        for b in 0..n_dim {
                            virial[[a, b]] += force * dir[b] * dir[a];
                        }
                    }
                }

            }
        }
    }

    if use_force_lj || use_force_deb { // we only need nonbonded contributions if we use these forces
    // loop over all cells
    for head_i in 0..n_cells {
    for head_j in 0..n_cells {
    for head_k in 0..n_cells {

        head_idx = head_array[[head_i, head_j, head_k]];

        // loop over all particles in the cell
        // loop over particle i
        while head_idx != -1 {
            
            // loop over all neighboring cells
            for n in 0..n_neighbour_cells {
                // rem_euclid is is the modulo function and is used to make the cells periodic
                cell_i = ((head_i as isize + neighbour_cells_idx[[n, 0]])).rem_euclid(n_cells as isize) as usize;
                cell_j = ((head_j as isize + neighbour_cells_idx[[n, 1]])).rem_euclid(n_cells as isize) as usize;
                cell_k = ((head_k as isize + neighbour_cells_idx[[n, 2]])).rem_euclid(n_cells as isize) as usize;

                // get the index of the head particle of the neighboring (or same) cell
                cell_idx = head_array[[cell_i, cell_j, cell_k]];

                // loop over all particles in the neighbouring (or same) cell
                // loop over particle j
                while cell_idx != -1 {

                    i = head_idx as usize;
                    j = cell_idx as usize;

                    if i > j { // only calculate half of the matrix and copy the negative values to the other half
                        force = 0.0;
                        dist2 = 0.0;
                        
                        // get tag pair
                        tag_pair.0 = tags[i];
                        tag_pair.1 = tags[j];
                
                        // note if pair is bonded
                        // bonded = bond_table[[i, j]];
                        
                        // in python:
                        // bond_list = [
                        //      [1, 99],
                        //      [0, 2],
                        //      [1, 3, 66],
                        //      [],
                        //      [3, 5],
                        //  ]
                        bonded = bond_list[i].contains(&j);
                        // bonded = bond_list[i].contains(&j);
                        // let bonded_to_i: Vec<usize> = {bond_list.get_item(i)}.extract();
                        // bonded = bonded_to_i.contains(&j);

                        // get distance vector
                        for d in 0..n_dim {
                            dir[d] = positions[[j, d]] - positions[[i, d]];
                            dir[d] -= l_box * (dir[d] / l_box).round();
                
                            // get squared distance
                            dist2 += dir[d].powi(2);
                        }
                        
                        // calculate distance
                        dist = dist2.sqrt();
                
                        if write_distances {
                            distances[[i, j]] = dist;
                            distances[[j, i]] = dist;
                        }
                
                        // check which forces to calculate
                        // calculate FORCES DIVIDED BY DISTANCE, so direction does not need to be normalized
                        //if bonded {
                        //    
                            // check if particles are bonded
                        //    if use_force_bonded {
                        //        // get bondforce
                        //        force += 2.0*force_constants[tag_pair] * (1.0 - bond_lengths[tag_pair]/dist);
                        //    }
                        //}
                
                        // check if distance is within cutoff
                        if dist2 < cutoff2 {
                            
                            if !bonded {
                                
                                // WCA is a non-bonded force    
                                if use_force_lj {

                                    if dist < wca_cutoff[tag_pair] { // enforce the force cutoff of the wca potential
                                    // get lennard jones force
                                    //WRONG: force += 4.0*epsilons_lj[tag_pair]*(-12.0*sigmas_lj[tag_pair].powi(12)/dist.powi(14) + 6.0*sigmas_lj[tag_pair].powi(7)/dist.powi(8));
                                        force += -24.0*epsilons_lj[tag_pair]*(2.0*(sigmas_lj[tag_pair]).powi(12)/dist.powi(14) - (sigmas_lj[tag_pair]).powi(6)/dist.powi(8));
                                    }
                                }
                            }
                            
                            if use_force_deb {
                                // get debye force
                                force += -charges[tag_pair.0]*charges[tag_pair.1]*lB_debye*(1.0+B_debye*dist)*(-B_debye*dist).exp()/dist.powi(3);
                            }
                        }


                
                        for d in 0..n_dim {
                            // dir does not need to be normalized because it happens in force calculation
                            force_total[[i, d]] += force * dir[d];
                            force_total[[j, d]] -= force * dir[d];
                        }

                        if calc_virial {

                            for a in 0..n_dim {
                                for b in 0..n_dim {
                                    virial[[a, b]] += force * dir[b] * dir[a];
                                }
                            }
                        }

                    }
                    cell_idx = list_array[cell_idx as usize];
                }
            }
            head_idx = list_array[head_idx as usize];
        }
    }}}
    }                
}


/// Updates the list and head arrays based on positions and cell_length
#[pyfunction]
fn update_linked_list(
    list_array:             &PyArray1<isize>,
    head_array:             &PyArray3<isize>,
    positions:              &PyArray2<f64>, 
    cell_length:            f64,
    ) {

    // load arrays 
    
    
    let mut list = unsafe { list_array.as_array_mut() };
    let mut head = unsafe { head_array.as_array_mut() };
    let pos      = unsafe { positions.as_array() };

    // Reset arrays to -1
    list.fill(-1);
    head.fill(-1);

    // Compute cell_index
    let shape = head.shape();
    let mut cell_index = Vec::with_capacity(pos.nrows());

    for row in pos.outer_iter() {
        let idx: Vec<usize> = row.iter()
            .map(|&v| ((v / cell_length).floor() as isize).max(0) as usize)
            .collect();

        // Clamp to valid bounds
        let x = idx[0].min(shape[0] - 1);
        let y = idx[1].min(shape[1] - 1);
        let z = idx[2].min(shape[2] - 1);

        cell_index.push([x, y, z]);
    }

    // Update linked list
    for (i, cell) in cell_index.iter().enumerate() {
        let x = cell[0];
        let y = cell[1];
        let z = cell[2];

        list[i] = head[[x, y, z]];
        head[[x, y, z]] = i as isize;
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_mucus(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(integrate, m)?)?;
    m.add_function(wrap_pyfunction!(get_dist_dir, m)?)?;
    m.add_function(wrap_pyfunction!(get_dist, m)?)?;
    //m.add_function(wrap_pyfunction!(get_forces, m)?)?;
    //m.add_function(wrap_pyfunction!(get_forces_cell_linked, m)?)?;
    //m.add_function(wrap_pyfunction!(get_forces_cell_linked_test, m)?)?;
    m.add_function(wrap_pyfunction!(update_linked_list, m)?)?;
    m.add_function(wrap_pyfunction!(get_dist_histogram, m)?)?;
    m.add_function(wrap_pyfunction!(get_forces_cell_linked_virial, m)?)?;
    Ok(())
}