import mucus as mc
import rust_mucus as rmc
import numpy as np
import h5py


def rdf(self,
        r_range = None,
        n_bins = None, #100
        bin_width = 0.05,
        tags = (0,0),
        save=True,
        overwrite=False,
        return_all=False):
    # TODO ADD DOCSTRING
    
    
    #NOTE: this rdf using the rust get_dist function works.
    #      it gives the exact same results as the one using the distance matrix
    #      which has been double checked with the VMD rdf plugin
    #      therefor it is save to assume that the rust get_dist function is correct
    
    key_dict = {"key": Filetypes.Rdf,
                "params": {"r_range": r_range, 
                            "n_bins": n_bins,
                            "bin_width": bin_width,
                            "tags": tags}}
    
    if self._exists(key_dict):
        fname = get_path(self.cfg, filetype=key_dict["key"])
        with h5py.File(fname, "r", locking=self.h5pylock) as fh5:
            key = key_dict["key"].key
            r = fh5[key][0,:]
            g_r = fh5[key][1,:]
            return r, g_r
    
    if not save and not return_all:
        raise ValueError("Either save or return_all has to be True")
    
    if r_range is None:
            r_range = np.array([0, self.cfg.lbox/2])
            
    if n_bins is None:   
        n_bins = int((r_range[1] - r_range[0]) / bin_width)

    n_dim = 3
    
    # create unit cell information
    #uc_vectors = np.repeat([np.array((lbox, lbox, lbox))], len(self.trajectory), axis=0)
    #uc_angles = np.repeat([np.array((90,90,90))], len(self.trajectory), axis=0)

    # create mask for distances
    mask_1 = self.topology.tags == tags[0]
    mask_2 = self.topology.tags == tags[1]
    mask_1, mask_2 = np.meshgrid(mask_1,mask_2)

    # mask_pairs is an n_particles x n_particles boolean array, where mask_pairs[i,j] is True if particle i and j are both of the correct type
    # the diagonal is set to False, as we do not want to calculate the distance of a particle with itself
    mask_pairs = np.logical_and(np.logical_and(mask_1, mask_2), np.logical_not(np.eye(self.cfg.n_particles, dtype=bool)))
    #! TODO only use upper or lower triangular part of mask_pairs

    # initialize rdf array
    g_r = np.zeros(n_bins)
    
    # initialize distance array
    distances = np.zeros_like(mask_pairs, dtype=np.float64)

    print(f"Calculating rdf for system {self.cfg.name_sys} ...\n")
    print("Started at ", datetime.datetime.now(), "\n") 
    print("Frame    of    Total")
    print("--------------------")
    report_stride = int(self.n_frames//10)
    
    if report_stride == 0:
        report_stride = 1

    fname_h5 = get_path(self.cfg, filetype=Filetypes.Trajectory)
    with h5py.File(fname_h5, "r", locking=self.h5pylock) as fh5:
        for i, frame_idx in enumerate(self.frame_indices):
            
            get_dist(fh5[Filetypes.Trajectory.key][frame_idx].astype(np.float64), distances, self.cfg.lbox, self.cfg.n_particles, n_dim)
            
            g_r_frame, edges = np.histogram(distances[mask_pairs], range=r_range, bins=n_bins)
            #! TODO TODO TODO TODO TODO TODO TODO 
            #       since an r_range is given not every atom is used for the histogram
            #       this might fuck with the normalization
            #! PROBABLY NORMALIZE BY SUM(BINS)
            g_r += g_r_frame
            if i%report_stride == 0:
                print(f"{i:<8d} of {self.n_frames:8d}") 
                
    r = 0.5 * (edges[1:] + edges[:-1])

    # ALLEN TILDERSLEY METHOD:
    # g_r[i] is average number of atoms which lie to a distance between r[i] and r[i]+dr to each other

    # number density of particles
    number_density = self.cfg.n_particles/self.cfg.lbox**3 # NOTE only for cubical box
    # now normalize by shell volume
    shell_volumes = (4 / 3) * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
    # normalisation
    g_r = g_r/ self.n_frames/ self.cfg.n_particles/ number_density/ shell_volumes
    
    # MDTRAJ METHOD:
    # Normalize by volume of the spherical shell.
    # See discussion https://github.com/mdtraj/mdtraj/pull/724. There might be
    # a less biased way to accomplish this. The conclusion was that this could
    # be interesting to try, but is likely not hugely consequential. This method
    # of doing the calculations matches the implementation in other packages like
    # AmberTools' cpptraj and gromacs g_rdf.

    # # unitcell_volumes = np.array(list(map(np.linalg.det, uc_vectors))) # this should be used if the unitcell is not cubic
    # unitcell_volumes = np.prod(uc_vectors, axis=1)
    # V = (4 / 3) * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
    # norm = len(pairs) * np.sum(1.0 / unitcell_volumes) * V # the trajectory length is implicitly included in the uc volumes
    # g_r = g_r.astype(np.float64) / norm

    # number_density = len(pairs) * np.sum(1.0 / unitcell_volumes) / natoms / len(self.trajectory)
    
    if save:
        self._save(key_dict, np.array([r, g_r]))
    
    if return_all:
        return r, g_r