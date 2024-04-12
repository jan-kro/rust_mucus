import numpy as np
import matplotlib.pyplot as plt
import rust_mucus
import mucus_rust as mc
from time import time

# python /home/janmak98/mucus_rust_test/rust_mucus/test_traj.py

# old system
cfg_path_old = "/net/storage/janmak98/masterthesis/output/test_systems/configs/cfg_mesh_tracer_6a_uncharged.toml"
cfg_old = mc.Config.from_toml(cfg_path_old)
top_old = mc.Topology(cfg_old)

# new_system
cfg_path_new = "/net/storage/janmak98/masterthesis/output/test_systems/configs/cfg_mesh_tracer_6a_uncharged_v5.toml"
cfg_new = mc.Config.from_toml(cfg_path_new)
top_new = mc.Topology(cfg_new)