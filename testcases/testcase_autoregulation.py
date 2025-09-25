"""
A python script to simulate stationary blood flow in microvascular networks with considering the vascular distensibility
and autoregulation mechanisms. In response to pressure perturbations (e.g., healthy conditions, ischaemic stroke), the
cerebral autoregulation feedback mechanisms act to change the wall stiffness (or the compliance), and hence the diameter,
of the autoregulatory microvessels.
Baseline is at healthy conditions for 100 and 10mmHg of the inlet and outlet boundary pressure, respectively.
The reference state for the distensibility law is computed based on the baseline condition.
Capabilities:
1. Import a network from file or generate a hexagonal network
2. Compute the edge transmissibilities with taking the impact of RBCs into account (Fahraeus, Fahraeus-Linquist effects)
3. Solve for flow rates, pressures and RBC velocities
4. Update the vessel diameters based on our autoregulation model
5. Save the results in a file
"""
import sys
import numpy as np
import igraph
import matplotlib.pyplot as plt

from source.flow_network import FlowNetwork
from source.distensibility import Distensibility
from source.autoregulation import Autoregulation
from types import MappingProxyType
import source.setup.setup as setup

import time
import concurrent.futures


# MappingProxyType is basically a const dict.
PARAMETERS = MappingProxyType(
    {
        # Setup parameters for blood flow model
        "read_network_option": 3,  # 1: generate hexagonal graph
                                   # 2: import graph from csv files
                                   # 3: import graph from igraph file (pickle file)
                                   # 4: todo import graph from edge_data and vertex_data pickle files
        "write_network_option": 4,  # 1: do not write anything
                                    # 2: write to igraph format # todo: handle overwriting data from import file
                                    # 3: write to vtp format
                                    # 4: write to two csv files
        "tube_haematocrit_option": 2,  # 1: No RBCs (ht=0)
                                       # 2: Constant haematocrit
        "rbc_impact_option": 3,  # 1: No RBCs (hd=0)
                                 # 2: Laws by Pries, Neuhaus, Gaehtgens (1992)
                                 # 3: Laws by Pries and Secomb (2005)
        "solver_option": 1,  # 1: Direct solver
                             # 2: PyAMG solver
                             # 3-...: other solvers

        # Elastic vessel - vascular properties (tube law) - Only required for distensibility and autoregulation models
        "pressure_external": 0.,                    # Constant external pressure
        "read_vascular_properties_option": 2,       # 1: Do not read anything
                                                    # 2: Read vascular properties from csv file
        "tube_law_ref_state_option": 4,             # 1: No update of diameters due to vessel distensibility
                                                    # 2: Passive diam changes, tube law. 1/D_ref ≈ 1/D. p_ext = p_base,
                                                        # d_ref = d_base
                                                    # 3: Passive diam changes, tube law. 1/D_ref ≈ 1/D. p_ext = const,
                                                        # d_ref computed based on Sherwin et al. (2003)
                                                    # 4: Passive diam changes, tube law. 1/D_ref ≈ 1/D. p_ext = const,
                                                        # d_ref computed based on Payne et al. (2023)

        "csv_path_vascular_properties": "/storage/homefs/cl22d209/microBlooM_B6_B_init_061_trial_050/testcase_healthy_autoregulation_curve/autoregulation_our_networks/data/parameters/B6_B_init_061/B6_B_init_061_all_parameters_Emodulus_correction_generation_07.csv",

        # Blood properties
        "ht_constant": 0.3,  # only required if RBC impact is considered
        "mu_plasma": 0.0012,

        # Hexagonal network properties. Only required for "read_network_option" 1
        "nr_of_hexagon_x": 3,
        "nr_of_hexagon_y": 3,
        "hexa_edge_length": 80e-6,
        "hexa_diameter": 18e-6,
        "hexa_boundary_vertices": [0, 27],  # 189
        "hexa_boundary_values": [13332., 8665.],
        "hexa_boundary_types": [1, 1],
        "stroke_edges": [0, 1],  # Example: Occlude 2 edges at inflow - manually assigning of blocked vessel ids
        "diameter_blocked_edges": .5e-6,

        # Import network from csv options. Only required for "read_network_option" 2
        "csv_path_vertex_data": "testcase_healthy_autoregulation_curve/data/network_payne/nodes.csv",
        "csv_path_edge_data": "testcase_healthy_autoregulation_curve/data/network_payne/edges.csv",
        "csv_path_boundary_data": "data/network/b6_B_pre_061/boundary_node_data.csv",
        "csv_diameter": "D", "csv_length": "L",
        "csv_edgelist_v1": "Source vx", "csv_edgelist_v2": "Target vx",
        "csv_coord_x": "x", "csv_coord_y": "y", "csv_coord_z": "z",
        "csv_boundary_vs": "nodeId", "csv_boundary_type": "boundaryType", "csv_boundary_value": "boundaryValue",

        # Import network from igraph option. Only required for "read_network_option" 3
        "pkl_path_igraph": "/storage/homefs/cl22d209/microBlooM_B6_B_init_061_trial_050/testcase_healthy_autoregulation_curve/autoregulation_our_networks/data/networks/B6_B_init_061/B6_B_init_061.pkl",
        "ig_diameter": "diameter", "ig_length": "length", "ig_coord_xyz": "coords",
        "ig_boundary_type": "boundaryType",  # 1: pressure & 2: flow rate
        "ig_boundary_value": "boundaryValue",

        # Write options
        "write_override_initial_graph": False,  # todo: currently does not do anything
        "write_path_igraph": "/storage/homefs/cl22d209/microBlooM_B6_B_init_061_trial_050/testcase_healthy_autoregulation_curve/autoregulation_our_networks/output/B6_B_init_061/trial_050/results",  # only required for "write_network_option" 2, 3, 4


        ##########################
        # Vessel distensibility options
        ##########################

        # Set up distensibility model
        "read_dist_parameters_option": 2,       # 1: Do not read anything
                                                # 2: Read from csv file

        "dist_pres_area_relation_option": 2,    # 1: No update of diameters due to vessel distensibility
                                                # 2: Relation based on Sherwin et al. (2003) - non linear p-A relation

        # Distensibility edge properties
        "csv_path_distensibility": "/storage/homefs/cl22d209/microBlooM_B6_B_init_061_trial_050/testcase_healthy_autoregulation_curve/autoregulation_our_networks/data/parameters/B6_B_init_061/B6_B_init_061_dist_parameters_Emodulus_correction_generation_07.csv",

        ##########################
        # Autoregulation options
        ##########################

        # Modelling constants
        "sensitivity_direct_stress": 4.,        # Sensitivity factor of direct stresses
        "sensitivity_shear_stress": .5,        # Sensitivity factor of direct stresses

        "relaxation_factor": 0.1,               # Alpha - relaxation factor

        # Set up distensibility model
        "read_auto_parameters_option": 2,       # 1: Do not read anything
                                                # 2: Read from csv file

        "base_compliance_relation_option": 2,   # 1: Do not specify compliance relation
                                                # 2: Baseline compliance using the definition C = dV/dPt based on Sherwin et al. (2023)

        "auto_feedback_model_option": 2,        # 1: No update of diameters due to autoregulation
                                                # 2: Our approach - Update diameters by adjusting the autoregulation model proposed by Payne et al. (2023)

        # Autoregulation edge properties
        "csv_path_autoregulation": "/storage/homefs/cl22d209/microBlooM_B6_B_init_061_trial_050/testcase_healthy_autoregulation_curve/autoregulation_our_networks/data/parameters/B6_B_init_061/B6_B_init_061_auto_parameters_Emodulus_correction_generation_07.csv",

        "sensitivity_analysis": False,
    }
)


def model_simulation(percent):

    print("\nPercent of Inlet Pressure Drop: " + str(round(percent * 100)) + "%")
    # Create object to set up the simulation and initialise the simulation
    setup_simulation = setup.SetupSimulation()
    # Initialise the implementations based on the parameters specified
    imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_velocity, imp_buildsystem, \
        imp_solver, imp_iterative, imp_balance, imp_read_vascular_properties, imp_tube_law_ref_state = setup_simulation.setup_bloodflow_model(PARAMETERS)

    imp_read_dist_parameters, imp_dist_pres_area_relation = setup_simulation.setup_distensibility_model(PARAMETERS)

    imp_read_auto_parameters, imp_auto_baseline, imp_auto_feedback_model = setup_simulation.setup_autoregulation_model(PARAMETERS)

    # Build flownetwork object and pass the implementations of the different submodules, which were selected in
    #  the parameter file
    flow_network = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem,
                               imp_solver, imp_velocity, imp_iterative, imp_balance, imp_read_vascular_properties,
                               imp_tube_law_ref_state, PARAMETERS)

    distensibility = Distensibility(flow_network, imp_read_dist_parameters, imp_dist_pres_area_relation)

    autoregulation = Autoregulation(flow_network, imp_read_auto_parameters, imp_auto_baseline, imp_auto_feedback_model)

    flow_network.percent = round(percent * 100)  ### REMOVE

    # Import or generate the network - Import data for the pre-stroke state
    print("Read network: ...")
    flow_network.read_network()
    print("Read network: DONE")

    # Baseline
    # Diameters at baseline.
    # They are needed to compute the reference pressure and diameters - only for distensibility_ref_state: 3
    print("Solve baseline flow (for reference): ...")
    flow_network.update_transmissibility()
    flow_network.update_blood_flow()
    print("Solve baseline flow (for reference): DONE")

    print("Check flow balance: ...")
    flow_network.check_flow_balance()
    print("Check flow balance: DONE")

    igraph_pkl_path = PARAMETERS["pkl_path_igraph"]  # existing igraph network
    graph = igraph.Graph.Read_Pickle(igraph_pkl_path)
    nr_vs = graph.vcount()
    COW_in = np.arange(nr_vs)[np.array(graph.vs["COW_in"]) == 1]
    inlet_edges = np.array(graph.incident(COW_in[0]))

    density = 1046  # kg/m^3
    # C57BL/6 I
    total_tissue_volume_B6_I = 9.92446030965163 * 1.00E-09  # m^3
    total_tissue_mass_B6_I = total_tissue_volume_B6_I * density * 1000  # g_tissue

    # C57BL/6 II
    # total_tissue_volume_B6_II = 10.072949889903173 * 1.00E-09  # m^3
    # total_tissue_mass_B6_II = total_tissue_volume_B6_II * density * 1000  # g_tissue

    # BALB/c I
    # total_tissue_volume_balbc_I = 9.863212663611106 * 1.00E-09  # m^3
    # total_tissue_mass_balbc_I = total_tissue_volume_balbc_I * density * 1000  # g_tissue

    # BALB/c II
    # total_tissue_volume_balbc_II = 11.935570165529526 * 1.00E-09 # m^3
    # total_tissue_mass_balbc_II = total_tissue_volume_balbc_II * density * 1000  # g_tissue

    flow_rate_inlet_edges_data_es_base = np.abs(flow_network.flow_rate[inlet_edges])
    total_flow_base = np.sum(flow_rate_inlet_edges_data_es_base) * 1E06 * 60  # ml_blood/min
    CBF_base = total_flow_base / total_tissue_mass_B6_I * 100  # ml_blood/(min*100g_tissue)

    print("Baseline - CBF [ml/min/100g]: {:.2f}".format(CBF_base))

    if percent == 1:
        flow_network.write_network()
        return 1.

    print("Initialise tube law for elastic vessels based on baseline results: ...")
    flow_network.initialise_tube_law()
    print("Initialise tube law for elastic vessels based on baseline results: Done")

    # Save pressure filed and diameters at baseline.
    autoregulation.diameter_baseline = np.copy(flow_network.diameter)
    autoregulation.pressure_baseline = np.copy(flow_network.pressure)
    autoregulation.flow_rate_baseline = np.copy(flow_network.flow_rate)

    flow_network.diameter_baseline = np.copy(flow_network.diameter)

    print("Initialise distensibility model based on baseline results: ...")
    distensibility.initialise_distensibility()
    print("Initialise distensibility model based on baseline results: DONE")

    print("Initialise autoregulation model: ...")
    autoregulation.initialise_autoregulation()
    autoregulation.alpha = PARAMETERS["relaxation_factor"]
    print("Initialise autoregulation model: DONE")

    # Change the intel pressure boundary condition - Mean arterial pressure (MAP) of the network
    print("Change the intel pressure boundary condition - MAP: ...")
    flow_network.boundary_val[0] *= percent  # change the inlet pressure -- a 15 % drop in the starting inlet pressure
    print("Change the intel pressure boundary condition - MAP: DONE")

    print("Autogulation Region - Autoregulatory vessels change their diameters based on Compliance feedback model", flush=True)
    # Update diameters and iterate (has to be improved)
    print("Update the diameters based on Compliance feedback model: ...", flush=True)
    tol = 1.0E-6
    autoregulation.diameter_previous = flow_network.diameter  # Previous diameters to monitor convergence of diameters

    max_rel_change_ar = np.array([])
    end_iteration = 0
    max_iterations = 2000000
    for i in range(max_iterations):
        autoregulation.iteration = i
        flow_network.update_transmissibility()
        flow_network.update_blood_flow()
        flow_network.check_flow_balance()
        distensibility.update_vessel_diameters_dist()
        autoregulation.update_vessel_diameters_auto()
        rel_change = np.abs(
            (flow_network.diameter - autoregulation.diameter_previous) / autoregulation.diameter_previous)
        max_rel_change_ar = np.append(max_rel_change_ar, np.max(rel_change))
        # convergence criteria
        if (i + 1) % 10 == 0:
            print("Autoregulation update: it=" + str(i + 1) + ", residual = " + "{:.2e}".format(np.max(rel_change))
                  + " um (tol = " + "{:.2e}".format(tol) + ")")

        if np.max(rel_change) < tol:
            print("Autoregulation update: DONE")
            end_iteration = i
            break
        else:
            autoregulation.diameter_previous = np.copy(flow_network.diameter)
            if i == max_iterations - 1:
                sys.exit("Fail to update the diameters based on Compliance feedback model ...")
    print("Update the diameters based on Compliance feedback model: DONE")

    # After pressure drop
    flow_rate_inlet_edges_data_es_B6_I = np.abs(flow_network.flow_rate[inlet_edges])
    total_flow_B6_I = np.sum(flow_rate_inlet_edges_data_es_B6_I) * 1E06 * 60  # ml_blood/min
    CBF = total_flow_B6_I / total_tissue_mass_B6_I * 100  # ml_blood/(min*100g_tissue)
    print("After pressure drop - CBF [ml/min/100g]: {:.2f}".format(CBF))
    print("Rel CBF: {:.5f}".format(CBF / CBF_base))
    fig, ax = plt.subplots()
    itarations = np.arange(1, end_iteration + 2, dtype=int)
    ax.plot(itarations, max_rel_change_ar)
    ax.set_yscale('log')
    ax.set_xlabel("Iterations", fontsize=16)
    ax.set_ylabel("Max Rel Diameter Change [-]", fontsize=16)
    ax.set_title("Convergence Curve", fontsize=16)
    plt.savefig("/storage/homefs/cl22d209/microBlooM_B6_B_init_061_trial_050/testcase_healthy_autoregulation_curve/autoregulation_our_networks/output/B6_B_init_061/trial_050/"
                "Convergence_curve_" + str(flow_network.percent) + ".png")
    plt.close()

    rel_stiffness = autoregulation.rel_stiffness
    rel_compliance = autoregulation.rel_compliance
    sens_shear = autoregulation.sens_shear
    sens_direct = autoregulation.sens_direct

    flow_network.rel_stiffness = np.ones(flow_network.nr_of_es) * (-1.)
    flow_network.rel_stiffness[autoregulation.eid_vessel_autoregulation] = rel_stiffness
    flow_network.rel_compliance = np.ones(flow_network.nr_of_es) * (-1.)
    flow_network.rel_compliance[autoregulation.eid_vessel_autoregulation] = rel_compliance
    flow_network.sens_shear = np.zeros(flow_network.nr_of_es)
    flow_network.sens_shear[autoregulation.eid_vessel_autoregulation] = sens_shear
    flow_network.sens_direct = np.zeros(flow_network.nr_of_es)
    flow_network.sens_direct[autoregulation.eid_vessel_autoregulation] = sens_direct

    flow_network.write_network()

    return


# Function to execute in parallel
def task(percent):
    print("\nPercent of Inlet Pressure Drop: " + str(round(percent * 100)) + "%")
    model_simulation(percent)
    return


print("[Network Name]")

# Number of CPUs to use
# num_cpus = multiprocessing.cpu_count()
num_cpus = 20

# List of items to process
inlet_percent = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1., 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])

# Record the start time
start_time = time.time()

# Using ThreadPoolExecutor with 20 workers
with concurrent.futures.ThreadPoolExecutor(max_workers=num_cpus) as executor:
    # Submit tasks and wait for them to complete
    futures = [executor.submit(task, percent) for percent in inlet_percent]
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        # print(f"\nResult from iteration {result}")

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"All tasks completed in {elapsed_time:.2f} seconds.", flush=True)


print("#########")
