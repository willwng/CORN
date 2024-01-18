#!/bin/bash
# in the directory you call this script from.
# This script will create the Parameters.py

ALPHA=1
KAPPA=0.0
MU=0
COMPRESSION=1.0
STRAIN=0.01
PYTHON_EXEC="python3"
### CREATE THE PARAMETERS.PY

function create_parameters {
  cat >parameters.py <<EOF

"""
Parameters.py is used for user input.
The following class contains variable assignments used in the code
"""
from typing import List, Optional
from datetime import datetime

from energyminimization.solvers.solver import MinimizationType
from result_handling.output_handler import OutputHandlerParameters
from result_handling.pickle_handler import PickleHandlerParameters
from visualization.visualize_lattice import VisualizerParameters
import numpy as np


class Parameters:
    # ----- Lattice Properties -----
    # --- Loading Lattice ---
    # If [load_lattice] is True then the lattice object will be loaded from [load_lattice_pickle_file]
    # Otherwise, a new lattice is created, and pickled into [save_lattice_pickle_file] if [create_pickle] is True
    load_lattice = False
    load_lattice_pickle_file = "prebuilt_lattices/triangular_100_101.pickle"

    # --- Generating new lattice ---
    # The following is for if the lattice is to be generated [load_lattice] = False)
    # Type of lattice:
    # (1) Kagome
    # (2) Triangular
    # (3) Square
    lattice_type: int = 2
    # Length of the lattice (typically number of nodes across). Must be int
    lattice_length: int = 10
    # Height of the lattice. May be int or float
    lattice_height: float = 10
    # Generic networks have slight displacements in node positions
    is_generic: bool = False
    d_shift: float = 0.0
    # Whether to add hinges between bonds - significantly adds to computation cost
    is_hinged: bool = False

    # ----- Bond Occupation Protocol -----
    # There are two protocols which are chosen based on [bond_occupation_protocol]:
    #   (1) Start a lattice with occupation [starting_prob_fill_remove] and remove [num_remove_bonds] bonds until the
    #       shear modulus is below [obj_tolerance]
    #   (2) Start a lattice with [starting_prob_fill_add] and add [num_add_bonds] bonds until bond occupation is
    #       [ending_prob_fill_add]
    #   (3) "Water Basin" approach: each bond is assigned a random number based on a seed. We then gradually
    #        increase p by [p_increment] and add the bond if it is greater than the seeded number
    #   (4) For each value in [prob_fills], reset the network and set the bond occupation to that value
    bond_occupation_protocol = 3
    # When we remove bonds, this is our stopping criteria: shear modulus is below this value
    obj_tolerance = 1e-6

    # --- Protocol 1 ---
    # Number of bonds to remove until shear modulus is zero
    num_remove_bonds: int = 50
    starting_prob_fill_remove: float = 1.0

    # --- Protocol 2 ---
    # Number of bonds to remove until shear modulus is zero
    num_add_bonds: int = 50
    starting_prob_fill_add: float = 0.3
    ending_prob_fill_add = 1.0

    # --- Protocol 3 ---
    seeds = [981289379013810923809,
    76105755378799601034376478149262058939,
    56324004661304621616150438374880780458,
    192167151297898667818033108618101795342,
    124167404520322653616097106459376495677,
    288966371549046310712576224175446077013,
    153569925971408292292414104523914595267,
    117182576888865872704532995033641503886,
    159775907590579164738534376682471282543,
    192613268657560427660936745616063985584,
    287135673689620568799057058918370259575,
    156212570237263731667504162741315464225,
    66769047479415046041815602411726613539,
    301775695896231232901966732679723394699,
    85731610689802507053772074830005515001,
    172858213639021532553672577235622173123,
    1693566961313999119362400946097155897,
    252898110122880037489029938214202068743,
    110594422611875839245714184299727960654,
    296930066437327365906383785394471480133,
    196863263823912190666134199408141000755,
    164225985921347422690819776140353492944,
    70912368558747392175489880119961436417,
    268466687887178491505686246962700966455,
    144648608144972572003955037847235348131,
    307353286227303009901251327192304612356,
    237314775156277832677288644062988551671,
    25455097223718632213298349998027084125,
    31780656961968198487148358991855551404,
    203227608645394658529241362915053610755,
    329174823751330968105306893587244148043,
    96488745731597891402029596220643789871,
    302077485763723826639123134917747291972,
    204310147820392760568928606123385796793,
    54514852140699264660146730596252774122,
    176223841349841241697625820559427810595,
    31610545500055344218642685906926063732,
    148753517427577034874255973200267748783,
    1810556395566387401948790806815949741,
    149770743054903729765646805007340801965,
    48757138470187903142278937452542963174,
    106853713110263411875507473211210151109,
    105694756402585780189237238626245449146,
    91548796538957742957331252640672517744,
    54067419840471604386445437062159779658,
    295222617671056622074195575856015636160,
    19041351450930651447214911431368334809,
    131568230306349725971030511167965191439,
    319187085269473523072981896167800073655,
    219221400980105413985684246754559137065,
    96246367281430923628562346711740767913,
    82269860669028522511940297400568044903,
    138229787241900306745756874083226066408,
    41144599820612278072431002223390858584,
    314920389479721521205372371984971589666,
    220778815303045068449553559507549348536,
    126161680047515929340528347007777297620,
    294942257152695939607497376007432041913,
    174573049914479971056135082268856525940,
    224190464323246362366459044041406572958,
    293042615092313761374381544734613005315,
    89362384290629736311084932670329441015,
    223764082107756832319653887377836070816,
    163015864599050702315519969474983712609,
    280211576649266087869508668369167576989,
    106871716493763000706991300308538848702,
    158839643327039332227649432435841023950,
    83245943842516240829033369138686792196,
    326919272894451060782496143583475431493,
    248094619905032618425158491302555128102,
    164428898366960883299402690013463090806,
    73036015245602380578288086536842835854,
    42698693487654995840214815695002324970,
    252359220349512771771888988203359902495,
    63419768270250909927810402433527446491,
    227103202613474185364255844127702919034,
    112248185106693969937299583857530017724,
    98377412317678441525509925604487912985,
    217648004720885553012875594515235952907,
    314706276249296862794706174239027905916,
    189232720995928024877646410793254057829,
    198572652335447456598162144486999390008,
    4150102060177035807286944487179544697,
    142184260687445856645238323804369345725,
    198642653253590257240421887486891058834,
    277515934209710024851821682524117096520,
    93747306145900666938714259458420684922,
    48303453580563385782659684517056339809,
    170803843482304665497482000675002949084,
    12850631239946333397564887960012515843,
    28294508945949694611302045296334589654,
    157320108379602035319399774173594409914,
    179396560320742330858631463487354629323,
    28733000154164130013974380749931544362,
    53914137174603235215933823181369199021,
    159491411778881128434658393004075187913,
    28695813545673348487745361752224257219,
    79899027799107523988983296828612059932,
    199739548156250475104498630018949515793,
    44505383207454449507532139078297592658]
    random_seed: Optional[int] = seeds[$iseed]
    # Whether to add bonds starting from [low_prob_fill_basin] or remove bonds starting from [high_prob_fill_basin]
    basin_add: bool = False
    # Whether we should fill basin with p, or simply add/remove bonds based on the order they would be added
    use_basin: bool = False
    p_delta: int = 0.01
    # If [ending_prob_fill_basin] is greater than the threshold for a thermodynamically legal network, then threshold
    low_prob_fill_basin: float = 0.6
    high_prob_fill_basin = 1

    # --- Protocol 4 ---
    prob_fills: List[float] = [1.0]

    # ----- Network generation -----
    # Type of network generation (how bonds are added):
    #   (0) Randomly remove bonds (will ignore pc_strength)
    #   (1) Correlated network (addition). Bonds are added with probability (1-pc_strength)^(max_neighbors-N_n)
    #       where N_n is bond neighbors
    #   (2) Correlated network (removal). Bonds are remove with probability 1-[(1-pc_strength)^(max_neighbors-N_n)]
    #   (3) Polarized network (addition). Bond probability (1-pc_strength)^(2-N_n) where N_n is co-linear bond neighbors
    #   (4) Polarized network (removal).
    #   (5) Super polarized network (q=1)
    #   (6) Polarized network (addition) with next-nearest-neighbor
    #   (7) Polarized network (removal) with next-nearest-neighbor
    #   (8) Custom distribution (requires 0 <= pc_strength <= 2/3). Fraction pc_strength are right, 1/3 horizontal, rest
    #       are left
    #   (9) Network contains only strands of bonds
    #   (10) Polarized addition network with parallel-nearest-neighbors
    #   (11) 4-Rule Polarized addition network with parallel-nearest-neighbors with voting (better for high strengths)
    #   (12) 2-Rule Polarized addition network with parallel-nearest-neighbors with voting (better for high strengths)
    #   (13) Probability of a bond is based on the direction that it leans
    #       [pc_strength] right, [1-pc_strength] for left and horizontal
    bond_generation: int = 13

    # For oriented networks, which direction to prefer
    target_direction: int = 0

    # pc_strength is the correlation/polarized strength (0 <= pc_strength < 1)
    # Ignored if bond_generation is 0
    strengths = [1.0, 1.1, 1.2, 1.5, 2.0]
    pc_strength = strengths[${PC_STRENGTH}]

    # ----- Mechanical Properties -----
    # alpha, kappa, mu: stretch, bend, transverse moduli
    stretch_mod = ${ALPHA}
    bend_mod = ${KAPPA}
    tran_mod = ${MU}

    # ----- Energy Minimization -----
    # Method to minimize the energy
    minimization_method: MinimizationType = MinimizationType.LINEAR_PRE
    # Minimization Tolerances: see minimization methods. For linear system this is norm(residual) <= tolerance
    tolerance = 1e-8

    # --- Lattice Shearing ---
    # Direction to shear the lattice (can be used as multiplier for [hor_shear]).
    shear_dir = 1
    # Shear strain (gamma). Each node is sheared by equation: delta_x = hor_shear * (height-lattice_height/2)
    hor_shear = ${STRAIN}

    # ----- Lattice pickling -----
    # All the parameters for handling pickling and visualization output
    pickle_handler_parameters = PickleHandlerParameters(
        create_lattice_pickle=False,
        create_final_pos_pickle=False,
        create_init_pdf=False,
        create_sheared_pdf=False,
        create_final_pdf=False,
        save_lattice_pickle_file="lattice_pickle.p",
        final_pos_pickle_file="final_pos.p",
        init_pos_pdf_file="init_pos.pdf",
        sheared_pos_pdf_file="sheared_pos.pdf",
        final_pos_pdf_file="final_pos.pdf"
    )

    # ----- Output Handling -----
    # (See OutputHandlerParameters for more details)
    today_date = datetime.now().strftime("%m-%d-%y-%H")
    run_folder_name: str = f"seed={str(random_seed)[:7]}-r={pc_strength}"

    output_handler_parameters = OutputHandlerParameters(
        inc_p=True,
        inc_shear_modulus=True,
        inc_ind_energies=True,
        inc_non_affinity=True,
        inc_bond_counts=True,
        inc_backbone_count=True,
        output_path="outputs",
        run_folder_name=run_folder_name,
        output_file="results.csv",
        save_parameters=True
    )

    # ----- Graphics and Postprocessing -----
    visualizer_parameters = VisualizerParameters(
        draw_nodes=False,
        draw_bonds=True,
        draw_pbc=True,
        node_color="black",
        bond_color="black",
        hor_shear=hor_shear
    )
EOF

}

# each run
# shellcheck disable=SC2034

for iseed in {0..0}; do
  for istr in {0..4}; do
    PC_STRENGTH=${istr}
    create_parameters
    ${PYTHON_EXEC} ./main.py
  done
done
