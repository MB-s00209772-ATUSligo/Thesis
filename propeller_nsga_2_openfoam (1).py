# NSGA-II Optimization for Marine Propeller Geometry using OpenFOAM
# Objective: Maximize Efficiency, Maximize Thrust, Minimize Noise

import os
import subprocess
import random
import numpy as np
from deap import base, creator, tools, algorithms
import math

# ===============================
# CONFIGURATION
# ===============================
PROPELLER_DIAMETER = 180.0  # mm
HUB_DIAMETER = 49.5         # mm

# Define parameter bounds: [num_blades, pitch, skew, rake, chord_ratio, twist, thickness_ratio]
PARAM_BOUNDS = [(2, 20),     # number of blades
                (15, 45),    # pitch angle in degrees
                (-15, 15),   # skew in degrees
                (-10, 10),   # rake in mm
                (0.1, 0.3),  # chord ratio (chord/diameter)
                (10, 45),    # twist angle in degrees
                (0.02, 0.08)] # thickness ratio (max thickness / chord)

# ===============================
# NACA PROFILE GENERATOR
# ===============================
def naca4_airfoil(naca_code='2412', num_points=100):
    m = int(naca_code[0]) / 100.0
    p = int(naca_code[1]) / 10.0
    t = int(naca_code[2:]) / 100.0

    x = [0.5 * (1 - math.cos(math.pi * i / (num_points - 1))) for i in range(num_points)]
    yt = [5 * t * (0.2969 * math.sqrt(xi) - 0.1260 * xi - 0.3516 * xi**2 + 0.2843 * xi**3 - 0.1015 * xi**4) for xi in x]

    yc = []
    dyc_dx = []
    for xi in x:
        if xi < p and p != 0:
            yc.append(m / (p**2) * (2 * p * xi - xi**2))
            dyc_dx.append(2 * m / (p**2) * (p - xi))
        elif p != 0:
            yc.append(m / ((1 - p)**2) * (1 - 2*p + 2*p*xi - xi**2))
            dyc_dx.append(2 * m / ((1 - p)**2) * (p - xi))
        else:
            yc.append(0)
            dyc_dx.append(0)

    theta = [math.atan(dy) for dy in dyc_dx]

    xu = [xi - yt[i]*math.sin(theta[i]) for i, xi in enumerate(x)]
    yu = [yc[i] + yt[i]*math.cos(theta[i]) for i in range(num_points)]
    xl = [xi + yt[i]*math.sin(theta[i]) for i, xi in enumerate(x)]
    yl = [yc[i] - yt[i]*math.cos(theta[i]) for i in range(num_points)]

    upper = list(zip(xu, yu))
    lower = list(zip(reversed(xl), reversed(yl)))
    return upper + lower

# ===============================
# OBJECTIVE FUNCTIONS
# ===============================
def evaluate_propeller(individual):
    num_blades = int(round(individual[0]))
    pitch, skew, rake, chord_ratio, twist, thickness_ratio = individual[1:]

    # 1. Generate CAD model
    cad_filename = generate_cad(num_blades, pitch, skew, rake, chord_ratio, twist, thickness_ratio)

    # 2. Run OpenFOAM simulation (external)
    try:
        thrust, torque, noise = run_openfoam_simulation(cad_filename)
    except:
        return 0, 0, 1e6  # penalize failed cases

    # 3. Compute efficiency
    omega = 100.0  # rad/s (example)
    V = 2.0        # m/s inflow velocity
    power_in = torque * omega
    power_out = thrust * V
    efficiency = power_out / power_in if power_in > 0 else 0

    return efficiency, thrust, -noise  # noise is minimized

# ===============================
# CAD + OpenFOAM Integration
# ===============================
def generate_cad(num_blades, pitch, skew, rake, chord_ratio, twist, thickness_ratio):
    """Generate a propeller CAD file using FreeCAD. Returns path to .stl file."""
    stl_file = "propeller_temp.stl"
    naca_profile = "2412"  # Use cambered profile, or change to "0012" for symmetric
    cmd = [
        "FreeCADCmd", "generate_propeller.py",
        str(num_blades), str(pitch), str(skew), str(rake),
        str(chord_ratio), str(twist), str(thickness_ratio),
        str(PROPELLER_DIAMETER), str(HUB_DIAMETER), stl_file,
        naca_profile
    ]
    subprocess.run(cmd, check=True)
    return stl_file

def run_openfoam_simulation(stl_file):
    """Run OpenFOAM pipeline and extract performance metrics."""
    bash_script = "run_openfoam_pipeline.sh"
    result_file = "cfd_results.txt"
    env = os.environ.copy()
    env["PROPELLER_STL"] = stl_file
    subprocess.run(["bash", bash_script], check=True, env=env)

    # Parse output from CFD post-processing
    try:
        with open(result_file, 'r') as f:
            lines = f.readlines()
            thrust = float(lines[0].strip())
            torque = float(lines[1].strip())
            noise = float(lines[2].strip())
    except Exception as e:
        raise RuntimeError("Failed to parse CFD results") from e

    return thrust, torque, noise

# ===============================
# NSGA-II SETUP
# ===============================
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
for i, bounds in enumerate(PARAM_BOUNDS):
    toolbox.register(f"attr_{i}", random.uniform, *bounds)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_0, toolbox.attr_1, toolbox.attr_2,
                  toolbox.attr_3, toolbox.attr_4, toolbox.attr_5,
                  toolbox.attr_6), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate_propeller)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=[b[0] for b in PARAM_BOUNDS],
                 up=[b[1] for b in PARAM_BOUNDS], eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=[b[0] for b in PARAM_BOUNDS],
                 up=[b[1] for b in PARAM_BOUNDS], eta=20.0, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# ===============================
# MAIN OPTIMIZATION LOOP
# ===============================
def main():
    pop = toolbox.population(n=10)
    hof = tools.ParetoFront()

    algorithms.eaMuPlusLambda(pop, toolbox, mu=10, lambda_=20, cxpb=0.6, mutpb=0.3,
                              ngen=5, stats=None, halloffame=hof, verbose=True)

    print("Best individuals:")
    for ind in hof:
        print(ind, ind.fitness.values)

if __name__ == '__main__':
    main()
