import os
import shutil
import subprocess
import numpy as np
from cadquery import Workplane
from deap import base, creator, tools, algorithms

# ========== CONFIGURATION ==========
HUB_DIAMETER = 49.5  # mm
PROP_DIAMETER = 180.0  # mm
BLADE_COUNT_RANGE = (3, 6)  # Min to max number of blades
BASE_CASE_PATH = "/HPC/matthew.barry/OpenFOAM-10/run/baseCavitatingFoamCase"
TRISURF_PATH = "constant/triSurface"

# ========== GEOMETRY GENERATION ==========
def generate_geometry(params, output_path, num_blades):
    pitch, chord, skew, rake, thickness = params
    blade = (
        Workplane("XY")
        .moveTo(HUB_DIAMETER / 2, 0)
        .lineTo(HUB_DIAMETER / 2 + chord, 0)
        .threePointArc((HUB_DIAMETER / 2 + chord / 2, thickness), (HUB_DIAMETER / 2, 0))
        .revolve(angleDegrees=360 / num_blades, axisStart=(0, 0, 0), axisEnd=(0, 0, 1))
    )
    blade.val().exportStl(output_path)

# ========== OPENFOAM AUTOMATION ==========
def setup_openfoam_case(ind_id, stl_path):
    case_dir = f"/HPC/matthew.barry/OpenFOAM-10/run/case_{ind_id}"
    shutil.copytree(BASE_CASE_PATH, case_dir)
    tri_dir = os.path.join(case_dir, TRISURF_PATH)
    os.makedirs(tri_dir, exist_ok=True)
    shutil.copy(stl_path, os.path.join(tri_dir, "propeller.stl"))
    return case_dir

def run_openfoam(case_dir):
    subprocess.run(["blockMesh"], cwd=case_dir)
    subprocess.run(["surfaceFeatureExtract"], cwd=case_dir)
    subprocess.run(["snappyHexMesh", "-overwrite"], cwd=case_dir)
    subprocess.run(["cavitatingFoam"], cwd=case_dir)

def extract_results(case_dir):
    try:
        with open(os.path.join(case_dir, "postProcessing/forces/0/forceCoeffs.dat")) as f:
            lines = f.readlines()
            last = lines[-1].split()
            thrust = float(last[2])  # Adjust if Z-force is elsewhere
    except:
        thrust = -9999.0

    try:
        with open(os.path.join(case_dir, "log.cavitatingFoam")) as f:
            content = f.read()
            cav = content.count("Cavitation")  # Replace with better metric
    except:
        cav = 9999.0

    return thrust, cav

# ========== FITNESS EVALUATION ==========
def evaluate(ind):
    blade_count = np.random.randint(BLADE_COUNT_RANGE[0], BLADE_COUNT_RANGE[1]+1)
    ind_id = hash(tuple(ind)) % 1000000
    stl_path = f"/HPC/matthew.barry/OpenFOAM-10/run/geometry_{ind_id}.stl"

    try:
        generate_geometry(ind, stl_path, blade_count)
        case_dir = setup_openfoam_case(ind_id, stl_path)
        run_openfoam(case_dir)
        thrust, cav = extract_results(case_dir)
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)
        os.remove(stl_path)

    return thrust, -cav

# ========== NSGA-II SETUP ==========
POP_SIZE = 50
NGEN = 40
BOUNDS = [(10, 40), (10, 30), (-20, 20), (-10, 10), (1, 10)]

creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_float", lambda: np.random.uniform(*BOUNDS[np.random.randint(0, len(BOUNDS))]))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=5)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=[b[0] for b in BOUNDS], high=[b[1] for b in BOUNDS], eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=[b[0] for b in BOUNDS], high=[b[1] for b in BOUNDS], eta=20.0, indpb=0.2)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", evaluate)

def main():
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    algorithms.eaMuPlusLambda(pop, toolbox, mu=POP_SIZE, lambda_=POP_SIZE,
                              cxpb=0.6, mutpb=0.3, ngen=NGEN, stats=stats,
                              halloffame=hof, verbose=True)

    return pop, stats, hof

if __name__ == "__main__":
    print("Starting optimization...")
    main()
