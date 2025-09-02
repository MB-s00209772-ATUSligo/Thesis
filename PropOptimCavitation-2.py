import os
import numpy as np
from deap import base, creator, tools, algorithms
import subprocess
import uuid
import shutil
from cadquery import Workplane

# ---------- CONFIG ----------
FOAM_RUN = "/HPC/matthew.barry/OpenFOAM-10/run"
BASE_CASE = os.path.join(FOAM_RUN, "propellerCase")
CFD_TIMEOUT = 600  # seconds
POP_SIZE = 50
NGEN = 30
HUB_DIAMETER = 49.5
PROPELLER_DIAMETER = 180
# ----------------------------

# ---------- LOAD OPENFOAM ENVIRONMENT ----------
foam_env = os.environ.copy()
foam_env["FOAM_RUN"] = FOAM_RUN
foam_env["FOAM_INST_DIR"] = "/HPC/matthew.barry/OpenFOAM-10"
foam_env["PATH"] = "/HPC/matthew.barry/OpenFOAM-10/platforms/linux64GccDPInt32Opt/bin:" + foam_env["PATH"]
# -----------------------------------------------

# ---------- GEOMETRY GENERATION ----------
def generate_geometry(params, output_path):
    pitch, chord, skew, rake, thickness = params
    blade = (
        Workplane("XY")
        .moveTo(HUB_DIAMETER / 2, 0)
        .lineTo(HUB_DIAMETER / 2 + chord, 0)
        .threePointArc((HUB_DIAMETER / 2 + chord / 2, thickness), (HUB_DIAMETER / 2, 0))
        .revolve(angleDegrees=360 / 3, axisStart=(0, 0, 0), axisEnd=(0, 0, 1))
    )
    blade.val().exportStl(output_path)
# ----------------------------------------

# ---------- PARSERS ----------
def read_force(case_dir, filename):
    try:
        filepath = os.path.join(case_dir, "postProcessing", "forceCoeffs", "0", filename)
        with open(filepath, 'r') as f:
            lines = [line for line in f if not line.startswith("#")]
        last_line = lines[-1].split()
        return float(last_line[2])  # Adjust index if needed
    except:
        return 0.0

def read_cavitation_level(case_dir):
    # TODO: Replace this placeholder with parsing from e.g. alpha.vapor field
    return np.random.random() * 0.2
# ----------------------------------------

# ---------- FITNESS FUNCTION ----------
def evaluate(individual):
    sim_id = str(uuid.uuid4())[:8]
    case_dir = os.path.join(FOAM_RUN, f"prop_case_{sim_id}")
    stl_path = os.path.join(case_dir, "constant", "triSurface", "prop.stl")

    try:
        # Setup case
        shutil.copytree(BASE_CASE, case_dir)
        os.makedirs(os.path.dirname(stl_path), exist_ok=True)
        generate_geometry(individual, stl_path)

        # Run CFD steps
        subprocess.run(["blockMesh"], cwd=case_dir, env=foam_env, timeout=CFD_TIMEOUT, check=True)
        subprocess.run(["snappyHexMesh", "-overwrite"], cwd=case_dir, env=foam_env, timeout=CFD_TIMEOUT, check=True)
        subprocess.run(["cavitatingFoam"], cwd=case_dir, env=foam_env, timeout=CFD_TIMEOUT, check=True)

        # Parse results
        thrust = read_force(case_dir, "forceCoeffs.dat")
        vapour = read_cavitation_level(case_dir)

        return thrust, -vapour  # Maximize thrust, minimize vapour

    except Exception as e:
        print(f"[ERROR] {e}")
        return 0.0, -1.0

    finally:
        shutil.rmtree(case_dir, ignore_errors=True)
# ----------------------------------------

# ---------- NSGA-II SETUP ----------
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))  # thrust↑, cavitation↓
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, 10, 100)  # Tune bounds if needed
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 5)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.4)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

def main():
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(5)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    print("Starting optimization...")
    pop, logbook = algorithms.eaMuPlusLambda(
        pop, toolbox,
        mu=POP_SIZE, lambda_=POP_SIZE,
        cxpb=0.7, mutpb=0.3,
        ngen=NGEN,
        stats=stats,
        halloffame=hof,
        verbose=True
    )
    return pop, logbook, hof

if __name__ == "__main__":
    main()
