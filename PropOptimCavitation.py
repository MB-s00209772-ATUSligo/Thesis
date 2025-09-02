import os

import subprocess

import shutil

import time

import uuid

from datetime import datetime

from deap import base, creator, tools, algorithms

import random

import numpy as np

import cadquery as cq

from cadquery import exporters



# === CONFIGURATION ===



# Paths

BASE_CASE_DIR = "/HPC/matthew.barry/OpenFOAM-10/run/propellerCase"

RESULTS_DIR = "/HPC/matthew.barry/results"

TRISURFACE_FILENAME = "propeller-innerCylinder.stl"



# OpenFOAM Binaries

BLOCKMESH_PATH = "/HPC/matthew.barry/OpenFOAM-10/platforms/linux64GccDPInt32Opt/bin/blockMesh"

SNAPPYHEXMESH_PATH = "/HPC/matthew.barry/OpenFOAM-10/platforms/linux64GccDPInt32Opt/bin/snappyHexMesh"

CAVITATINGFOAM_PATH = "/HPC/matthew.barry/OpenFOAM-10/platforms/linux64GccDPInt32Opt/bin/cavitatingFoam"



# Environment

FOAM_ENV = os.environ.copy()

FOAM_ENV["FOAM_RUN"] = BASE_CASE_DIR



# === PROP GEOMETRY CLASS ===



class Propeller:

    def __init__(self, pitch, chord, skew):

        self.pitch = pitch

        self.chord = chord

        self.skew = skew



    def generate_cad(self):

        # Simplified placeholder propeller CAD

        obj = cq.Workplane("XY").circle(self.chord).extrude(self.pitch)

        return obj



    def export_stl(self, output_path):

        model = self.generate_cad()

        exporters.export(model, output_path)

        print(f"[INFO] Exported STL: {output_path}")



# === OPENFOAM SIMULATION FUNCTION ===



def run_openfoam_simulation(case_dir):

    constant_triSurface_dir = os.path.join(case_dir, "constant", "triSurface")

    os.makedirs(constant_triSurface_dir, exist_ok=True)



    # Create symlink to STL

    stl_source = os.path.join(case_dir, TRISURFACE_FILENAME)

    stl_dest = os.path.join(constant_triSurface_dir, TRISURFACE_FILENAME)

    if os.path.exists(stl_dest):

        os.remove(stl_dest)

    os.symlink(stl_source, stl_dest)



    # Copy base case files

    for subdir in ["system", "0", "constant"]:

        src = os.path.join(BASE_CASE_DIR, subdir)

        dst = os.path.join(case_dir, subdir)

        if not os.path.exists(dst):

            shutil.copytree(src, dst, dirs_exist_ok=True)



    # Run OpenFOAM commands

    try:

        subprocess.run([BLOCKMESH_PATH], cwd=case_dir, env=FOAM_ENV, check=True)

        subprocess.run([SNAPPYHEXMESH_PATH, "-overwrite"], cwd=case_dir, env=FOAM_ENV, check=True)

        subprocess.run([CAVITATINGFOAM_PATH], cwd=case_dir, env=FOAM_ENV, check=True)

    except subprocess.CalledProcessError as e:

        print(f"[ERROR] OpenFOAM command failed: {e}")

        return False, {"thrust": 0, "noise": 1e6}



    # Dummy result extraction (replace with actual OpenFOAM log parsing)

    return True, {"thrust": random.uniform(10, 100), "noise": random.uniform(0, 1)}



# === EVALUATION FUNCTION ===



def evaluate(individual):

    pitch, chord, skew = individual

    prop = Propeller(pitch, chord, skew)



    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    case_id = f"case_{timestamp}_{uuid.uuid4().hex[:4]}"

    case_dir = os.path.join(RESULTS_DIR, case_id)

    os.makedirs(case_dir, exist_ok=True)



    stl_path = os.path.join(case_dir, TRISURFACE_FILENAME)

    prop.export_stl(stl_path)



    sim_ok, metrics = run_openfoam_simulation(case_dir)

    thrust = metrics["thrust"]

    noise = metrics["noise"]



    return -thrust, noise



# === DEAP SETUP ===



creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))

creator.create("Individual", list, fitness=creator.FitnessMulti)



toolbox = base.Toolbox()

toolbox.register("attr_pitch", random.uniform, 0.02, 0.08)

toolbox.register("attr_chord", random.uniform, 0.01, 0.04)

toolbox.register("attr_skew", random.uniform, -15, 15)

toolbox.register("individual", tools.initCycle, creator.Individual,

                 (toolbox.attr_pitch, toolbox.attr_chord, toolbox.attr_skew), n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)

toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=[0.02, 0.01, -15], up=[0.08, 0.04, 15], eta=15.0)

toolbox.register("mutate", tools.mutPolynomialBounded, low=[0.02, 0.01, -15], up=[0.08, 0.04, 15], eta=20.0, indpb=0.2)

toolbox.register("select", tools.selNSGA2)



# === MAIN FUNCTION ===



def main():

    print("Starting optimization...")

    pop = toolbox.population(n=4)

    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)

    stats.register("avg", np.mean, axis=0)

    stats.register("min", np.min, axis=0)

    stats.register("max", np.max, axis=0)



    population, logbook = algorithms.eaMuPlusLambda(pop, toolbox,

                                                    mu=4,

                                                    lambda_=8,

                                                    cxpb=0.6,

                                                    mutpb=0.3,

                                                    ngen=3,

                                                    stats=stats,

                                                    halloffame=hof,

                                                    verbose=True)



    best = hof[0]

    print("\nBest design found:")

    print(f"Pitch: {best[0]:.4f}, Chord: {best[1]:.4f}, Skew: {best[2]:.2f}")

    print(f"Fitness (thrust, noise): {best.fitness.values}")

    return best, logbook



if __name__ == "__main__":

    main()


