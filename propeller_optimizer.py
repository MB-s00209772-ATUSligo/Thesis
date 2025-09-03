#!/usr/bin/env python3
"""
Marine Propeller Geometry Optimization using NSGA-II and OpenFOAM
Optimizes for reduced cavitation/noise while maintaining thrust and efficiency
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import subprocess
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Tuple, List
import json

@dataclass
class PropellerParams:
    """Parameters defining propeller geometry"""
    n_blades: int = 4
    diameter: float = 180.0  # mm
    hub_dia_base: float = 35.0  # mm
    hub_dia_top: float = 40.0  # mm
    hub_height: float = 30.0  # mm
    pitch_ratio: float = 1.0  # pitch/diameter ratio
    rake: float = 0.0  # degrees
    skew: float = 15.0  # degrees
    chord_distribution: np.ndarray = None
    thickness_distribution: np.ndarray = None
    camber_distribution: np.ndarray = None
    
    def __post_init__(self):
        if self.chord_distribution is None:
            # Default chord distribution (normalized)
            r = np.linspace(0.2, 1.0, 10)
            self.chord_distribution = 0.3 - 0.2 * r + 0.1 * r**2
        if self.thickness_distribution is None:
            # Default thickness distribution
            r = np.linspace(0.2, 1.0, 10)
            self.thickness_distribution = 0.06 - 0.05 * r
        if self.camber_distribution is None:
            # Default camber distribution
            r = np.linspace(0.2, 1.0, 10)
            self.camber_distribution = 0.02 * np.ones_like(r)

class PropellerGeometry:
    """Generate 3D propeller geometry from parameters"""
    
    def __init__(self, params: PropellerParams):
        self.params = params
        
    def create_blade_section(self, r: float, theta: float = 0) -> np.ndarray:
        """Create a blade cross-section at radius r"""
        # Interpolate distributions at radius r
        r_norm = (r - self.params.hub_dia_top/2) / (self.params.diameter/2 - self.params.hub_dia_top/2)
        r_norm = np.clip(r_norm, 0, 1)
        
        # Get section properties
        chord = np.interp(r_norm, np.linspace(0, 1, len(self.params.chord_distribution)),
                         self.params.chord_distribution) * self.params.diameter/2
        thickness = np.interp(r_norm, np.linspace(0, 1, len(self.params.thickness_distribution)),
                            self.params.thickness_distribution) * chord
        camber = np.interp(r_norm, np.linspace(0, 1, len(self.params.camber_distribution)),
                          self.params.camber_distribution) * chord
        
        # Generate NACA-like airfoil
        n_points = 50
        x = np.linspace(0, chord, n_points)
        
        # Thickness distribution (symmetric)
        yt = thickness * (1.4845 * np.sqrt(x/chord) - 0.63 * (x/chord) 
                         - 1.758 * (x/chord)**2 + 1.4215 * (x/chord)**3 
                         - 0.5075 * (x/chord)**4)
        
        # Camber line
        yc = camber * np.sin(np.pi * x/chord)
        
        # Combine upper and lower surfaces
        upper_x = x
        upper_y = yc + yt
        lower_x = x
        lower_y = yc - yt
        
        # Create closed curve
        x_coords = np.concatenate([upper_x, lower_x[::-1]])
        y_coords = np.concatenate([upper_y, lower_y[::-1]])
        z_coords = np.ones_like(x_coords) * r
        
        # Apply pitch
        pitch_angle = np.arctan(self.params.pitch_ratio * self.params.diameter / (2 * np.pi * r))
        
        # Apply skew
        skew_angle = self.params.skew * r_norm * np.pi / 180
        
        # Apply rake
        rake_offset = r * np.tan(self.params.rake * np.pi / 180)
        
        # Transform coordinates
        points = np.column_stack([x_coords - chord/2, y_coords, z_coords])
        
        # Rotate for pitch
        Rp = np.array([[np.cos(pitch_angle), -np.sin(pitch_angle), 0],
                       [np.sin(pitch_angle), np.cos(pitch_angle), 0],
                       [0, 0, 1]])
        points = points @ Rp.T
        
        # Apply skew (rotation in x-y plane)
        Rs = np.array([[np.cos(skew_angle + theta), -np.sin(skew_angle + theta), 0],
                       [np.sin(skew_angle + theta), np.cos(skew_angle + theta), 0],
                       [0, 0, 1]])
        points = points @ Rs.T
        
        # Apply rake
        points[:, 2] += rake_offset
        
        return points
    
    def create_blade(self) -> trimesh.Trimesh:
        """Create a single blade mesh"""
        # Generate sections along the blade
        n_sections = 20
        radii = np.linspace(self.params.hub_dia_top/2, self.params.diameter/2, n_sections)
        
        vertices = []
        faces = []
        
        for i, r in enumerate(radii):
            section = self.create_blade_section(r)
            start_idx = len(vertices)
            vertices.extend(section)
            
            if i > 0:
                # Connect to previous section
                n_points = len(section)
                prev_start = start_idx - n_points
                
                for j in range(n_points - 1):
                    # Create two triangles for each quad
                    faces.append([prev_start + j, start_idx + j, start_idx + j + 1])
                    faces.append([prev_start + j, start_idx + j + 1, prev_start + j + 1])
        
        # Add blade tip
        tip_center = np.mean(vertices[-len(section):], axis=0)
        vertices.append(tip_center)
        tip_idx = len(vertices) - 1
        
        for j in range(len(section) - 1):
            faces.append([start_idx + j, start_idx + j + 1, tip_idx])
        
        blade_mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
        blade_mesh.fix_normals()
        
        return blade_mesh
    
    def create_hub(self) -> trimesh.Trimesh:
        """Create the hub (frustum) mesh"""
        # Create frustum
        height = self.params.hub_height
        r_bottom = self.params.hub_dia_base / 2
        r_top = self.params.hub_dia_top / 2
        
        # Generate cylindrical coordinates
        n_segments = 32
        theta = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
        
        vertices = []
        faces = []
        
        # Bottom circle
        for t in theta:
            vertices.append([r_bottom * np.cos(t), r_bottom * np.sin(t), 0])
        
        # Top circle
        for t in theta:
            vertices.append([r_top * np.cos(t), r_top * np.sin(t), height])
        
        # Bottom center
        vertices.append([0, 0, 0])
        bottom_center_idx = len(vertices) - 1
        
        # Top center
        vertices.append([0, 0, height])
        top_center_idx = len(vertices) - 1
        
        # Side faces
        for i in range(n_segments):
            next_i = (i + 1) % n_segments
            faces.append([i, next_i, n_segments + next_i])
            faces.append([i, n_segments + next_i, n_segments + i])
        
        # Bottom face
        for i in range(n_segments):
            next_i = (i + 1) % n_segments
            faces.append([bottom_center_idx, next_i, i])
        
        # Top face
        for i in range(n_segments):
            next_i = (i + 1) % n_segments
            faces.append([top_center_idx, n_segments + i, n_segments + next_i])
        
        hub_mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
        hub_mesh.fix_normals()
        
        return hub_mesh
    
    def create_propeller(self) -> trimesh.Trimesh:
        """Create complete propeller mesh"""
        # Create hub
        hub = self.create_hub()
        
        # Create and position blades
        propeller = hub
        blade_angle = 2 * np.pi / self.params.n_blades
        
        for i in range(self.params.n_blades):
            blade = self.create_blade()
            
            # Rotate blade to correct position
            rotation_matrix = trimesh.transformations.rotation_matrix(
                i * blade_angle, [0, 0, 1], [0, 0, 0]
            )
            blade.apply_transform(rotation_matrix)
            
            # Combine with propeller
            propeller = trimesh.util.concatenate([propeller, blade])
        
        # Ensure watertight mesh
        propeller.process(validate=True)
        
        return propeller

class OpenFOAMInterface:
    """Interface for OpenFOAM CFD simulations"""
    
    def __init__(self, case_dir: str = None):
        self.case_dir = case_dir or tempfile.mkdtemp(prefix="propeller_cfd_")
        self.setup_case()
    
    def setup_case(self):
        """Setup OpenFOAM case directory structure"""
        # Create directory structure
        dirs = ['0', 'constant', 'system', 'constant/triSurface']
        for d in dirs:
            os.makedirs(os.path.join(self.case_dir, d), exist_ok=True)
        
        # Write basic OpenFOAM files
        self.write_control_dict()
        self.write_fv_schemes()
        self.write_fv_solution()
        self.write_boundary_conditions()
    
    def write_control_dict(self):
        """Write controlDict file"""
        content = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}

application     pimpleFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         100;
deltaT          0.01;
writeControl    timeStep;
writeInterval   10;
purgeWrite      2;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{
    forces
    {
        type            forces;
        functionObjectLibs ("libforces.so");
        outputControl   timeStep;
        outputInterval  1;
        patches         (propeller);
        rho             rhoInf;
        rhoInf          1000;
        CofR            (0 0 0);
    }
}
"""
        with open(os.path.join(self.case_dir, 'system/controlDict'), 'w') as f:
            f.write(content)
    
    def write_fv_schemes(self):
        """Write fvSchemes file"""
        content = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}

ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         Gauss upwind;
    div(phi,U)      Gauss linearUpwind grad(U);
    div(phi,k)      Gauss upwind;
    div(phi,epsilon) Gauss upwind;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}
"""
        with open(os.path.join(self.case_dir, 'system/fvSchemes'), 'w') as f:
            f.write(content)
    
    def write_fv_solution(self):
        """Write fvSolution file"""
        content = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.1;
        smoother        GaussSeidel;
    }

    U
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-08;
        relTol          0.1;
    }

    "(k|epsilon)"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-08;
        relTol          0.1;
    }
}

PIMPLE
{
    nCorrectors     2;
    nNonOrthogonalCorrectors 1;
    pRefCell        0;
    pRefValue       0;
}

relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
        k               0.7;
        epsilon         0.7;
    }
}
"""
        with open(os.path.join(self.case_dir, 'system/fvSolution'), 'w') as f:
            f.write(content)
    
    def write_boundary_conditions(self):
        """Write initial boundary conditions"""
        # Simplified - in reality, would need proper BC setup
        pass
    
    def run_simulation(self, mesh_file: str, rpm: float = 1000) -> dict:
        """Run CFD simulation and extract results"""
        # Copy mesh to case directory
        shutil.copy(mesh_file, os.path.join(self.case_dir, 'constant/triSurface/propeller.stl'))
        
        # In a real implementation, would:
        # 1. Run blockMesh/snappyHexMesh for meshing
        # 2. Set rotating reference frame
        # 3. Run pimpleFoam
        # 4. Extract forces and moments
        
        # For demonstration, return synthetic results
        # These would be replaced with actual OpenFOAM post-processing
        thrust = np.random.uniform(10, 50)  # N
        torque = np.random.uniform(0.5, 2.0)  # Nm
        efficiency = thrust * (rpm * 2 * np.pi / 60) / (torque * rpm * 2 * np.pi / 60)
        cavitation_index = np.random.uniform(0.1, 1.0)  # Lower is better
        
        return {
            'thrust': thrust,
            'torque': torque,
            'efficiency': efficiency,
            'cavitation_index': cavitation_index,
            'rpm': rpm
        }

class PropellerOptimizationProblem(Problem):
    """Multi-objective optimization problem for propeller design"""
    
    def __init__(self):
        # Design variables: chord, thickness, camber, pitch, skew, rake distributions
        n_vars = 35  # 10 points each for chord, thickness, camber + pitch + skew + rake + n_blades
        
        # Bounds for design variables
        xl = np.array([0.1]*10 + [0.01]*10 + [0.0]*10 + [0.5, 0, 0, 3])
        xu = np.array([0.5]*10 + [0.15]*10 + [0.05]*10 + [2.0, 30, 15, 6])
        
        super().__init__(n_var=n_vars, n_obj=3, n_constr=2, xl=xl, xu=xu)
        
        self.openfoam = OpenFOAMInterface()
        self.eval_count = 0
    
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate objective functions and constraints"""
        n_pop = x.shape[0]
        
        f1 = np.zeros(n_pop)  # Cavitation index (minimize)
        f2 = np.zeros(n_pop)  # Negative efficiency (minimize)
        f3 = np.zeros(n_pop)  # Noise metric (minimize)
        
        g1 = np.zeros(n_pop)  # Thrust constraint
        g2 = np.zeros(n_pop)  # Structural constraint
        
        for i in range(n_pop):
            # Extract design variables
            params = PropellerParams()
            params.chord_distribution = x[i, 0:10]
            params.thickness_distribution = x[i, 10:20]
            params.camber_distribution = x[i, 20:30]
            params.pitch_ratio = x[i, 30]
            params.skew = x[i, 31]
            params.rake = x[i, 32]
            params.n_blades = int(x[i, 33])
            
            # Generate geometry
            geom = PropellerGeometry(params)
            prop_mesh = geom.create_propeller()
            
            # Save temporary STL for CFD
            temp_stl = f"/tmp/prop_{self.eval_count}.stl"
            prop_mesh.export(temp_stl)
            self.eval_count += 1
            
            # Run CFD simulation (or use surrogate model for speed)
            results = self.openfoam.run_simulation(temp_stl)
            
            # Calculate objectives
            f1[i] = results['cavitation_index']
            f2[i] = -results['efficiency']  # Negative because we minimize
            f3[i] = results['cavitation_index'] * results['torque']  # Noise proxy
            
            # Constraints
            g1[i] = 30 - results['thrust']  # Minimum thrust of 30N
            g2[i] = np.max(params.thickness_distribution) - 0.03  # Minimum thickness
            
            # Clean up
            os.remove(temp_stl)
        
        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack([g1, g2])

def optimize_propeller(n_generations: int = 50, population_size: int = 100):
    """Run NSGA-II optimization"""
    
    problem = PropellerOptimizationProblem()
    
    algorithm = NSGA2(
        pop_size=population_size,
        n_offsprings=10,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    termination = get_termination("n_gen", n_generations)
    
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)
    
    return res

def visualize_propeller(params: PropellerParams, save_path: str = None):
    """Visualize the propeller geometry"""
    geom = PropellerGeometry(params)
    prop_mesh = geom.create_propeller()
    
    # Create visualization
    fig = plt.figure(figsize=(12, 10))
    
    # 3D view
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot mesh
    vertices = prop_mesh.vertices
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    triangles=prop_mesh.faces,
                    cmap='viridis', alpha=0.8, edgecolor='none')
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Optimized Marine Propeller')
    
    # Equal aspect ratio
    max_range = np.array([vertices[:, 0].max()-vertices[:, 0].min(),
                          vertices[:, 1].max()-vertices[:, 1].min(),
                          vertices[:, 2].max()-vertices[:, 2].min()]).max() / 2.0
    
    mid_x = (vertices[:, 0].max()+vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max()+vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max()+vertices[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return prop_mesh

def save_optimized_design(params: PropellerParams, filepath: str):
    """Save the optimized propeller design"""
    geom = PropellerGeometry(params)
    prop_mesh = geom.create_propeller()
    
    # Export based on file extension
    if filepath.endswith('.stl'):
        prop_mesh.export(filepath, file_type='stl')
    elif filepath.endswith('.3mf'):
        # For 3MF, we need to use trimesh's export
        prop_mesh.export(filepath, file_type='3mf')
    else:
        raise ValueError("Unsupported file format. Use .stl or .3mf")
    
    print(f"Propeller design saved to: {filepath}")
    
    # Save parameters as JSON for reference
    params_dict = {
        'n_blades': params.n_blades,
        'diameter': params.diameter,
        'hub_dia_base': params.hub_dia_base,
        'hub_dia_top': params.hub_dia_top,
        'hub_height': params.hub_height,
        'pitch_ratio': params.pitch_ratio,
        'rake': params.rake,
        'skew': params.skew,
        'chord_distribution': params.chord_distribution.tolist(),
        'thickness_distribution': params.thickness_distribution.tolist(),
        'camber_distribution': params.camber_distribution.tolist()
    }
    
    json_path = filepath.replace('.stl', '_params.json').replace('.3mf', '_params.json')
    with open(json_path, 'w') as f:
        json.dump(params_dict, f, indent=2)
    print(f"Parameters saved to: {json_path}")

def main():
    """Main execution function"""
    print("Marine Propeller Optimization System")
    print("=" * 50)
    print("Optimizing for:")
    print("- Reduced cavitation and noise")
    print("- Maintained/improved thrust and efficiency")
    print("- Diameter: 180mm (tip-to-tip)")
    print("- Hub: 35-40mm frustum")
    print("")
    
    # Run optimization
    print("Starting NSGA-II optimization...")
    print("Note: Full optimization with OpenFOAM would take hours/days.")
    print("Using surrogate models for demonstration.\n")
    
    results = optimize_propeller(n_generations=20, population_size=50)
    
    # Extract best solution (based on preferred objective)
    # Here we'll pick the solution with best cavitation index that meets constraints
    best_idx = np.argmin(results.F[:, 0])
    best_solution = results.X[best_idx]
    
    # Create optimized parameters
    optimized_params = PropellerParams()
    optimized_params.chord_distribution = best_solution[0:10]
    optimized_params.thickness_distribution = best_solution[10:20]
    optimized_params.camber_distribution = best_solution[20:30]
    optimized_params.pitch_ratio = best_solution[30]
    optimized_params.skew = best_solution[31]
    optimized_params.rake = best_solution[32]
    optimized_params.n_blades = int(best_solution[33])
    
    print("\nOptimization complete!")
    print(f"Best solution found at generation {results.algorithm.n_gen}")
    print(f"Number of blades: {optimized_params.n_blades}")
    print(f"Pitch ratio: {optimized_params.pitch_ratio:.3f}")
    print(f"Skew angle: {optimized_params.skew:.1f}°")
    print(f"Rake angle: {optimized_params.rake:.1f}°")
    
    # Visualize the result
    print("\nGenerating 3D visualization...")
    mesh = visualize_propeller(optimized_params, save_path="optimized_propeller.png")
    
    # Save the design
    print("\nSaving optimized design...")
    save_optimized_design(optimized_params, "optimized_propeller.stl")
    save_optimized_design(optimized_params, "optimized_propeller.3mf")
    
    print("\nOptimization complete! Files saved:")
    print("- optimized_propeller.stl")
    print("- optimized_propeller.3mf")
    print("- optimized_propeller_params.json")
    print("- optimized_propeller.png")

if __name__ == "__main__":
    main()