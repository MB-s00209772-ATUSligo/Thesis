import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import xml.etree.ElementTree as ET
import base64
import struct
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Try to import trimesh, fall back to custom implementation if not available
try:
    import trimesh
    TRIMESH_AVAILABLE = True
    print("Using trimesh library for mesh generation")
except ImportError:
    TRIMESH_AVAILABLE = False
    print("trimesh not available, using built-in mesh generation")
    print("To install trimesh: pip install trimesh")
    
    # Custom minimal mesh class
    class CustomMesh:
        def __init__(self, vertices, faces):
            self.vertices = np.array(vertices)
            self.faces = np.array(faces)
            
        def export(self, file_type='stl_ascii'):
            if file_type == 'stl_ascii':
                return self._export_stl_ascii()
            return ""
            
        def _export_stl_ascii(self):
            stl_content = "solid propeller\n"
            for face in self.faces:
                v1, v2, v3 = self.vertices[face]
                normal = np.cross(v2 - v1, v3 - v1)
                normal = normal / np.linalg.norm(normal)
                
                stl_content += f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n"
                stl_content += "    outer loop\n"
                stl_content += f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n"
                stl_content += f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n"
                stl_content += f"      vertex {v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}\n"
                stl_content += "    endloop\n"
                stl_content += "  endfacet\n"
            stl_content += "endsolid propeller\n"
            return stl_content
            
        def remove_duplicate_faces(self):
            # Simple duplicate removal
            unique_faces = []
            for face in self.faces:
                if not any(np.array_equal(face, uf) for uf in unique_faces):
                    unique_faces.append(face)
            self.faces = np.array(unique_faces)
            
        def remove_unreferenced_vertices @dataclass
class PropellerGeometry:
    """Represents propeller geometry parameters"""
    num_blades: int = 3
    chord_distribution: List[float] = None  # Chord length at different radial positions
    pitch_distribution: List[float] = None  # Pitch at different radial positions
    thickness_distribution: List[float] = None  # Thickness at different radial positions
    skew_distribution: List[float] = None  # Blade skew distribution
    rake_angle: float = 0.0  # Blade rake angle
    
    def __post_init__(self):
        if self.chord_distribution is None:
            self.chord_distribution = [0.15, 0.12, 0.08, 0.05, 0.02]
        if self.pitch_distribution is None:
            self.pitch_distribution = [0.8, 0.9, 1.0, 1.1, 1.2]
        if self.thickness_distribution is None:
            self.thickness_distribution = [0.08, 0.06, 0.04, 0.03, 0.02]
        if self.skew_distribution is None:
            self.skew_distribution = [0.0, 0.1, 0.15, 0.12, 0.08]

class PropellerPerformanceModel:
    """Simplified propeller performance and noise model"""
    
    def __init__(self, radius=0.09):  # 90mm radius
        self.radius = radius
        self.rho = 1025  # Water density kg/m³
        self.operating_speed = 1500  # RPM
        self.advance_speed = 5.0  # m/s
        
    def calculate_performance(self, geometry: PropellerGeometry) -> Tuple[float, float]:
        """Calculate thrust coefficient and efficiency"""
        # Simplified momentum theory with corrections
        
        # Radial positions (normalized)
        r_positions = np.linspace(0.2, 1.0, len(geometry.chord_distribution))
        
        # Calculate local advance ratio
        J = self.advance_speed / (self.operating_speed / 60 * 2 * self.radius)
        
        # Thrust calculation using blade element theory approximation
        thrust_coefficient = 0.0
        torque_coefficient = 0.0
        
        for i, r_norm in enumerate(r_positions):
            r = r_norm * self.radius
            chord = geometry.chord_distribution[i] * self.radius
            pitch_ratio = geometry.pitch_distribution[i]
            thickness_ratio = geometry.thickness_distribution[i]
            
            # Local advance angle
            phi = np.arctan(J / (np.pi * r_norm))
            
            # Simplified lift coefficient (depends on geometry)
            cl = 2 * np.pi * (pitch_ratio * np.pi / 8 - phi) * (1 - thickness_ratio)
            
            # Simplified drag coefficient
            cd = 0.01 + thickness_ratio**2 * 0.1
            
            # Local thrust and torque contributions
            dr = self.radius / len(r_positions)
            dT = 0.5 * self.rho * (self.operating_speed * 2 * np.pi / 60 * r)**2 * chord * cl * dr
            dQ = 0.5 * self.rho * (self.operating_speed * 2 * np.pi / 60 * r)**2 * chord * cd * r * dr
            
            thrust_coefficient += dT * geometry.num_blades
            torque_coefficient += dQ * geometry.num_blades
        
        # Normalize coefficients
        n = self.operating_speed / 60
        thrust_coefficient /= self.rho * n**2 * (2 * self.radius)**4
        torque_coefficient /= self.rho * n**2 * (2 * self.radius)**5
        
        # Efficiency
        if torque_coefficient > 0:
            efficiency = J * thrust_coefficient / (2 * np.pi * torque_coefficient)
        else:
            efficiency = 0.0
            
        return thrust_coefficient, min(efficiency, 1.0)
    
    def calculate_noise_metrics(self, geometry: PropellerGeometry) -> float:
        """Calculate noise-related metrics (lower is better)"""
        # Noise sources: thickness noise, loading noise, quadrupole noise
        
        r_positions = np.linspace(0.2, 1.0, len(geometry.chord_distribution))
        
        # Thickness noise (proportional to blade thickness and tip speed)
        tip_speed = self.operating_speed * 2 * np.pi / 60 * self.radius
        thickness_noise = 0.0
        
        for i, r_norm in enumerate(r_positions):
            thickness = geometry.thickness_distribution[i]
            chord = geometry.chord_distribution[i]
            local_speed = tip_speed * r_norm
            
            # Thickness noise contribution
            thickness_noise += thickness * chord * local_speed**3 * r_norm
        
        # Loading noise (proportional to thrust variations)
        loading_noise = 0.0
        for i in range(len(geometry.pitch_distribution)-1):
            pitch_gradient = abs(geometry.pitch_distribution[i+1] - geometry.pitch_distribution[i])
            loading_noise += pitch_gradient * geometry.chord_distribution[i]
        
        # Tip vortex noise (related to tip loading and geometry)
        tip_loading = geometry.chord_distribution[-1] * geometry.pitch_distribution[-1]
        tip_noise = tip_loading * tip_speed**2
        
        # Skew effect on noise (skew can reduce noise)
        skew_factor = 1.0 - np.mean(geometry.skew_distribution) * 0.3
        
        total_noise = (thickness_noise + loading_noise * 0.5 + tip_noise) * skew_factor
        
        return total_noise

class NSGA2Optimizer:
    """NSGA-II multi-objective optimizer for propeller design"""
    
    def __init__(self, performance_model: PropellerPerformanceModel, 
                 population_size=50, generations=100):
        self.performance_model = performance_model
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.fronts = []
        
    def create_individual(self) -> PropellerGeometry:
        """Create a random propeller geometry"""
        # Parameter bounds
        chord_bounds = (0.02, 0.20)  # 2-20cm chord
        pitch_bounds = (0.6, 1.4)   # Pitch ratio
        thickness_bounds = (0.01, 0.10)  # Thickness ratio
        skew_bounds = (0.0, 0.25)   # Skew distribution
        
        geometry = PropellerGeometry()
        geometry.num_blades = random.choice([3, 4, 5])  # 3-5 blades
        
        # Generate smooth distributions
        n_sections = 5
        geometry.chord_distribution = self._generate_smooth_distribution(
            chord_bounds, n_sections, decreasing=True)
        geometry.pitch_distribution = self._generate_smooth_distribution(
            pitch_bounds, n_sections, decreasing=False)
        geometry.thickness_distribution = self._generate_smooth_distribution(
            thickness_bounds, n_sections, decreasing=True)
        geometry.skew_distribution = self._generate_smooth_distribution(
            skew_bounds, n_sections, decreasing=False)
        
        geometry.rake_angle = random.uniform(-15, 15)  # degrees
        
        return geometry
    
    def _generate_smooth_distribution(self, bounds, n_sections, decreasing=True):
        """Generate smooth distribution along blade span"""
        base_values = [random.uniform(bounds[0], bounds[1]) for _ in range(n_sections)]
        
        if decreasing:
            # Sort in decreasing order for chord/thickness
            base_values.sort(reverse=True)
        else:
            # Add some smoothness constraint
            base_values.sort()
            if random.random() > 0.5:
                base_values.reverse()
        
        # Apply smoothing
        smoothed = []
        for i in range(n_sections):
            if i == 0 or i == n_sections - 1:
                smoothed.append(base_values[i])
            else:
                # Weighted average with neighbors
                smoothed.append(0.5 * base_values[i] + 
                              0.25 * base_values[i-1] + 
                              0.25 * base_values[i+1])
        
        return smoothed
    
    def evaluate_objectives(self, geometry: PropellerGeometry) -> Tuple[float, float]:
        """Evaluate objectives: minimize noise, maximize efficiency"""
        try:
            thrust_coeff, efficiency = self.performance_model.calculate_performance(geometry)
            noise_level = self.performance_model.calculate_noise_metrics(geometry)
            
            # Objectives: minimize noise, maximize efficiency
            # Convert efficiency maximization to minimization
            obj1 = noise_level
            obj2 = -efficiency  # Negative for minimization
            
            return obj1, obj2
        except:
            # Return worst possible values for invalid geometries
            return 1e6, 1e6
    
    def dominates(self, obj1: Tuple[float, float], obj2: Tuple[float, float]) -> bool:
        """Check if obj1 dominates obj2"""
        return (obj1[0] <= obj2[0] and obj1[1] <= obj2[1] and
                (obj1[0] < obj2[0] or obj1[1] < obj2[1]))
    
    def fast_non_dominated_sort(self, objectives: List[Tuple[float, float]]) -> List[List[int]]:
        """Fast non-dominated sorting"""
        n = len(objectives)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self.dominates(objectives[i], objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self.dominates(objectives[j], objectives[i]):
                        domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        front_idx = 0
        while len(fronts[front_idx]) > 0:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            
            if next_front:
                fronts.append(next_front)
            front_idx += 1
        
        return fronts[:-1] if not fronts[-1] else fronts
    
    def calculate_crowding_distance(self, front: List[int], 
                                  objectives: List[Tuple[float, float]]) -> List[float]:
        """Calculate crowding distance for solutions in a front"""
        if len(front) <= 2:
            return [float('inf')] * len(front)
        
        distances = [0.0] * len(front)
        n_obj = len(objectives[0])
        
        for obj_idx in range(n_obj):
            # Sort by objective value
            front_sorted = sorted(front, key=lambda x: objectives[x][obj_idx])
            
            # Set boundary points to infinity
            obj_values = [objectives[i][obj_idx] for i in front_sorted]
            obj_range = max(obj_values) - min(obj_values)
            
            if obj_range > 0:
                distances[front.index(front_sorted[0])] = float('inf')
                distances[front.index(front_sorted[-1])] = float('inf')
                
                for i in range(1, len(front_sorted) - 1):
                    idx = front.index(front_sorted[i])
                    distances[idx] += (obj_values[i+1] - obj_values[i-1]) / obj_range
        
        return distances
    
    def crossover(self, parent1: PropellerGeometry, 
                  parent2: PropellerGeometry) -> PropellerGeometry:
        """Create offspring through crossover"""
        child = PropellerGeometry()
        
        # Blade number crossover
        child.num_blades = random.choice([parent1.num_blades, parent2.num_blades])
        
        # Distribution crossover
        alpha = random.random()
        child.chord_distribution = [
            alpha * p1 + (1-alpha) * p2 
            for p1, p2 in zip(parent1.chord_distribution, parent2.chord_distribution)
        ]
        child.pitch_distribution = [
            alpha * p1 + (1-alpha) * p2 
            for p1, p2 in zip(parent1.pitch_distribution, parent2.pitch_distribution)
        ]
        child.thickness_distribution = [
            alpha * p1 + (1-alpha) * p2 
            for p1, p2 in zip(parent1.thickness_distribution, parent2.thickness_distribution)
        ]
        child.skew_distribution = [
            alpha * p1 + (1-alpha) * p2 
            for p1, p2 in zip(parent1.skew_distribution, parent2.skew_distribution)
        ]
        
        child.rake_angle = alpha * parent1.rake_angle + (1-alpha) * parent2.rake_angle
        
        return child
    
    def mutate(self, individual: PropellerGeometry, mutation_rate=0.1) -> PropellerGeometry:
        """Apply mutation to individual"""
        if random.random() > mutation_rate:
            return individual
        
        mutated = PropellerGeometry(
            num_blades=individual.num_blades,
            chord_distribution=individual.chord_distribution.copy(),
            pitch_distribution=individual.pitch_distribution.copy(),
            thickness_distribution=individual.thickness_distribution.copy(),
            skew_distribution=individual.skew_distribution.copy(),
            rake_angle=individual.rake_angle
        )
        
        # Randomly mutate one parameter
        mutation_type = random.randint(0, 4)
        
        if mutation_type == 0:  # Chord distribution
            idx = random.randint(0, len(mutated.chord_distribution) - 1)
            mutated.chord_distribution[idx] *= random.uniform(0.8, 1.2)
            mutated.chord_distribution[idx] = max(0.02, min(0.20, mutated.chord_distribution[idx]))
        
        elif mutation_type == 1:  # Pitch distribution
            idx = random.randint(0, len(mutated.pitch_distribution) - 1)
            mutated.pitch_distribution[idx] *= random.uniform(0.9, 1.1)
            mutated.pitch_distribution[idx] = max(0.6, min(1.4, mutated.pitch_distribution[idx]))
        
        elif mutation_type == 2:  # Thickness distribution
            idx = random.randint(0, len(mutated.thickness_distribution) - 1)
            mutated.thickness_distribution[idx] *= random.uniform(0.8, 1.2)
            mutated.thickness_distribution[idx] = max(0.01, min(0.10, mutated.thickness_distribution[idx]))
        
        elif mutation_type == 3:  # Skew distribution
            idx = random.randint(0, len(mutated.skew_distribution) - 1)
            mutated.skew_distribution[idx] += random.uniform(-0.05, 0.05)
            mutated.skew_distribution[idx] = max(0.0, min(0.25, mutated.skew_distribution[idx]))
        
        else:  # Rake angle
            mutated.rake_angle += random.uniform(-5, 5)
            mutated.rake_angle = max(-15, min(15, mutated.rake_angle))
        
        return mutated
    
    def optimize(self) -> Tuple[List[PropellerGeometry], List[Tuple[float, float]]]:
        """Run NSGA-II optimization"""
        print("Initializing population...")
        
        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]
        
        best_solutions = []
        best_objectives = []
        
        for generation in range(self.generations):
            print(f"Generation {generation + 1}/{self.generations}")
            
            # Evaluate objectives
            objectives = [self.evaluate_objectives(ind) for ind in population]
            
            # Non-dominated sorting
            fronts = self.fast_non_dominated_sort(objectives)
            
            # Select best solutions from first front
            if generation % 10 == 0:  # Store progress every 10 generations
                first_front_solutions = [population[i] for i in fronts[0]]
                first_front_objectives = [objectives[i] for i in fronts[0]]
                best_solutions.extend(first_front_solutions)
                best_objectives.extend(first_front_objectives)
            
            # Create next generation
            new_population = []
            
            # Elitism: keep best solutions
            front_idx = 0
            while len(new_population) < self.population_size and front_idx < len(fronts):
                front = fronts[front_idx]
                
                if len(new_population) + len(front) <= self.population_size:
                    new_population.extend([population[i] for i in front])
                else:
                    # Calculate crowding distance and select best
                    distances = self.calculate_crowding_distance(front, objectives)
                    sorted_front = sorted(zip(front, distances), key=lambda x: x[1], reverse=True)
                    
                    remaining = self.population_size - len(new_population)
                    new_population.extend([population[i] for i, _ in sorted_front[:remaining]])
                
                front_idx += 1
            
            # Generate offspring through crossover and mutation
            offspring = []
            while len(offspring) < self.population_size:
                parent1 = random.choice(new_population)
                parent2 = random.choice(new_population)
                
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                offspring.append(child)
            
            population = offspring[:self.population_size]
        
        # Final evaluation
        final_objectives = [self.evaluate_objectives(ind) for ind in population]
        final_fronts = self.fast_non_dominated_sort(final_objectives)
        
        # Return Pareto front
        pareto_solutions = [population[i] for i in final_fronts[0]]
        pareto_objectives = [final_objectives[i] for i in final_fronts[0]]
        
        return pareto_solutions, pareto_objectives

class PropellerMeshGenerator:
    """Generate 3D mesh of propeller geometry"""
    
    def __init__(self, radius=0.09):
        self.radius = radius
        
    def generate_blade_profile(self, r_norm: float, chord: float, 
                             thickness: float, pitch: float) -> np.ndarray:
        """Generate blade cross-section at given radial position"""
        n_points = 50
        
        # NACA-like airfoil profile
        x = np.linspace(0, 1, n_points)
        
        # Thickness distribution (modified NACA equation)
        yt = thickness * 5 * (0.2969 * np.sqrt(x) - 0.1260 * x - 
                             0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        
        # Create upper and lower surfaces
        xu = x * chord
        yu_upper = yt
        yu_lower = -yt
        
        # Combine to form closed profile
        profile_x = np.concatenate([xu, xu[::-1]])
        profile_y = np.concatenate([yu_upper, yu_lower[::-1]])
        profile_z = np.zeros_like(profile_x)
        
        return np.column_stack([profile_x, profile_y, profile_z])
    
    def generate_propeller_mesh(self, geometry: PropellerGeometry) -> trimesh.Trimesh:
        """Generate complete propeller mesh"""
        all_vertices = []
        all_faces = []
        vertex_offset = 0
        
        # Hub geometry
        hub_radius = 0.15 * self.radius
        hub_height = 0.1 * self.radius
        
        # Generate hub
        hub = trimesh.creation.cylinder(radius=hub_radius, height=hub_height)
        hub.apply_translation([0, 0, -hub_height/2])
        
        all_vertices.extend(hub.vertices)
        all_faces.extend(hub.faces)
        vertex_offset = len(hub.vertices)
        
        # Generate blades
        r_positions = np.linspace(0.2, 1.0, len(geometry.chord_distribution))
        
        for blade_idx in range(geometry.num_blades):
            blade_angle = blade_idx * 2 * np.pi / geometry.num_blades
            
            # Generate blade sections
            blade_sections = []
            for i, r_norm in enumerate(r_positions):
                r = r_norm * self.radius
                chord = geometry.chord_distribution[i] * self.radius
                thickness = geometry.thickness_distribution[i] * chord
                pitch = geometry.pitch_distribution[i]
                skew = geometry.skew_distribution[i] * self.radius
                
                # Generate profile
                profile = self.generate_blade_profile(r_norm, chord, thickness, pitch)
                
                # Apply transformations
                # Pitch rotation
                pitch_angle = np.arctan(pitch * 2 * np.pi * r / (2 * np.pi * r))
                rotation_matrix = np.array([
                    [np.cos(pitch_angle), -np.sin(pitch_angle), 0],
                    [np.sin(pitch_angle), np.cos(pitch_angle), 0],
                    [0, 0, 1]
                ])
                profile = profile @ rotation_matrix.T
                
                # Apply skew (forward sweep)
                profile[:, 1] += skew
                
                # Position at radius
                profile[:, 0] += r
                
                # Rotate around hub
                rotation_blade = np.array([
                    [np.cos(blade_angle), -np.sin(blade_angle), 0],
                    [np.sin(blade_angle), np.cos(blade_angle), 0],
                    [0, 0, 1]
                ])
                profile = profile @ rotation_blade.T
                
                blade_sections.append(profile)
            
            # Create blade mesh by connecting sections
            for i in range(len(blade_sections) - 1):
                section1 = blade_sections[i]
                section2 = blade_sections[i + 1]
                
                # Create triangular faces between sections
                n_points = len(section1)
                
                for j in range(n_points - 1):
                    # Two triangles per quad
                    v1 = vertex_offset + j
                    v2 = vertex_offset + j + 1
                    v3 = vertex_offset + n_points + j
                    v4 = vertex_offset + n_points + j + 1
                    
                    all_faces.append([v1, v2, v3])
                    all_faces.append([v2, v4, v3])
                
                all_vertices.extend(section1)
                vertex_offset += len(section1)
            
            # Add last section
            all_vertices.extend(blade_sections[-1])
            vertex_offset += len(blade_sections[-1])
        
        # Create mesh
        vertices = np.array(all_vertices)
        faces = np.array(all_faces)
        
        try:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            mesh.fix_normals()
            return mesh
        except:
            # Fallback: create simple hub if blade generation fails
            return hub

def create_3mf_file(mesh: trimesh.Trimesh, filename: str):
    """Create 3MF file from trimesh"""
    # Convert to STL first, then embed in 3MF structure
    stl_data = mesh.export(file_type='stl_ascii')
    
    # Create 3MF XML structure
    root = ET.Element("model", unit="millimeter")
    root.set("xmlns", "http://schemas.microsoft.com/3dmanufacturing/core/2015/02")
    
    # Resources
    resources = ET.SubElement(root, "resources")
    
    # Object
    obj = ET.SubElement(resources, "object", id="1", type="model")
    
    # Mesh
    mesh_elem = ET.SubElement(obj, "mesh")
    
    # Vertices
    vertices_elem = ET.SubElement(mesh_elem, "vertices")
    for vertex in mesh.vertices:
        v_elem = ET.SubElement(vertices_elem, "vertex")
        v_elem.set("x", f"{vertex[0]*1000:.3f}")  # Convert to mm
        v_elem.set("y", f"{vertex[1]*1000:.3f}")
        v_elem.set("z", f"{vertex[2]*1000:.3f}")
    
    # Triangles
    triangles_elem = ET.SubElement(mesh_elem, "triangles")
    for face in mesh.faces:
        t_elem = ET.SubElement(triangles_elem, "triangle")
        t_elem.set("v1", str(face[0]))
        t_elem.set("v2", str(face[1]))
        t_elem.set("v3", str(face[2]))
    
    # Build
    build = ET.SubElement(root, "build")
    item = ET.SubElement(build, "item", objectid="1")
    
    # Write to file
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(filename, encoding='utf-8', xml_declaration=True)
    
    print(f"3MF file saved as: {filename}")

def main():
    """Main optimization and mesh generation pipeline"""
    print("Marine Propeller Optimization using NSGA-II")
    print("=" * 50)
    
    # Initialize performance model
    performance_model = PropellerPerformanceModel(radius=0.09)  # 90mm
    
    # Initialize optimizer
    optimizer = NSGA2Optimizer(performance_model, population_size=30, generations=50)
    
    # Run optimization
    print("Starting optimization...")
    pareto_solutions, pareto_objectives = optimizer.optimize()
    
    print(f"\nOptimization complete! Found {len(pareto_solutions)} Pareto optimal solutions.")
    
    # Find the best compromise solution (closest to ideal point)
    # Normalize objectives for fair comparison
    noise_values = [obj[0] for obj in pareto_objectives]
    efficiency_values = [-obj[1] for obj in pareto_objectives]  # Convert back to positive
    
    min_noise = min(noise_values)
    max_noise = max(noise_values)
    min_eff = min(efficiency_values)
    max_eff = max(efficiency_values)
    
    best_idx = 0
    best_distance = float('inf')
    
    for i, (noise, eff) in enumerate(zip(noise_values, efficiency_values)):
        # Normalize and calculate distance to ideal point (0, 1)
        norm_noise = (noise - min_noise) / (max_noise - min_noise) if max_noise > min_noise else 0
        norm_eff = (eff - min_eff) / (max_eff - min_eff) if max_eff > min_eff else 0
        
        distance = np.sqrt(norm_noise**2 + (1 - norm_eff)**2)
        
        if distance < best_distance:
            best_distance = distance
            best_idx = i
    
    best_solution = pareto_solutions[best_idx]
    best_objectives = pareto_objectives[best_idx]
    
    print(f"\nBest compromise solution:")
    print(f"Number of blades: {best_solution.num_blades}")
    print(f"Noise level: {best_objectives[0]:.4f}")
    print(f"Efficiency: {-best_objectives[1]:.4f}")
    print(f"Chord distribution: {[f'{c:.3f}' for c in best_solution.chord_distribution]}")
    print(f"Pitch distribution: {[f'{p:.3f}' for p in best_solution.pitch_distribution]}")
    
    # Generate mesh for best solution
    print("\nGenerating 3D mesh...")
    mesh_generator = PropellerMeshGenerator(radius=0.09)
    propeller_mesh = mesh_generator.generate_propeller_mesh(best_solution)
    
    print(f"Mesh generated: {len(propeller_mesh.vertices)} vertices, {len(propeller_mesh.faces)} faces")
    
    # Create 3MF file
    create_3mf_file(propeller_mesh, "optimized_propeller.3mf")
    
    # Plot Pareto front
    plt.figure(figsize=(10, 6))
    plt.scatter(noise_values, efficiency_values, alpha=0.7, s=50)
    plt.scatter(noise_values[best_idx], efficiency_values[best_idx], 
                color='red', s=100, marker='*', label='Best Compromise')
    plt.xlabel('Noise Level (lower is better)')
    plt.ylabel('Efficiency (higher is better)')
    plt.title('Pareto Front: Propeller Noise vs Efficiency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('pareto_front.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary
    print("\n" + "="*50)
    print("OPTIMIZATION SUMMARY")
    print("="*50)
    print(f"Propeller radius: 90mm")
    print(f"Pareto solutions found: {len(pareto_solutions)}")
    print(f"Best solution efficiency: {-best_objectives[1]:.3f}")
    print(f"Best solution noise level: {best_objectives[0]:.3f}")
    print(f"3MF file generated: optimized_propeller.3mf")
    print(f"Pareto front plot saved: pareto_front.png")
    
    return best_solution, propeller_mesh

if __name__ == "__main__":
    main()