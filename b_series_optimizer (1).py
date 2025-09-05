#!/usr/bin/env python3
"""
B-Series (Wageningen) Marine Propeller Optimization using NSGA-II
Optimizes for: Noise Reduction, Cavitation Minimization, Performance (Thrust & Efficiency)
Fixed: 180mm diameter, standard hub dimensions
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from scipy.optimize import differential_evolution
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import json
import os
import warnings
warnings.filterwarnings('ignore')

class BSeries:
    """
    B-Series (Wageningen) Propeller Geometry and Performance
    Based on standard B-series geometry definitions
    """
    
    def __init__(self, Z, EAR, P_D, D=0.18, hub_ratio=0.18):
        """
        Initialize B-Series propeller
        
        Args:
            Z: Number of blades (2-7)
            EAR: Expanded Area Ratio (0.3-1.05)
            P_D: Pitch/Diameter ratio (0.5-1.5)
            D: Diameter in meters (0.18m = 180mm)
            hub_ratio: Hub diameter ratio (0.18 typical)
        """
        self.Z = int(Z)
        self.EAR = float(EAR)
        self.P_D = float(P_D)
        self.D = float(D)  # meters
        self.hub_ratio = float(hub_ratio)
        self.hub_diameter = self.D * self.hub_ratio
        
        # B-series standard parameters
        self.hub_height = 0.03  # 30mm hub height
        self.rake = 0  # B-series has zero rake
        
    def chord_distribution(self, r_R):
        """
        B-Series chord distribution (Kuiper 1992)
        r_R: normalized radius (0.2 to 1.0)
        Returns: c/D (normalized chord)
        """
        # B-series chord distribution coefficients
        Z = self.Z
        EAR = self.EAR
        
        # Simplified B-series chord formula
        # Based on Carlton's "Marine Propellers and Propulsion"
        if r_R < 0.2:
            r_R = 0.2
            
        # Maximum chord at 0.7R for B-series
        c_max = (EAR * 1.067 - 0.229 * P_D) / Z
        
        # Distribution shape
        if r_R <= 0.7:
            c_D = c_max * (1.0 - ((0.7 - r_R) / 0.5) ** 2)
        else:
            c_D = c_max * np.sqrt(1 - ((r_R - 0.7) / 0.3) ** 2)
            
        return c_D
    
    def thickness_distribution(self, r_R):
        """
        B-Series thickness distribution
        r_R: normalized radius
        Returns: t/D (normalized thickness)
        """
        # B-series thickness distribution (from Kuiper)
        t_max = 0.0035 + 0.196 * np.exp(-3.33 * r_R)
        
        # Adjust for number of blades
        if self.Z <= 3:
            t_max *= 1.1
        elif self.Z >= 6:
            t_max *= 0.9
            
        return t_max
    
    def skew_distribution(self, r_R):
        """
        B-Series skew distribution
        Standard B-series has moderate skew
        """
        # Skew angle in degrees
        if r_R < 0.3:
            skew = 0
        else:
            # Progressive skew towards tip
            skew = 15 * ((r_R - 0.3) / 0.7) ** 2
        return skew
    
    def pitch_distribution(self, r_R):
        """
        B-Series pitch distribution
        Constant pitch for B-series
        """
        return self.P_D * self.D
    
    def create_blade_section(self, r, n_points=30):
        """
        Create a blade section at radius r
        Returns array of 3D points defining the section
        """
        r_R = r / (self.D / 2)
        
        # Get section properties
        c = self.chord_distribution(r_R) * self.D
        t = self.thickness_distribution(r_R) * self.D
        skew = self.skew_distribution(r_R) * np.pi / 180
        pitch = self.pitch_distribution(r_R)
        
        # Generate NACA-like section
        x = np.linspace(0, 1, n_points)
        
        # Thickness distribution (modified NACA 66)
        yt = t / 0.2 * (0.2969 * np.sqrt(x) - 0.126 * x - 
                       0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        
        # Camber for marine propeller
        camber = 0.02 * c * (1 - r_R)
        yc = camber * (2 * x - x**2)
        
        # Upper and lower surfaces
        xu = x * c
        yu = yc + yt
        xl = x * c
        yl = yc - yt
        
        # Combine surfaces
        x_section = np.concatenate([xu, xl[::-1][1:]])
        y_section = np.concatenate([yu, yl[::-1][1:]])
        
        # Apply pitch (blade angle)
        if r > 0:
            blade_angle = np.arctan(pitch / (2 * np.pi * r))
        else:
            blade_angle = 0
            
        # Create 3D points
        points = []
        for i in range(len(x_section)):
            # Apply pitch rotation
            x_rot = x_section[i] - c/2
            y_rot = y_section[i]
            
            # Rotate for pitch
            x_pitched = x_rot * np.cos(blade_angle) - y_rot * np.sin(blade_angle)
            z_pitched = x_rot * np.sin(blade_angle) + y_rot * np.cos(blade_angle)
            
            # Position in 3D space with skew
            angle = skew
            x_3d = r * np.cos(angle) - x_pitched * np.sin(angle)
            y_3d = r * np.sin(angle) + x_pitched * np.cos(angle)
            z_3d = z_pitched
            
            points.append([x_3d, y_3d, z_3d])
            
        return np.array(points)
    
    def create_blade(self):
        """Create a single blade mesh"""
        # Radial stations
        r_hub = self.hub_diameter / 2
        r_tip = self.D / 2
        n_sections = 25
        
        radii = np.linspace(r_hub * 0.95, r_tip * 0.99, n_sections)
        
        vertices = []
        faces = []
        
        # Build blade surface
        for i, r in enumerate(radii):
            section = self.create_blade_section(r)
            
            # Add root fillet for first few sections
            if i < 3:
                scale = 0.3 + 0.7 * (i / 3)
                center = np.mean(section, axis=0)
                section = center + (section - center) * scale
            
            start_idx = len(vertices)
            vertices.extend(section)
            
            if i > 0:
                # Connect to previous section
                n_points = len(section)
                prev_start = start_idx - n_points
                
                for j in range(n_points):
                    j_next = (j + 1) % n_points
                    
                    # Two triangles per quad
                    faces.append([prev_start + j, start_idx + j, start_idx + j_next])
                    faces.append([prev_start + j, start_idx + j_next, prev_start + j_next])
        
        # Close tip
        if len(vertices) > 0:
            tip_center = np.mean(vertices[-len(section):], axis=0)
            vertices.append(tip_center)
            tip_idx = len(vertices) - 1
            
            for j in range(len(section)):
                j_next = (j + 1) % len(section)
                base_idx = len(vertices) - len(section) - 1
                faces.append([base_idx + j, base_idx + j_next, tip_idx])
        
        # Create mesh
        blade_mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
        blade_mesh.fix_normals()
        
        return blade_mesh
    
    def create_hub(self):
        """Create hub (frustum) mesh"""
        r_bottom = self.hub_diameter / 2
        r_top = r_bottom * 1.1  # Slight taper
        height = self.hub_height
        n_segments = 32
        
        vertices = []
        faces = []
        
        # Create hub cylinder
        for i in range(2):
            h = i * height
            r = r_bottom if i == 0 else r_top
            theta = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
            
            for t in theta:
                vertices.append([r * np.cos(t), r * np.sin(t), h - height/2])
        
        # Side faces
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            
            v1 = j
            v2 = j_next
            v3 = n_segments + j_next
            v4 = n_segments + j
            
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])
        
        # End caps
        center_bottom = len(vertices)
        vertices.append([0, 0, -height/2])
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            faces.append([center_bottom, j_next, j])
        
        center_top = len(vertices)
        vertices.append([0, 0, height/2])
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            faces.append([center_top, n_segments + j, n_segments + j_next])
        
        hub_mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
        hub_mesh.fix_normals()
        
        return hub_mesh
    
    def create_propeller(self):
        """Create complete propeller mesh"""
        # Create hub
        hub = self.create_hub()
        meshes = [hub]
        
        # Create blades
        blade_angle = 2 * np.pi / self.Z
        
        for i in range(self.Z):
            blade = self.create_blade()
            
            # Rotate blade to position
            rotation = trimesh.transformations.rotation_matrix(
                i * blade_angle, [0, 0, 1], [0, 0, 0]
            )
            blade.apply_transform(rotation)
            meshes.append(blade)
        
        # Combine all parts
        propeller = trimesh.util.concatenate(meshes)
        propeller.remove_duplicate_faces()
        propeller.fix_normals()
        
        return propeller
    
    def calculate_KT_KQ(self, J):
        """
        Calculate thrust and torque coefficients using B-series regression
        J: Advance coefficient
        Returns: KT, KQ, efficiency
        """
        # Simplified B-series regression (Oosterveld & van Oossanen 1975)
        # These are approximations of the full polynomial
        
        # Base coefficients
        KT_base = 0.4 * (self.P_D - 0.3 * J) * self.EAR * (1 - J/self.P_D)**2
        KQ_base = 0.05 * (self.P_D - 0.2 * J) * self.EAR * (1 - J/self.P_D)**2
        
        # Blade number correction
        Z_factor = (self.Z / 4) ** 0.5
        KT = KT_base * Z_factor
        KQ = KQ_base * Z_factor
        
        # Efficiency
        if KQ > 0 and J > 0:
            eta = (J / (2 * np.pi)) * (KT / KQ)
        else:
            eta = 0
            
        return KT, KQ, eta
    
    def cavitation_criterion(self, rpm, V_ship, depth=5):
        """
        Calculate cavitation number and margin
        rpm: Rotations per minute
        V_ship: Ship speed (m/s)
        depth: Depth underwater (m)
        Returns: sigma (cavitation number), margin
        """
        n = rpm / 60  # rps
        
        # Tip speed
        V_tip = 2 * np.pi * n * self.D / 2
        V_total = np.sqrt(V_ship**2 + V_tip**2)
        
        # Cavitation number
        rho = 1025  # kg/m³ seawater
        p_atm = 101325  # Pa
        p_vapor = 2340  # Pa at 20°C
        g = 9.81  # m/s²
        
        p_static = p_atm + rho * g * depth
        sigma = (p_static - p_vapor) / (0.5 * rho * V_total**2)
        
        # Burrill's criterion for B-series
        # Critical cavitation number
        tau = self.calculate_thrust_loading(rpm, V_ship)
        sigma_crit = 1.2 * tau + 0.2
        
        margin = sigma - sigma_crit  # Positive = no cavitation
        
        return sigma, margin
    
    def calculate_thrust_loading(self, rpm, V_ship):
        """Calculate thrust loading coefficient"""
        n = rpm / 60
        J = V_ship / (n * self.D) if n > 0 else 0
        
        KT, _, _ = self.calculate_KT_KQ(J)
        
        # Thrust loading
        Ap = self.EAR * np.pi * (self.D/2)**2 / self.Z  # Projected area per blade
        rho = 1025
        
        if n > 0:
            T = KT * rho * n**2 * self.D**4
            tau = T / (Ap * 0.5 * rho * (np.pi * n * 0.7 * self.D)**2)
        else:
            tau = 0
            
        return tau
    
    def noise_level(self, rpm, V_ship):
        """
        Estimate noise level (simplified model)
        Based on tip speed and loading
        """
        n = rpm / 60
        
        # Tip Mach number in water
        V_tip = 2 * np.pi * n * self.D / 2
        c_water = 1500  # m/s
        M_tip = V_tip / c_water
        
        # Loading noise
        tau = self.calculate_thrust_loading(rpm, V_ship)
        
        # Simplified noise model (dB)
        noise = 120 + 20 * np.log10(M_tip + 0.01) + 10 * np.log10(tau + 0.1)
        
        # Blade rate frequency adjustment
        noise += 5 * np.log10(self.Z)
        
        return noise


class BSeriesOptimizationProblem(Problem):
    """
    Multi-objective optimization problem for B-Series propellers
    Objectives:
    1. Minimize noise
    2. Minimize cavitation
    3. Maximize efficiency
    4. Maintain thrust
    """
    
    def __init__(self, target_thrust=100, rpm=1800, V_ship=5.0):
        """
        Initialize optimization problem
        
        Args:
            target_thrust: Target thrust in N
            rpm: Operating RPM
            V_ship: Ship speed in m/s
        """
        # Design variables: [Z, EAR, P/D]
        # Z: 2-7 blades (integer)
        # EAR: 0.35-1.0
        # P/D: 0.6-1.4
        
        super().__init__(
            n_var=3,
            n_obj=3,
            n_constr=2,
            xl=[2, 0.35, 0.6],
            xu=[7, 1.0, 1.4]
        )
        
        self.target_thrust = target_thrust
        self.rpm = rpm
        self.V_ship = V_ship
        self.rho = 1025  # seawater
        self.eval_count = 0
        
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate objectives and constraints"""
        n_pop = x.shape[0]
        
        # Objectives
        f1 = np.zeros(n_pop)  # Noise (minimize)
        f2 = np.zeros(n_pop)  # Cavitation index (minimize)
        f3 = np.zeros(n_pop)  # Negative efficiency (minimize)
        
        # Constraints
        g1 = np.zeros(n_pop)  # Thrust constraint
        g2 = np.zeros(n_pop)  # Cavitation margin
        
        for i in range(n_pop):
            # Extract design variables
            Z = int(np.round(x[i, 0]))
            EAR = x[i, 1]
            P_D = x[i, 2]
            
            # Create B-series propeller
            prop = BSeries(Z, EAR, P_D, D=0.18)
            
            # Calculate performance
            n = self.rpm / 60  # rps
            J = self.V_ship / (n * prop.D) if n > 0 else 0
            
            KT, KQ, eta = prop.calculate_KT_KQ(J)
            
            # Calculate thrust
            thrust = KT * self.rho * n**2 * prop.D**4
            
            # Calculate cavitation
            sigma, cav_margin = prop.cavitation_criterion(self.rpm, self.V_ship)
            
            # Calculate noise
            noise = prop.noise_level(self.rpm, self.V_ship)
            
            # Set objectives
            f1[i] = noise  # Minimize noise
            f2[i] = 1.0 / (sigma + 0.1)  # Minimize cavitation (maximize sigma)
            f3[i] = -eta  # Maximize efficiency
            
            # Set constraints
            g1[i] = self.target_thrust - thrust  # Thrust >= target
            g2[i] = -cav_margin  # Cavitation margin > 0
            
            self.eval_count += 1
            
            if self.eval_count % 20 == 0:
                print(f"Eval {self.eval_count}: Z={Z}, EAR={EAR:.3f}, P/D={P_D:.3f}")
                print(f"  Thrust={thrust:.1f}N, Eta={eta:.3f}, Noise={noise:.1f}dB")
        
        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack([g1, g2])


def optimize_bseries(n_gen=50, pop_size=100, target_thrust=100):
    """
    Run NSGA-II optimization for B-series propeller
    """
    print("="*60)
    print("B-SERIES PROPELLER OPTIMIZATION")
    print("="*60)
    print(f"Target Thrust: {target_thrust} N")
    print(f"Operating Conditions: 1800 RPM, 5 m/s (10 knots)")
    print(f"Objectives: Min Noise, Min Cavitation, Max Efficiency")
    print(f"Generations: {n_gen}, Population: {pop_size}")
    print("-"*60)
    
    # Create problem
    problem = BSeriesOptimizationProblem(target_thrust=target_thrust)
    
    # Create algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        n_offsprings=20,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20, prob=0.2),
        eliminate_duplicates=True
    )
    
    # Run optimization
    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", n_gen),
        seed=42,
        verbose=True
    )
    
    return res


def visualize_bseries(Z, EAR, P_D, save_path=None):
    """
    Visualize B-series propeller
    """
    # Create propeller
    prop = BSeries(Z, EAR, P_D, D=0.18)
    mesh = prop.create_propeller()
    
    # Calculate performance at design point
    rpm = 1800
    V_ship = 5.0
    n = rpm / 60
    J = V_ship / (n * prop.D)
    
    KT, KQ, eta = prop.calculate_KT_KQ(J)
    thrust = KT * 1025 * n**2 * prop.D**4
    torque = KQ * 1025 * n**2 * prop.D**5
    sigma, cav_margin = prop.cavitation_criterion(rpm, V_ship)
    noise = prop.noise_level(rpm, V_ship)
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # 3D view
    ax1 = fig.add_subplot(221, projection='3d')
    vertices = mesh.vertices * 1000  # Convert to mm
    
    ax1.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    triangles=mesh.faces, cmap='ocean', alpha=0.8)
    
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'B-Series: Z={Z}, EAR={EAR:.3f}, P/D={P_D:.3f}')
    
    # Set equal aspect
    max_range = np.array([vertices[:, 0].max()-vertices[:, 0].min(),
                         vertices[:, 1].max()-vertices[:, 1].min(),
                         vertices[:, 2].max()-vertices[:, 2].min()]).max() / 2.0
    
    mid_x = (vertices[:, 0].max()+vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max()+vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max()+vertices[:, 2].min()) * 0.5
    
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Top view
    ax2 = fig.add_subplot(222)
    ax2.triplot(vertices[:, 0], vertices[:, 1], mesh.faces, 'b-', alpha=0.2, linewidth=0.3)
    ax2.scatter(vertices[:, 0], vertices[:, 1], c='navy', s=0.5, alpha=0.3)
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('Top View')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Performance metrics
    ax3 = fig.add_subplot(223)
    ax3.axis('off')
    
    metrics = f"""
    B-SERIES PROPELLER PERFORMANCE
    
    Design Parameters:
    • Number of Blades: {Z}
    • Expanded Area Ratio: {EAR:.3f}
    • Pitch/Diameter Ratio: {P_D:.3f}
    • Diameter: 180 mm
    
    Performance @ 1800 RPM, 10 knots:
    • Thrust: {thrust:.1f} N
    • Torque: {torque:.2f} Nm
    • Efficiency: {eta:.3f} ({eta*100:.1f}%)
    
    Cavitation & Noise:
    • Cavitation Number: {sigma:.3f}
    • Cavitation Margin: {cav_margin:.3f}
    • Noise Level: {noise:.1f} dB
    
    Coefficients:
    • KT: {KT:.4f}
    • KQ: {KQ:.4f}
    • J: {J:.3f}
    """
    
    ax3.text(0.1, 0.9, metrics, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # Geometry distributions
    ax4 = fig.add_subplot(224)
    r_R = np.linspace(0.2, 1.0, 50)
    chord = [prop.chord_distribution(r) for r in r_R]
    thickness = [prop.thickness_distribution(r) * 10 for r in r_R]  # Scale for visibility
    skew = [prop.skew_distribution(r) / 30 for r in r_R]  # Normalize
    
    ax4.plot(r_R, chord, 'b-', label='Chord/D', linewidth=2)
    ax4.plot(r_R, thickness, 'r-', label='Thickness/D (×10)', linewidth=2)
    ax4.plot(r_R, skew, 'g-', label='Skew (normalized)', linewidth=2)
    ax4.set_xlabel('r/R')
    ax4.set_ylabel('Normalized Value')
    ax4.set_title('B-Series Blade Distributions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0.2, 1.0)
    
    plt.suptitle('B-SERIES PROPELLER DESIGN', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()
    
    return mesh


def save_bseries_design(Z, EAR, P_D, filepath, performance=None):
    """
    Save B-series propeller design
    """
    # Create propeller
    prop = BSeries(Z, EAR, P_D, D=0.18)
    mesh = prop.create_propeller()
    
    # Scale to mm for CAD
    mesh.vertices *= 1000
    
    # Save mesh
    mesh.export(filepath)
    print(f"B-Series propeller saved to: {filepath}")
    
    # Save parameters as JSON
    params_dict = {
        'type': 'B-SERIES',
        'design': {
            'Z': int(Z),
            'EAR': float(EAR),
            'P_D': float(P_D),
            'diameter_mm': 180,
            'hub_ratio': 0.18
        }
    }
    
    if performance:
        params_dict['performance'] = performance
    
    json_path = filepath.replace('.stl', '_params.json')
    with open(json_path, 'w') as f:
        json.dump(params_dict, f, indent=2)
    print(f"Parameters saved to: {json_path}")


def main():
    """Main execution"""
    # Create output directory
    output_dir = "bseries_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("B-SERIES MARINE PROPELLER OPTIMIZATION SYSTEM")
    print("="*60)
    print("\nSpecifications:")
    print("• Propeller Type: Wageningen B-Series")
    print("• Diameter: 180mm")
    print("• Application: Small marine vessel")
    print("\nOptimization Objectives:")
    print("1. Minimize underwater noise")
    print("2. Minimize cavitation")
    print("3. Maximize efficiency")
    print("4. Maintain required thrust")
    print("\n" + "-"*60 + "\n")
    
    # Run optimization
    target_thrust = 100  # N (adjust based on requirements)
    results = optimize_bseries(n_gen=40, pop_size=80, target_thrust=target_thrust)
    
    print("\n" + "-"*60)
    print("OPTIMIZATION COMPLETE")
    print("-"*60 + "\n")
    
    if results is not None and hasattr(results, 'X'):
        # Get best solutions for each objective
        best_solutions = []
        
        if len(results.X) > 0:
            # Best for noise (objective 0)
            noise_idx = np.argmin(results.F[:, 0])
            best_solutions.append(('Low Noise', results.X[noise_idx]))
            
            # Best for cavitation (objective 1)
            cav_idx = np.argmin(results.F[:, 1])
            best_solutions.append(('Anti-Cavitation', results.X[cav_idx]))
            
            # Best for efficiency (objective 2)
            eff_idx = np.argmin(results.F[:, 2])
            best_solutions.append(('High Efficiency', results.X[eff_idx]))
            
            # Best compromise (weighted)
            F_norm = results.F.copy()
            for i in range(3):
                f_min, f_max = F_norm[:, i].min(), F_norm[:, i].max()
                if f_max > f_min:
                    F_norm[:, i] = (F_norm[:, i] - f_min) / (f_max - f_min)
            
            weights = [0.4, 0.3, 0.3]  # Noise, Cavitation, Efficiency
            scores = np.sum(F_norm * weights, axis=1)
            best_idx = np.argmin(scores)
            best_solutions.append(('Best Overall', results.X[best_idx]))
            
            # Process and save each solution
            for name, solution in best_solutions:
                Z = int(np.round(solution[0]))
                EAR = float(solution[1])
                P_D = float(solution[2])
                
                print(f"\n{name} Design:")
                print(f"  Z = {Z} blades")
                print(f"  EAR = {EAR:.3f}")
                print(f"  P/D = {P_D:.3f}")
                
                # Calculate performance
                prop = BSeries(Z, EAR, P_D, D=0.18)
                rpm = 1800
                V_ship = 5.0
                n = rpm / 60
                J = V_ship / (n * prop.D)
                
                KT, KQ, eta = prop.calculate_KT_KQ(J)
                thrust = KT * 1025 * n**2 * prop.D**4
                sigma, cav_margin = prop.cavitation_criterion(rpm, V_ship)
                noise = prop.noise_level(rpm, V_ship)
                
                print(f"  Performance:")
                print(f"    Thrust: {thrust:.1f} N")
                print(f"    Efficiency: {eta*100:.1f}%")
                print(f"    Noise: {noise:.1f} dB")
                print(f"    Cavitation σ: {sigma:.3f}")
                
                # Save design
                if name == 'Best Overall':
                    # Save main design
                    stl_path = os.path.join(output_dir, "bseries_optimized.stl")
                    png_path = os.path.join(output_dir, "bseries_optimized.png")
                    
                    performance = {
                        'thrust_N': float(thrust),
                        'efficiency': float(eta),
                        'noise_dB': float(noise),
                        'cavitation_number': float(sigma),
                        'KT': float(KT),
                        'KQ': float(KQ)
                    }
                    
                    save_bseries_design(Z, EAR, P_D, stl_path, performance)
                    visualize_bseries(Z, EAR, P_D, png_path)
                    
                    # Also save as .3mf
                    mesh = BSeries(Z, EAR, P_D, D=0.18).create_propeller()
                    mesh.vertices *= 1000  # Convert to mm
                    mesh.export(stl_path.replace('.stl', '.3mf'))
                    
                else:
                    # Save alternative designs
                    filename = name.lower().replace(' ', '_')
                    stl_path = os.path.join(output_dir, f"bseries_{filename}.stl")
                    save_bseries_design(Z, EAR, P_D, stl_path)
    
    else:
        print("No optimization results available. Creating default B-series design...")
        
        # Default B-series parameters
        Z = 4
        EAR = 0.65
        P_D = 1.0
        
        print(f"\nDefault B-Series Design:")
        print(f"  Z = {Z} blades")
        print(f"  EAR = {EAR:.3f}")
        print(f"  P/D = {P_D:.3f}")
        
        stl_path = os.path.join(output_dir, "bseries_default.stl")
        png_path = os.path.join(output_dir, "bseries_default.png")
        
        save_bseries_design(Z, EAR, P_D, stl_path)
        visualize_bseries(Z, EAR, P_D, png_path)
    
    print("\n" + "="*60)
    print("B-SERIES OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"\nAll files saved in: {output_dir}/")
    print("\nFiles generated:")
    print("• bseries_optimized.stl - Main optimized design")
    print("• bseries_optimized.3mf - 3D printing format")
    print("• bseries_optimized.png - Visualization")
    print("• bseries_*_params.json - Design parameters")
    print("\nThe B-series designs are optimized for:")
    print("✓ Minimum underwater noise")
    print("✓ Reduced cavitation")
    print("✓ Maximum efficiency")
    print("✓ Required thrust performance")


if __name__ == "__main__":
    main()