#!/usr/bin/env python3
"""
Marine Propeller Geometry Optimization using NSGA-II with Built-in CFD
FIXED VERSION: Enhanced parameter ranges and geometric variations for visually distinct results
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
from dataclasses import dataclass
from typing import Tuple, List
import json
from scipy.integrate import simpson
from scipy.interpolate import interp1d
import os
import warnings
warnings.filterwarnings('ignore')

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
    
    def naca_airfoil(self, x, thickness, camber, camber_pos=0.4):
        """Generate NACA 4-digit airfoil coordinates"""
        # Thickness distribution (NACA 00xx)
        yt = 5 * thickness * (
            0.2969 * np.sqrt(x) 
            - 0.1260 * x 
            - 0.3516 * x**2 
            + 0.2843 * x**3 
            - 0.1015 * x**4
        )
        
        # Mean camber line
        if camber == 0:
            yc = np.zeros_like(x)
            dyc_dx = np.zeros_like(x)
        else:
            yc = np.zeros_like(x)
            dyc_dx = np.zeros_like(x)
            
            # Forward of max camber position
            mask = x <= camber_pos
            if camber_pos > 0:
                yc[mask] = camber / camber_pos**2 * (2 * camber_pos * x[mask] - x[mask]**2)
                dyc_dx[mask] = 2 * camber / camber_pos**2 * (camber_pos - x[mask])
            
            # Aft of max camber position
            mask = x > camber_pos
            if camber_pos < 1:
                yc[mask] = camber / (1 - camber_pos)**2 * ((1 - 2*camber_pos) + 2*camber_pos*x[mask] - x[mask]**2)
                dyc_dx[mask] = 2 * camber / (1 - camber_pos)**2 * (camber_pos - x[mask])
        
        # Angle of camber line
        theta = np.arctan(dyc_dx)
        
        # Upper and lower surface coordinates
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)
        
        return xu, yu, xl, yl
        
    def create_blade_section(self, r: float, theta: float = 0) -> np.ndarray:
        """Create a blade cross-section at radius r using complex airfoil geometry"""
        # Normalize radius
        r_norm = (r - self.params.hub_dia_top/2) / (self.params.diameter/2 - self.params.hub_dia_top/2)
        r_norm = np.clip(r_norm, 0, 1)
        
        # Get section properties with smooth transitions
        chord = np.interp(r_norm, np.linspace(0, 1, len(self.params.chord_distribution)),
                         self.params.chord_distribution) * self.params.diameter/2
        
        # ENHANCED: Apply stronger thickness variation
        thickness_ratio = np.interp(r_norm, np.linspace(0, 1, len(self.params.thickness_distribution)),
                                   self.params.thickness_distribution)
        # Extra thickness near root for structural strength
        if r_norm < 0.2:
            thickness_ratio *= (1 + 0.5 * (0.2 - r_norm) / 0.2)  # Increased from 0.3 to 0.5
        
        # ENHANCED: Apply stronger camber variation
        camber_ratio = np.interp(r_norm, np.linspace(0, 1, len(self.params.camber_distribution)),
                                self.params.camber_distribution)
        
        # Generate complex NACA airfoil
        n_points = 30
        x_normalized = np.linspace(0, 1, n_points//2)
        
        # Get NACA coordinates
        xu, yu, xl, yl = self.naca_airfoil(x_normalized, thickness_ratio, camber_ratio)
        
        # Scale by chord length
        xu *= chord
        yu *= chord
        xl *= chord
        yl *= chord
        
        # Combine upper and lower surfaces (closed curve)
        airfoil_x = np.concatenate([xu, xl[::-1][1:]])
        airfoil_y = np.concatenate([yu, yl[::-1][1:]])
        
        # ENHANCED: Stronger pitch angle variation
        pitch_angle = np.arctan(self.params.pitch_ratio * self.params.diameter / (2 * np.pi * r)) if r > 0 else 0
        
        # ENHANCED: More aggressive twist distribution
        twist_factor = 1.0 - 0.5 * r_norm  # Increased from 0.3 to 0.5
        pitch_angle *= twist_factor
        
        # ENHANCED: Stronger skew angle application
        skew_angle = self.params.skew * (0.5 + r_norm) * np.pi / 180  # Progressive skew
        
        # Create 3D points for this section
        points = []
        for i in range(len(airfoil_x)):
            # Center the airfoil
            x_centered = airfoil_x[i] - chord/2
            y_centered = airfoil_y[i]
            
            # Apply pitch rotation (blade twist)
            x_rot = x_centered * np.cos(pitch_angle) - y_centered * np.sin(pitch_angle)
            z_rot = x_centered * np.sin(pitch_angle) + y_centered * np.cos(pitch_angle)
            
            # Position in 3D space (blade extends radially in X-Y plane)
            angle = theta + skew_angle
            
            # ENHANCED: More aggressive sweep near tip
            if r_norm > 0.6:  # Changed from 0.7
                sweep_angle = (r_norm - 0.6) * 0.4 * np.pi / 4  # Increased sweep
                x_rot += chord * 0.3 * np.sin(sweep_angle)
            
            x = r * np.cos(angle) - x_rot * np.sin(angle)
            y = r * np.sin(angle) + x_rot * np.cos(angle)
            z = z_rot + self.params.hub_height/2
            
            # ENHANCED: Stronger rake effect
            if self.params.rake != 0:
                rake_angle = self.params.rake * np.pi / 180
                z += r_norm * np.tan(rake_angle) * self.params.diameter/6  # Increased from /10
                # More pronounced forward lean
                x += r_norm * np.sin(rake_angle) * self.params.diameter/20  # Increased from /30
                y += r_norm * np.cos(rake_angle) * self.params.diameter/40  # Added Y component
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def create_blade_with_fillet(self) -> trimesh.Trimesh:
        """Create a single blade mesh with fillet at root"""
        # Generate blade sections from inside hub to tip
        n_sections = 35  # Increased for smoother geometry
        
        # Start inside the hub for proper intersection
        r_start = self.params.hub_dia_base/2 * 0.8
        r_hub = self.params.hub_dia_top/2
        r_tip = self.params.diameter/2
        
        # Create fillet region with more sections for smooth transition
        fillet_radius = (r_hub - r_start) * 0.5
        r_fillet_end = r_hub + fillet_radius * 2
        
        # Distribution of radial stations
        radii_root = np.linspace(r_start, r_hub, 5)
        radii_fillet = np.linspace(r_hub, r_fillet_end, 8)
        radii_blade = np.linspace(r_fillet_end, r_tip, n_sections - 13)
        radii = np.concatenate([radii_root, radii_fillet[1:], radii_blade[1:]])
        
        vertices = []
        faces = []
        
        # Build blade surface by connecting sections
        for i, r in enumerate(radii):
            # Apply fillet scaling near root
            if r < r_fillet_end:
                # Smooth transition from hub to blade
                blend_factor = (r - r_start) / (r_fillet_end - r_start)
                blend_factor = 0.5 - 0.5 * np.cos(blend_factor * np.pi)  # Smooth S-curve
                
                # Scale chord for fillet
                scale_factor = 0.3 + 0.7 * blend_factor
            else:
                scale_factor = 1.0
            
            section = self.create_blade_section(r)
            
            # Apply fillet scaling to section
            if scale_factor < 1.0:
                section_center = np.mean(section, axis=0)
                section = section_center + (section - section_center) * scale_factor
            
            start_idx = len(vertices)
            vertices.extend(section)
            
            if i > 0:
                # Connect current section to previous section
                n_points = len(section)
                prev_start = start_idx - n_points
                
                for j in range(n_points):
                    # Next point index (wrapping around)
                    j_next = (j + 1) % n_points
                    
                    # Create two triangles for each quad
                    v1 = prev_start + j
                    v2 = start_idx + j
                    v3 = start_idx + j_next
                    v4 = prev_start + j_next
                    
                    faces.append([v1, v2, v3])
                    faces.append([v1, v3, v4])
        
        # Cap the blade tip with rounded end
        if len(vertices) > 0:
            last_section_start = len(vertices) - len(section)
            tip_vertices = vertices[last_section_start:]
            
            # Create rounded tip by adding intermediate ring
            tip_center = np.mean(tip_vertices, axis=0)
            
            # Add ring of vertices for rounded tip
            for v in tip_vertices:
                scaled_v = tip_center + (v - tip_center) * 0.5
                vertices.append(scaled_v)
            
            ring_start = len(vertices) - len(tip_vertices)
            
            # Connect outer ring to scaled ring
            n_points = len(section)
            for j in range(n_points):
                j_next = (j + 1) % n_points
                faces.append([last_section_start + j, last_section_start + j_next, ring_start + j_next])
                faces.append([last_section_start + j, ring_start + j_next, ring_start + j])
            
            # Add tip center and close
            vertices.append(tip_center)
            tip_idx = len(vertices) - 1
            
            for j in range(n_points):
                j_next = (j + 1) % n_points
                faces.append([ring_start + j, ring_start + j_next, tip_idx])
        
        # Create mesh
        vertices = np.array(vertices)
        faces = np.array(faces)
        
        blade_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        blade_mesh.fix_normals()
        
        return blade_mesh
    
    def create_hub(self) -> trimesh.Trimesh:
        """Create the hub (frustum) mesh with smooth edges"""
        height = self.params.hub_height
        r_bottom = self.params.hub_dia_base / 2
        r_top = self.params.hub_dia_top / 2
        
        # More segments for smoother hub
        n_segments = 48
        n_height = 10
        
        vertices = []
        faces = []
        
        # Generate vertices for each height level
        for i in range(n_height + 1):
            h = i * height / n_height
            # Interpolate radius
            r = r_bottom + (r_top - r_bottom) * i / n_height
            
            # Add slight bulge in middle for aesthetic
            if 0.3 < i/n_height < 0.7:
                bulge = 0.02 * self.params.hub_dia_base * np.sin((i/n_height - 0.3) * np.pi / 0.4)
                r += bulge
            
            theta = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
            for t in theta:
                vertices.append([r * np.cos(t), r * np.sin(t), h])
        
        # Create side faces
        for i in range(n_height):
            for j in range(n_segments):
                j_next = (j + 1) % n_segments
                
                v1 = i * n_segments + j
                v2 = i * n_segments + j_next
                v3 = (i + 1) * n_segments + j_next
                v4 = (i + 1) * n_segments + j
                
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])
        
        # Add bottom center and cap
        vertices.append([0, 0, 0])
        bottom_center_idx = len(vertices) - 1
        
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            faces.append([bottom_center_idx, j_next, j])
        
        # Add top center and cap
        vertices.append([0, 0, height])
        top_center_idx = len(vertices) - 1
        
        top_ring_start = n_height * n_segments
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            faces.append([top_center_idx, top_ring_start + j, top_ring_start + j_next])
        
        hub_mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
        hub_mesh.fix_normals()
        
        return hub_mesh
    
    def create_propeller(self) -> trimesh.Trimesh:
        """Create complete propeller mesh with properly intersecting blades"""
        # Create hub
        hub = self.create_hub()
        
        # Create and position blades around the hub
        meshes = [hub]
        blade_angle = 2 * np.pi / self.params.n_blades
        
        for i in range(self.params.n_blades):
            # Create blade with fillet
            blade = self.create_blade_with_fillet()
            
            # Rotate blade to correct position around hub
            rotation_matrix = trimesh.transformations.rotation_matrix(
                i * blade_angle, [0, 0, 1], [0, 0, 0]
            )
            blade.apply_transform(rotation_matrix)
            
            meshes.append(blade)
        
        # Combine all meshes
        propeller = trimesh.util.concatenate(meshes)
        
        # Clean up and ensure proper mesh
        propeller.process(validate=True)
        propeller.remove_duplicate_faces()
        propeller.remove_degenerate_faces()
        propeller.fix_normals()
        propeller.fill_holes()
        
        # Attempt to make watertight
        propeller.process(validate=True)
        
        return propeller

class BEMTSolver:
    """Blade Element Momentum Theory solver for propeller performance"""
    
    def __init__(self, params: PropellerParams, operating_conditions: dict):
        self.params = params
        self.water_density = operating_conditions.get('rho', 1000.0)  # kg/m³
        self.kinematic_viscosity = operating_conditions.get('nu', 1.0e-6)  # m²/s
        self.vapor_pressure = operating_conditions.get('pv', 2340)  # Pa
        self.ambient_pressure = operating_conditions.get('p_inf', 101325)  # Pa
        self.depth = operating_conditions.get('depth', 5.0)  # m
        
    def get_blade_properties(self, r_norm):
        """Get interpolated blade properties at normalized radius"""
        # Interpolate from distributions
        chord = np.interp(r_norm, np.linspace(0, 1, len(self.params.chord_distribution)),
                         self.params.chord_distribution) * self.params.diameter/2000  # Convert to m
        thickness = np.interp(r_norm, np.linspace(0, 1, len(self.params.thickness_distribution)),
                            self.params.thickness_distribution)
        camber = np.interp(r_norm, np.linspace(0, 1, len(self.params.camber_distribution)),
                          self.params.camber_distribution)
        
        # Pitch angle
        r_actual = r_norm * self.params.diameter/2000  # m
        if r_actual > 0:
            pitch_angle = np.arctan(self.params.pitch_ratio * self.params.diameter/1000 / (2 * np.pi * r_actual))
        else:
            pitch_angle = 0
            
        return chord, thickness, camber, pitch_angle
    
    def lift_drag_coefficients(self, alpha, Re):
        """Calculate lift and drag coefficients using thin airfoil theory + viscous corrections"""
        # Thin airfoil theory for lift
        Cl = 2 * np.pi * np.sin(alpha)
        
        # Drag coefficient (laminar + pressure drag)
        Cd0 = 0.008 + 1.0/Re**0.5 * 0.1  # Simplified drag model
        Cdi = Cl**2 / (np.pi * 5)  # Induced drag (AR=5 assumed)
        Cd = Cd0 + Cdi
        
        # Stall correction
        alpha_stall = 15 * np.pi/180
        if abs(alpha) > alpha_stall:
            stall_factor = np.exp(-2*(abs(alpha) - alpha_stall))
            Cl *= stall_factor
            Cd *= (2 - stall_factor)
        
        return Cl, Cd
    
    def calculate_cavitation_number(self, r, V_ship, rpm):
        """Calculate local cavitation number"""
        # Local velocity
        omega = rpm * 2 * np.pi / 60  # rad/s
        V_tangential = omega * r
        V_total = np.sqrt(V_ship**2 + V_tangential**2)
        
        # Cavitation number σ
        p_static = self.ambient_pressure + self.water_density * 9.81 * self.depth
        sigma = (p_static - self.vapor_pressure) / (0.5 * self.water_density * V_total**2)
        
        return sigma
    
    def burrill_cavitation_criterion(self, r_norm, thrust_loading, sigma):
        """Burrill diagram cavitation criterion"""
        # Projected area ratio
        chord, _, _, _ = self.get_blade_properties(r_norm)
        # Avoid division by zero
        if r_norm > 0.01:
            projected_area_ratio = self.params.n_blades * chord / (2 * np.pi * r_norm * self.params.diameter/2000)
        else:
            projected_area_ratio = 0.1
        
        # Burrill limit (simplified)
        tau_crit = (sigma + 0.2) * projected_area_ratio
        
        # Margin to cavitation (positive = no cavitation)
        margin = tau_crit - thrust_loading
        
        return margin
    
    def keller_criterion(self):
        """Keller's minimum blade area criterion for cavitation avoidance"""
        # Keller's formula for minimum blade area ratio
        Z = self.params.n_blades
        depth_factor = (self.depth + 10.33) / 10.33  # Depth in meters, 10.33m = 1 atm
        
        # Minimum expanded area ratio
        min_EAR = (1.3 + 0.3 * Z) / depth_factor + 0.2
        
        # Calculate actual EAR
        actual_EAR = self.calculate_expanded_area_ratio()
        
        # Return margin (positive = good)
        return actual_EAR - min_EAR
    
    def calculate_expanded_area_ratio(self):
        """Calculate the expanded area ratio of the propeller"""
        # Integrate blade area
        r_values = np.linspace(0.2, 1.0, 50)
        blade_areas = []
        
        for r_norm in r_values:
            chord, _, _, _ = self.get_blade_properties(r_norm)
            r_actual = r_norm * self.params.diameter/2000
            blade_areas.append(chord * r_actual)
        
        # Total expanded area
        total_blade_area = self.params.n_blades * 2 * simpson(blade_areas, x=r_values * self.params.diameter/2000)
        disc_area = np.pi * (self.params.diameter/2000)**2
        
        return total_blade_area / disc_area
    
    def solve(self, V_ship: float, rpm: float) -> dict:
        """Solve for propeller performance using BEMT"""
        n = rpm / 60  # rps
        omega = 2 * np.pi * n  # rad/s
        J = V_ship / (n * self.params.diameter/1000) if n > 0 else 0  # Advance ratio
        
        # Discretize blade into elements
        n_elements = 30
        r_hub = self.params.hub_dia_top/2000  # m
        r_tip = self.params.diameter/2000  # m
        r_values = np.linspace(r_hub, r_tip, n_elements)
        
        # Initialize totals
        total_thrust = 0
        total_torque = 0
        cavitation_indices = []
        noise_levels = []
        
        # Iterate over blade elements
        for i, r in enumerate(r_values):
            r_norm = (r - r_hub) / (r_tip - r_hub)
            
            # Get blade properties
            chord, thickness, camber, pitch_angle = self.get_blade_properties(r_norm)
            
            # Blade element velocities
            V_axial = V_ship
            V_tangential = omega * r
            V_rel = np.sqrt(V_axial**2 + V_tangential**2)
            
            # Flow angles
            phi = np.arctan2(V_axial, V_tangential) if V_tangential > 0 else 0
            alpha = pitch_angle - phi  # Angle of attack
            
            # Reynolds number
            Re = V_rel * chord / self.kinematic_viscosity
            Re = max(Re, 1e4)  # Minimum Re for stability
            
            # Lift and drag coefficients
            Cl, Cd = self.lift_drag_coefficients(alpha, Re)
            
            # Forces per unit span
            dr = r_values[1] - r_values[0] if i < n_elements-1 else r_values[-1] - r_values[-2]
            dL = 0.5 * self.water_density * V_rel**2 * chord * Cl * dr
            dD = 0.5 * self.water_density * V_rel**2 * chord * Cd * dr
            
            # Thrust and torque contributions
            dT = dL * np.cos(phi) - dD * np.sin(phi)
            dQ = (dL * np.sin(phi) + dD * np.cos(phi)) * r
            
            # Account for all blades
            total_thrust += self.params.n_blades * dT
            total_torque += self.params.n_blades * dQ
            
            # Cavitation analysis
            sigma = self.calculate_cavitation_number(r, V_ship, rpm)
            thrust_loading = dT / (0.5 * self.water_density * V_rel**2 * chord * dr) if V_rel > 0 else 0
            cav_margin = self.burrill_cavitation_criterion(r_norm, thrust_loading, sigma)
            
            # Store cavitation index (lower sigma = more cavitation)
            cavitation_indices.append(1.0 / (sigma + 0.1))
            
            # Noise estimation (simplified - based on tip speed and loading)
            tip_mach = V_tangential / 1500  # Speed of sound in water ~1500 m/s
            loading_noise = 10 * np.log10(abs(thrust_loading) + 1)
            thickness_noise = 20 * np.log10(tip_mach + 0.1) * thickness
            noise_levels.append(loading_noise + thickness_noise)
        
        # Calculate efficiency
        if total_torque > 0 and omega > 0:
            efficiency = (total_thrust * V_ship) / (omega * total_torque)
        else:
            efficiency = 0
        
        # Cavitation metrics
        max_cavitation_index = max(cavitation_indices) if cavitation_indices else 1.0
        keller_margin = self.keller_criterion()
        
        # Noise metrics
        overall_noise = 10 * np.log10(sum(10**(nl/10) for nl in noise_levels)) if noise_levels else 0
        
        # Tip vortex cavitation check
        tip_speed = omega * r_tip
        tip_cavitation_number = self.calculate_cavitation_number(r_tip, V_ship, rpm)
        
        return {
            'thrust': total_thrust,  # N
            'torque': total_torque,  # Nm
            'efficiency': efficiency,
            'cavitation_index': max_cavitation_index,
            'keller_margin': keller_margin,
            'noise_level': overall_noise,  # dB
            'tip_cavitation_number': tip_cavitation_number,
            'KT': total_thrust / (self.water_density * n**2 * (self.params.diameter/1000)**4) if n > 0 else 0,
            'KQ': total_torque / (self.water_density * n**2 * (self.params.diameter/1000)**5) if n > 0 else 0,
            'J': J
        }

class PropellerOptimizationProblem(Problem):
    """Multi-objective optimization problem for propeller design"""
    
    def __init__(self):
        # Design variables: chord, thickness, camber, pitch, skew, rake distributions
        n_vars = 34  # 10 points each for chord, thickness, camber + pitch + skew + rake + n_blades
        
        # ENHANCED: Wider bounds for more visual variation
        xl = np.array(
            [0.10]*10 +  # Chord: wider range (was 0.18)
            [0.02]*10 +  # Thickness: slightly wider (was 0.03)
            [0.0]*10 +   # Camber: same
            [0.5,        # Pitch ratio: much wider (was 0.8)
             -30,        # Skew: allow negative skew (was 0)
             -15,        # Rake: allow negative rake (was 0)
             2]          # n_blades: start from 2 (was 3)
        )
        xu = np.array(
            [0.45]*10 +  # Chord: wider range (was 0.35)
            [0.15]*10 +  # Thickness: wider (was 0.10)
            [0.08]*10 +  # Camber: much wider (was 0.03)
            [2.0,        # Pitch ratio: much wider (was 1.4)
             45,         # Skew: much wider (was 20)
             25,         # Rake: much wider (was 8)
             7]          # n_blades: up to 7 (was 5)
        )
        
        super().__init__(n_var=n_vars, n_obj=3, n_constr=2, xl=xl, xu=xu)
        
        self.eval_count = 0
        
        # Operating conditions - adjusted for 180mm propeller
        self.operating_conditions = {
            'V_ship': 3.0,  # m/s
            'rpm': 1500,  # RPM
            'rho': 1025.0,  # kg/m³ (seawater)
            'nu': 1.35e-6,  # m²/s
            'depth': 5.0,  # m
            'p_inf': 101325,  # Pa
            'pv': 2340  # Pa
        }
    
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate objective functions and constraints"""
        n_pop = x.shape[0]
        
        f1 = np.zeros(n_pop)  # Cavitation index (minimize)
        f2 = np.zeros(n_pop)  # Negative efficiency (minimize)
        f3 = np.zeros(n_pop)  # Noise level (minimize)
        
        g1 = np.zeros(n_pop)  # Thrust constraint
        g2 = np.zeros(n_pop)  # Structural constraint
        
        for i in range(n_pop):
            try:
                # Extract design variables
                params = PropellerParams()
                params.chord_distribution = x[i, 0:10]
                params.thickness_distribution = x[i, 10:20]
                params.camber_distribution = x[i, 20:30]
                params.pitch_ratio = x[i, 30]
                params.skew = x[i, 31]
                params.rake = x[i, 32]
                params.n_blades = int(x[i, 33])
                
                # Run BEMT analysis
                solver = BEMTSolver(params, self.operating_conditions)
                results = solver.solve(
                    V_ship=self.operating_conditions['V_ship'],
                    rpm=self.operating_conditions['rpm']
                )
                
                # Calculate objectives
                f1[i] = results['cavitation_index']  # Minimize cavitation
                f2[i] = -results['efficiency'] if results['efficiency'] > 0 else 1.0  # Maximize efficiency
                f3[i] = results['noise_level']  # Minimize noise
                
                # Constraints (relaxed for feasibility)
                g1[i] = 15 - results['thrust']  # Minimum thrust of 15N (reduced from 20)
                g2[i] = 0.02 - np.min(params.thickness_distribution)  # Minimum thickness (reduced from 0.025)
                
            except Exception as e:
                # Handle any numerical errors
                f1[i] = 10.0  # High penalty values
                f2[i] = 1.0
                f3[i] = 100.0
                g1[i] = 100.0
                g2[i] = 100.0
            
            self.eval_count += 1
            
            # Print progress with more detail
            if self.eval_count % 50 == 0:
                print(f"Eval {self.eval_count}: Cav={f1[i]:.3f}, Eff={-f2[i]:.3f}, Noise={f3[i]:.1f}dB")
                print(f"  Params: Blades={int(x[i, 33])}, Pitch={x[i, 30]:.2f}, Skew={x[i, 31]:.1f}°, Rake={x[i, 32]:.1f}°")
        
        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack([g1, g2])

def optimize_propeller(n_generations: int = 50, population_size: int = 100):
    """Run NSGA-II optimization"""
    
    problem = PropellerOptimizationProblem()
    
    # ENHANCED: More aggressive mutation for diversity
    algorithm = NSGA2(
        pop_size=population_size,
        n_offsprings=30,  # Increased from 20
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.95, eta=10),  # More aggressive crossover
        mutation=PM(eta=15, prob=0.3),  # Higher mutation probability
        eliminate_duplicates=True
    )
    
    termination = get_termination("n_gen", n_generations)
    
    print(f"Starting optimization with {population_size} population size for {n_generations} generations")
    print("Objectives: Minimize [Cavitation, -Efficiency, Noise]")
    print("Enhanced parameter ranges for greater geometric variation")
    print("-" * 60)
    
    try:
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=None,  # Random seed for more variation
                       save_history=True,
                       verbose=True)
        return res
    except Exception as e:
        print(f"Optimization error: {e}")
        return None

def visualize_propeller_comparison(params_list: List[PropellerParams], save_path: str = None):
    """Visualize multiple propeller designs for comparison"""
    n_designs = min(len(params_list), 4)  # Show up to 4 designs
    
    fig = plt.figure(figsize=(20, 12))
    
    for idx, params in enumerate(params_list[:n_designs]):
        geom = PropellerGeometry(params)
        prop_mesh = geom.create_propeller()
        
        # 3D view
        ax = fig.add_subplot(2, n_designs, idx + 1, projection='3d')
        
        vertices = prop_mesh.vertices
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                       triangles=prop_mesh.faces,
                       cmap='viridis', alpha=0.8, edgecolor='none')
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'Design {idx+1}: {params.n_blades} blades')
        
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
        ax.view_init(elev=20, azim=45)
        
        # Top view
        ax2 = fig.add_subplot(2, n_designs, n_designs + idx + 1)
        ax2.triplot(vertices[:, 0], vertices[:, 1], prop_mesh.faces, 'k-', alpha=0.3, linewidth=0.5)
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_title(f'Top View - P/D:{params.pitch_ratio:.2f}, Skew:{params.skew:.0f}°')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Propeller Design Variations', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    try:
        plt.show()
    except:
        plt.close()

def visualize_propeller(params: PropellerParams, save_path: str = None):
    """Visualize the propeller geometry with performance metrics"""
    geom = PropellerGeometry(params)
    prop_mesh = geom.create_propeller()
    
    # Calculate performance for display
    solver = BEMTSolver(params, {
        'V_ship': 3.0,
        'rpm': 1500,
        'rho': 1025.0,
        'nu': 1.35e-6,
        'depth': 5.0,
        'p_inf': 101325,
        'pv': 2340
    })
    perf = solver.solve(V_ship=3.0, rpm=1500)
    
    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    
    # 3D view
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Plot mesh
    vertices = prop_mesh.vertices
    ax1.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    triangles=prop_mesh.faces,
                    cmap='viridis', alpha=0.8, edgecolor='none')
    
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'Optimized Marine Propeller - {params.n_blades} Blades')
    
    # Equal aspect ratio
    max_range = np.array([vertices[:, 0].max()-vertices[:, 0].min(),
                          vertices[:, 1].max()-vertices[:, 1].min(),
                          vertices[:, 2].max()-vertices[:, 2].min()]).max() / 2.0
    
    mid_x = (vertices[:, 0].max()+vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max()+vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max()+vertices[:, 2].min()) * 0.5
    
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    ax1.view_init(elev=20, azim=45)
    
    # Top view
    ax2 = fig.add_subplot(222)
    ax2.triplot(vertices[:, 0], vertices[:, 1], prop_mesh.faces, 'k-', alpha=0.3, linewidth=0.5)
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('Top View')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Performance metrics display
    ax3 = fig.add_subplot(223)
    ax3.axis('off')
    metrics_text = f"""
    Performance Metrics @ 1500 RPM, 3 m/s:
    
    Thrust: {perf['thrust']:.1f} N
    Torque: {perf['torque']:.2f} Nm
    Efficiency: {perf['efficiency']:.3f} ({perf['efficiency']*100:.1f}%)
    
    Cavitation Metrics:
    Cavitation Index: {perf['cavitation_index']:.3f}
    Tip Cavitation Number: {perf['tip_cavitation_number']:.3f}
    Keller Margin: {perf['keller_margin']:.3f}
    Noise Level: {perf['noise_level']:.1f} dB
    
    Geometry:
    Number of Blades: {params.n_blades}
    Diameter: {params.diameter:.1f} mm
    Pitch Ratio: {params.pitch_ratio:.3f}
    Skew: {params.skew:.1f}°
    Rake: {params.rake:.1f}°
    
    Non-dimensional Coefficients:
    KT: {perf['KT']:.4f}
    KQ: {perf['KQ']:.4f}
    J: {perf['J']:.3f}
    """
    ax3.text(0.1, 0.9, metrics_text, transform=ax3.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    # Blade distribution plots
    ax4 = fig.add_subplot(224)
    r_norm = np.linspace(0, 1, len(params.chord_distribution))
    ax4.plot(r_norm, params.chord_distribution, 'b-', label='Chord', linewidth=2)
    ax4.plot(r_norm, params.thickness_distribution, 'r-', label='Thickness', linewidth=2)
    ax4.plot(r_norm, params.camber_distribution * 10, 'g-', label='Camber (×10)', linewidth=2)  # Scale for visibility
    ax4.set_xlabel('Normalized Radius (r/R)')
    ax4.set_ylabel('Normalized Value')
    ax4.set_title('Blade Distributions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    try:
        plt.show()
    except:
        plt.close()
    
    return prop_mesh

def save_optimized_design(params: PropellerParams, filepath: str, performance: dict = None):
    """Save the optimized propeller design"""
    geom = PropellerGeometry(params)
    prop_mesh = geom.create_propeller()
    
    # Export based on file extension
    if filepath.endswith('.stl'):
        prop_mesh.export(filepath, file_type='stl')
    elif filepath.endswith('.3mf'):
        prop_mesh.export(filepath, file_type='3mf')
    else:
        raise ValueError("Unsupported file format. Use .stl or .3mf")
    
    print(f"Propeller design saved to: {filepath}")
    
    # Save parameters and performance as JSON for reference
    params_dict = {
        'geometry': {
            'n_blades': params.n_blades,
            'diameter': params.diameter,
            'hub_dia_base': params.hub_dia_base,
            'hub_dia_top': params.hub_dia_top,
            'hub_height': params.hub_height,
            'pitch_ratio': params.pitch_ratio,
            'rake': params.rake,
            'skew': params.skew,
            'chord_distribution': params.chord_distribution.tolist() if params.chord_distribution is not None else None,
            'thickness_distribution': params.thickness_distribution.tolist() if params.thickness_distribution is not None else None,
            'camber_distribution': params.camber_distribution.tolist() if params.camber_distribution is not None else None
        }
    }
    
    if performance:
        params_dict['performance'] = performance
    
    json_path = filepath.replace('.stl', '_params.json').replace('.3mf', '_params.json')
    with open(json_path, 'w') as f:
        json.dump(params_dict, f, indent=2)
    print(f"Parameters saved to: {json_path}")

def main():
    """Main execution function"""
    
    # Define output directory
    output_dir = "/HPC/matthew.barry/propeller_output"
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    except Exception as e:
        print(f"Warning: Could not create output directory: {e}")
        print(f"Using current directory instead")
        output_dir = "."
    
    print("\n" + "="*60)
    print("Marine Propeller Optimization System - ENHANCED VERSION")
    print("Built-in CFD using Blade Element Momentum Theory")
    print("="*60)
    print("\nEnhanced Features:")
    print("✓ Wider parameter ranges for greater variation")
    print("✓ Stronger geometric transformations")
    print("✓ More aggressive optimization strategy")
    print("✓ Support for 2-7 blades")
    print("✓ Negative skew and rake allowed")
    print("\nOptimization Objectives:")
    print("1. Minimize cavitation (noise reduction)")
    print("2. Maximize efficiency")
    print("3. Minimize acoustic noise")
    print("\n" + "-"*60 + "\n")
    
    # Run optimization
    print("Starting NSGA-II optimization with enhanced variation...\n")
    
    results = optimize_propeller(n_generations=30, population_size=80)
    
    print("\n" + "-"*60)
    print("Optimization Complete!")
    print("-"*60 + "\n")
    
    # Process results and extract multiple diverse solutions
    diverse_params_list = []
    
    if results is not None and hasattr(results, 'X') and results.X is not None and len(results.X) > 0:
        # Get top solutions with diversity
        n_solutions = min(4, len(results.X))
        
        # Sort by different objectives to get diverse solutions
        if hasattr(results, 'F') and results.F is not None:
            # Best for cavitation
            cav_idx = np.argmin(results.F[:, 0])
            # Best for efficiency
            eff_idx = np.argmin(results.F[:, 1])
            # Best for noise
            noise_idx = np.argmin(results.F[:, 2])
            # Best compromise
            F_norm = results.F.copy()
            for i in range(3):
                if F_norm[:, i].max() > F_norm[:, i].min():
                    F_norm[:, i] = (F_norm[:, i] - F_norm[:, i].min()) / (F_norm[:, i].max() - F_norm[:, i].min())
            scores = np.sum(F_norm * [0.4, 0.4, 0.2], axis=1)
            comp_idx = np.argmin(scores)
            
            indices = [comp_idx, cav_idx, eff_idx, noise_idx]
            indices = list(set(indices))[:n_solutions]  # Remove duplicates
            
            for idx in indices:
                params = PropellerParams()
                params.chord_distribution = results.X[idx, 0:10]
                params.thickness_distribution = results.X[idx, 10:20]
                params.camber_distribution = results.X[idx, 20:30]
                params.pitch_ratio = results.X[idx, 30]
                params.skew = results.X[idx, 31]
                params.rake = results.X[idx, 32]
                params.n_blades = int(results.X[idx, 33])
                diverse_params_list.append(params)
    
    # If no optimization results, create diverse default designs
    if not diverse_params_list:
        print("Creating diverse default designs...")
        for i in range(4):
            params = PropellerParams()
            r = np.linspace(0.2, 1.0, 10)
            params.chord_distribution = 0.2 + 0.1 * np.random.rand() - 0.1 * r + 0.05 * r**2
            params.thickness_distribution = 0.04 + 0.02 * np.random.rand() - 0.03 * r
            params.camber_distribution = 0.01 + 0.01 * np.random.rand() * np.ones_like(r)
            params.n_blades = np.random.choice([3, 4, 5])
            params.pitch_ratio = 0.8 + 0.6 * np.random.rand()
            params.skew = -10 + 30 * np.random.rand()
            params.rake = -5 + 15 * np.random.rand()
            diverse_params_list.append(params)
    
    # Use the first (best compromise) solution as the main result
    optimized_params = diverse_params_list[0]
    
    # Calculate final performance
    solver = BEMTSolver(optimized_params, {
        'V_ship': 3.0,
        'rpm': 1500,
        'rho': 1025.0,
        'nu': 1.35e-6,
        'depth': 5.0,
        'p_inf': 101325,
        'pv': 2340
    })
    
    try:
        final_performance = solver.solve(V_ship=3.0, rpm=1500)
        
        print("\nBest Solution Found:")
        print(f"  Number of blades: {optimized_params.n_blades}")
        print(f"  Pitch ratio: {optimized_params.pitch_ratio:.3f}")
        print(f"  Skew angle: {optimized_params.skew:.1f}°")
        print(f"  Rake angle: {optimized_params.rake:.1f}°")
        print(f"\nPerformance Metrics (@ 1500 RPM, 3 m/s):")
        print(f"  Thrust: {final_performance['thrust']:.1f} N")
        print(f"  Efficiency: {final_performance['efficiency']*100:.1f}%")
        print(f"  Cavitation Index: {final_performance['cavitation_index']:.3f}")
        print(f"  Noise Level: {final_performance['noise_level']:.1f} dB")
        
    except Exception as e:
        print(f"Error calculating performance: {e}")
        final_performance = None
    
    # Define file paths
    stl_path = os.path.join(output_dir, "optimized_propeller.stl")
    threedf_path = os.path.join(output_dir, "optimized_propeller.3mf")
    png_path = os.path.join(output_dir, "optimized_propeller.png")
    comparison_path = os.path.join(output_dir, "propeller_comparison.png")
    
    # Visualize the comparison
    print("\n" + "-"*60)
    print("Generating comparison visualization...")
    print("-"*60)
    
    try:
        visualize_propeller_comparison(diverse_params_list, save_path=comparison_path)
        print(f"Comparison saved to: {comparison_path}")
    except Exception as e:
        print(f"Warning: Could not generate comparison: {e}")
    
    # Visualize the main result
    print("\nGenerating main result visualization...")
    
    try:
        mesh = visualize_propeller(optimized_params, save_path=png_path)
        print(f"Visualization saved to: {png_path}")
    except Exception as e:
        print(f"Warning: Could not generate visualization: {e}")
    
    # Save the designs
    print("\n" + "-"*60)
    print("Saving optimized design files...")
    print("-"*60)
    
    try:
        # Save main design
        save_optimized_design(optimized_params, stl_path, final_performance)
        save_optimized_design(optimized_params, threedf_path, final_performance)
        
        # Save additional diverse designs
        for i, params in enumerate(diverse_params_list[1:], 1):
            alt_stl = os.path.join(output_dir, f"alternative_design_{i}.stl")
            save_optimized_design(params, alt_stl)
        
        print("\n" + "="*60)
        print("Optimization Complete! Files saved:")
        print("="*60)
        print(f"✓ {stl_path}")
        print(f"✓ {threedf_path}")
        print(f"✓ {comparison_path}")
        print(f"✓ Alternative designs saved")
        print(f"\nAll files saved in: {output_dir}")
        
    except Exception as e:
        print(f"Error saving files: {e}")

if __name__ == "__main__":
    main()