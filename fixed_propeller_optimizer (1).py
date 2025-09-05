#!/usr/bin/env python3
"""
Marine Propeller Geometry Optimization using NSGA-II with Built-in CFD
MARINE-SPECIFIC VERSION: Proper boat/ship propeller geometry with wide blades and marine characteristics
Based on Wageningen B-series and modern marine propeller design principles
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
    """Parameters defining MARINE propeller geometry"""
    n_blades: int = 4
    diameter: float = 180.0  # mm
    hub_dia_base: float = 35.0  # mm (typical marine hub ratio ~0.18)
    hub_dia_top: float = 40.0  # mm
    hub_height: float = 30.0  # mm
    pitch_ratio: float = 1.0  # pitch/diameter ratio
    rake: float = 5.0  # degrees (aft rake typical for marine)
    skew: float = 20.0  # degrees (moderate skew for marine)
    expanded_area_ratio: float = 0.65  # EAR - typical marine value 0.4-1.0
    cup: float = 0.02  # Cup ratio (trailing edge cup)
    chord_distribution: np.ndarray = None
    thickness_distribution: np.ndarray = None
    camber_distribution: np.ndarray = None
    
    def __post_init__(self):
        if self.chord_distribution is None:
            # MARINE chord distribution - much wider blades
            # Based on typical marine propeller geometry
            r = np.linspace(0.2, 1.0, 10)
            # Wide at root, gradual taper to tip - marine style
            self.chord_distribution = 0.45 * (1.0 - 0.3 * r - 0.15 * r**2)
            
        if self.thickness_distribution is None:
            # MARINE thickness distribution - much thicker
            r = np.linspace(0.2, 1.0, 10)
            # Very thick at root for strength, still substantial at tip
            self.thickness_distribution = 0.12 * (1.0 - 0.5 * r)
            
        if self.camber_distribution is None:
            # MARINE camber distribution
            r = np.linspace(0.2, 1.0, 10)
            # More camber for thrust generation in water
            self.camber_distribution = 0.04 * (1.0 - 0.3 * r)

class MarinePropellerGeometry:
    """Generate 3D MARINE propeller geometry"""
    
    def __init__(self, params: PropellerParams):
        self.params = params
    
    def marine_blade_section(self, x, thickness, camber, r_norm):
        """Generate marine propeller blade section (modified for water operation)
        Uses ogival/elliptical sections near root, transitioning to modified NACA at tip
        """
        # Marine propellers use different sections at different radii
        if r_norm < 0.3:
            # Near hub - use thick elliptical/ogival section
            # These are structurally strong and cavitation-resistant
            t = thickness * 1.5  # Extra thickness near root
            
            # Elliptical thickness distribution
            yt = t * np.sqrt(1 - (2*x - 1)**2) * 2.5
            
            # Simple camber
            yc = camber * (1 - x) * x * 4
            
        elif r_norm < 0.7:
            # Mid-blade - modified thick NACA-style section
            # Thickness distribution for marine use
            yt = 5 * thickness * 1.2 * (
                0.2969 * np.sqrt(x) 
                - 0.1260 * x 
                - 0.2516 * x**2  # Less reduction for thicker profile
                + 0.2843 * x**3 
                - 0.1515 * x**4
            )
            
            # Camber for lift generation
            yc = camber * (2 * x - x**2)
            
        else:
            # Tip region - thinner but still substantial
            # Modified NACA for tip
            yt = 5 * thickness * (
                0.2969 * np.sqrt(x) 
                - 0.1260 * x 
                - 0.3516 * x**2 
                + 0.2843 * x**3 
                - 0.1015 * x**4
            )
            
            # Reduced camber at tip
            yc = camber * 0.7 * (2 * x - x**2)
        
        # Add cup effect (trailing edge bend) - characteristic of marine propellers
        if self.params.cup > 0 and x > 0.7:
            cup_effect = self.params.cup * (x - 0.7) / 0.3
            yc += cup_effect * thickness
        
        # Calculate upper and lower surfaces
        xu = x
        yu = yc + yt
        xl = x
        yl = yc - yt
        
        return xu, yu, xl, yl
        
    def create_blade_section(self, r: float, theta: float = 0) -> np.ndarray:
        """Create a MARINE blade cross-section at radius r"""
        # Normalize radius
        r_norm = (r - self.params.hub_dia_top/2) / (self.params.diameter/2 - self.params.hub_dia_top/2)
        r_norm = np.clip(r_norm, 0, 1)
        
        # Get section properties - MARINE SPECIFIC
        # Use expanded area ratio to determine chord
        base_chord = np.interp(r_norm, np.linspace(0, 1, len(self.params.chord_distribution)),
                              self.params.chord_distribution) * self.params.diameter/2
        
        # Scale chord by expanded area ratio for proper blade width
        chord = base_chord * (self.params.expanded_area_ratio / 0.5) ** 0.5
        
        # Marine thickness - much thicker than aircraft
        thickness_ratio = np.interp(r_norm, np.linspace(0, 1, len(self.params.thickness_distribution)),
                                   self.params.thickness_distribution)
        
        # Extra thickness at root for marine strength requirements
        if r_norm < 0.3:
            thickness_ratio *= (1 + 1.0 * (0.3 - r_norm) / 0.3)  # Double thickness at hub
        
        # Marine camber
        camber_ratio = np.interp(r_norm, np.linspace(0, 1, len(self.params.camber_distribution)),
                                self.params.camber_distribution)
        
        # Generate marine blade section
        n_points = 40  # More points for smoother marine blade
        x_normalized = np.linspace(0, 1, n_points//2)
        
        # Get marine section coordinates
        xu, yu, xl, yl = self.marine_blade_section(x_normalized, thickness_ratio, camber_ratio, r_norm)
        
        # Scale by chord length
        xu *= chord
        yu *= chord
        xl *= chord
        yl *= chord
        
        # Combine upper and lower surfaces
        airfoil_x = np.concatenate([xu, xl[::-1][1:]])
        airfoil_y = np.concatenate([yu, yl[::-1][1:]])
        
        # MARINE PITCH DISTRIBUTION
        # Marine propellers typically have more constant pitch
        pitch_angle = np.arctan(self.params.pitch_ratio * self.params.diameter / (2 * np.pi * r)) if r > 0 else 0
        
        # Less aggressive twist for marine propellers
        twist_factor = 1.0 - 0.2 * r_norm  # Reduced twist variation
        pitch_angle *= twist_factor
        
        # MARINE SKEW - progressive skew is common
        # Skew increases towards tip to reduce vibration
        skew_angle = self.params.skew * (r_norm ** 1.5) * np.pi / 180
        
        # Create 3D points for this section
        points = []
        for i in range(len(airfoil_x)):
            # Position blade section
            x_blade = airfoil_x[i] - chord * 0.3  # Offset for proper alignment
            y_blade = airfoil_y[i]
            
            # Apply pitch rotation
            x_rot = x_blade * np.cos(pitch_angle) - y_blade * np.sin(pitch_angle)
            z_rot = x_blade * np.sin(pitch_angle) + y_blade * np.cos(pitch_angle)
            
            # Apply skew
            angle = theta + skew_angle
            
            # Position in 3D space
            x = r * np.cos(angle) - x_rot * np.sin(angle)
            y = r * np.sin(angle) + x_rot * np.cos(angle)
            z = z_rot + self.params.hub_height/2
            
            # MARINE RAKE - typically aft rake
            if self.params.rake != 0:
                rake_angle = self.params.rake * np.pi / 180
                # Aft rake (blade tilts backward)
                z += r_norm * np.tan(rake_angle) * self.params.diameter/8
                # Slight radial displacement
                displacement = r_norm * np.sin(rake_angle) * self.params.diameter/25
                x += displacement * np.cos(angle)
                y += displacement * np.sin(angle)
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def create_marine_blade(self) -> trimesh.Trimesh:
        """Create a single MARINE blade mesh with proper root fillet"""
        # Marine propellers need more sections for smooth curvature
        n_sections = 40
        
        # Start from hub with proper fillet
        r_start = self.params.hub_dia_base/2 * 0.85
        r_hub = self.params.hub_dia_top/2
        r_tip = self.params.diameter/2
        
        # Marine fillet is larger and smoother
        fillet_radius = (r_hub - r_start) * 0.8
        r_fillet_end = r_hub + fillet_radius * 3
        
        # Distribution of radial stations - more density near hub for marine blades
        radii_root = np.linspace(r_start, r_hub, 6)
        radii_fillet = np.linspace(r_hub, r_fillet_end, 10)
        radii_blade = np.linspace(r_fillet_end, r_tip * 0.98, n_sections - 16)  # Don't go all the way to tip
        radii = np.concatenate([radii_root, radii_fillet[1:], radii_blade[1:]])
        
        vertices = []
        faces = []
        
        # Build blade surface
        for i, r in enumerate(radii):
            # Marine fillet - broader and smoother transition
            if r < r_fillet_end:
                blend_factor = (r - r_start) / (r_fillet_end - r_start)
                # Smoother S-curve for marine applications
                blend_factor = 0.5 - 0.5 * np.cos(blend_factor * np.pi)
                # Marine blades need stronger root connection
                scale_factor = 0.2 + 0.8 * blend_factor
            else:
                scale_factor = 1.0
            
            section = self.create_blade_section(r)
            
            # Apply fillet scaling
            if scale_factor < 1.0:
                section_center = np.mean(section, axis=0)
                # Scale with elliptical profile for marine strength
                scale_x = scale_factor
                scale_y = scale_factor ** 0.7  # Less reduction in thickness
                for j in range(len(section)):
                    diff = section[j] - section_center
                    section[j] = section_center + [diff[0] * scale_x, diff[1] * scale_x, diff[2] * scale_y]
            
            start_idx = len(vertices)
            vertices.extend(section)
            
            if i > 0:
                # Connect sections
                n_points = len(section)
                prev_start = start_idx - n_points
                
                for j in range(n_points):
                    j_next = (j + 1) % n_points
                    
                    v1 = prev_start + j
                    v2 = start_idx + j
                    v3 = start_idx + j_next
                    v4 = prev_start + j_next
                    
                    faces.append([v1, v2, v3])
                    faces.append([v1, v3, v4])
        
        # Marine blade tip - rounded and thick
        if len(vertices) > 0:
            last_section_start = len(vertices) - len(section)
            tip_vertices = vertices[last_section_start:]
            
            # Marine tips are often more rounded/elliptical
            tip_center = np.mean(tip_vertices, axis=0)
            
            # Create rounded tip with multiple rings for smooth closure
            for scale in [0.7, 0.4]:
                ring_start = len(vertices)
                for v in tip_vertices:
                    scaled_v = tip_center + (v - tip_center) * scale
                    vertices.append(scaled_v)
                
                # Connect rings
                n_points = len(section)
                prev_ring = ring_start - n_points if scale == 0.7 else ring_start - n_points
                for j in range(n_points):
                    j_next = (j + 1) % n_points
                    faces.append([prev_ring + j, prev_ring + j_next, ring_start + j_next])
                    faces.append([prev_ring + j, ring_start + j_next, ring_start + j])
            
            # Close tip
            vertices.append(tip_center)
            tip_idx = len(vertices) - 1
            last_ring = len(vertices) - len(section) - 1
            
            for j in range(n_points):
                j_next = (j + 1) % n_points
                faces.append([last_ring - n_points + j, last_ring - n_points + j_next, tip_idx])
        
        # Create mesh
        vertices = np.array(vertices)
        faces = np.array(faces)
        
        blade_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        blade_mesh.fix_normals()
        
        return blade_mesh
    
    def create_marine_hub(self) -> trimesh.Trimesh:
        """Create MARINE propeller hub with ogival/streamlined shape"""
        height = self.params.hub_height
        r_bottom = self.params.hub_dia_base / 2
        r_top = self.params.hub_dia_top / 2
        
        # Marine hubs are often more streamlined
        n_segments = 64  # Smoother for marine applications
        n_height = 15
        
        vertices = []
        faces = []
        
        # Generate vertices with marine hub profile
        for i in range(n_height + 1):
            h = i * height / n_height
            h_norm = i / n_height
            
            # Marine hub profile - more bulbous/ogival shape
            # Parabolic/ogival profile common in marine propellers
            if h_norm < 0.5:
                # Forward section - expanding
                profile_factor = np.sin(h_norm * np.pi)
                r = r_bottom + (r_top - r_bottom) * profile_factor
                # Add bulge for hydrodynamic shape
                r += 0.03 * self.params.hub_dia_base * profile_factor
            else:
                # Aft section - contracting slightly
                profile_factor = 0.5 + 0.5 * np.cos((h_norm - 0.5) * 2 * np.pi)
                r = r_top - (r_top - r_bottom) * 0.1 * (1 - profile_factor)
            
            theta = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
            for t in theta:
                vertices.append([r * np.cos(t), r * np.sin(t), h])
        
        # Create faces
        for i in range(n_height):
            for j in range(n_segments):
                j_next = (j + 1) % n_segments
                
                v1 = i * n_segments + j
                v2 = i * n_segments + j_next
                v3 = (i + 1) * n_segments + j_next
                v4 = (i + 1) * n_segments + j
                
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])
        
        # Add rounded bottom cap (marine hubs often have rounded ends)
        vertices.append([0, 0, -height * 0.1])  # Slightly extended for streamlining
        bottom_center_idx = len(vertices) - 1
        
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            faces.append([bottom_center_idx, j_next, j])
        
        # Add rounded top cap
        vertices.append([0, 0, height * 1.1])  # Slightly extended
        top_center_idx = len(vertices) - 1
        
        top_ring_start = n_height * n_segments
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            faces.append([top_center_idx, top_ring_start + j, top_ring_start + j_next])
        
        hub_mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
        hub_mesh.fix_normals()
        
        return hub_mesh
    
    def create_propeller(self) -> trimesh.Trimesh:
        """Create complete MARINE propeller mesh"""
        # Create marine hub
        hub = self.create_marine_hub()
        
        # Create and position blades
        meshes = [hub]
        blade_angle = 2 * np.pi / self.params.n_blades
        
        # Add slight angular offset between blades for marine applications
        # This helps with vibration reduction
        angular_offset = 0.02 if self.params.n_blades > 3 else 0
        
        for i in range(self.params.n_blades):
            # Create marine blade
            blade = self.create_marine_blade()
            
            # Rotate blade to position with slight offset
            angle = i * blade_angle + i * angular_offset
            rotation_matrix = trimesh.transformations.rotation_matrix(
                angle, [0, 0, 1], [0, 0, 0]
            )
            blade.apply_transform(rotation_matrix)
            
            meshes.append(blade)
        
        # Combine all meshes
        propeller = trimesh.util.concatenate(meshes)
        
        # Clean up
        propeller.process(validate=True)
        propeller.remove_duplicate_faces()
        propeller.remove_degenerate_faces()
        propeller.fix_normals()
        propeller.fill_holes()
        
        return propeller

class BEMTSolver:
    """Blade Element Momentum Theory solver for MARINE propeller performance"""
    
    def __init__(self, params: PropellerParams, operating_conditions: dict):
        self.params = params
        self.water_density = operating_conditions.get('rho', 1025.0)  # kg/m³ seawater
        self.kinematic_viscosity = operating_conditions.get('nu', 1.35e-6)  # m²/s seawater
        self.vapor_pressure = operating_conditions.get('pv', 2340)  # Pa
        self.ambient_pressure = operating_conditions.get('p_inf', 101325)  # Pa
        self.depth = operating_conditions.get('depth', 5.0)  # m
        
    def get_blade_properties(self, r_norm):
        """Get interpolated blade properties at normalized radius"""
        # Use expanded area ratio to scale chord appropriately
        base_chord = np.interp(r_norm, np.linspace(0, 1, len(self.params.chord_distribution)),
                              self.params.chord_distribution) * self.params.diameter/2000
        chord = base_chord * (self.params.expanded_area_ratio / 0.5) ** 0.5
        
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
        """Calculate lift and drag coefficients for MARINE conditions"""
        # Marine propeller lift curve - different from aircraft
        # Account for 3D effects and water operation
        Cl = 2 * np.pi * np.sin(alpha) * 0.9  # Reduced for 3D effects in water
        
        # Marine drag - higher due to water viscosity
        Cd0 = 0.012 + 2.0/Re**0.5 * 0.1  # Higher base drag for marine
        Cdi = Cl**2 / (np.pi * 4)  # Lower aspect ratio for marine blades
        Cd = Cd0 + Cdi
        
        # Stall characteristics for marine propellers
        alpha_stall = 12 * np.pi/180  # Earlier stall in water
        if abs(alpha) > alpha_stall:
            stall_factor = np.exp(-3*(abs(alpha) - alpha_stall))
            Cl *= stall_factor
            Cd *= (3 - 2*stall_factor)  # Higher drag penalty
        
        return Cl, Cd
    
    def calculate_cavitation_number(self, r, V_ship, rpm):
        """Calculate local cavitation number for marine conditions"""
        omega = rpm * 2 * np.pi / 60  # rad/s
        V_tangential = omega * r
        V_total = np.sqrt(V_ship**2 + V_tangential**2)
        
        # Include depth effects
        p_static = self.ambient_pressure + self.water_density * 9.81 * self.depth
        sigma = (p_static - self.vapor_pressure) / (0.5 * self.water_density * V_total**2)
        
        return sigma
    
    def calculate_expanded_area_ratio(self):
        """Calculate actual expanded area ratio of the propeller"""
        r_values = np.linspace(0.2, 1.0, 50)
        blade_areas = []
        
        for r_norm in r_values:
            chord, _, _, _ = self.get_blade_properties(r_norm)
            r_actual = r_norm * self.params.diameter/2000
            blade_areas.append(chord * r_actual)
        
        total_blade_area = self.params.n_blades * 2 * simpson(blade_areas, x=r_values * self.params.diameter/2000)
        disc_area = np.pi * (self.params.diameter/2000)**2
        
        return total_blade_area / disc_area
    
    def solve(self, V_ship: float, rpm: float) -> dict:
        """Solve for MARINE propeller performance using BEMT"""
        n = rpm / 60  # rps
        omega = 2 * np.pi * n  # rad/s
        J = V_ship / (n * self.params.diameter/1000) if n > 0 else 0  # Advance ratio
        
        # Discretize blade
        n_elements = 30
        r_hub = self.params.hub_dia_top/2000  # m
        r_tip = self.params.diameter/2000  # m
        r_values = np.linspace(r_hub, r_tip, n_elements)
        
        total_thrust = 0
        total_torque = 0
        cavitation_indices = []
        noise_levels = []
        
        for i, r in enumerate(r_values):
            r_norm = (r - r_hub) / (r_tip - r_hub)
            
            # Get blade properties
            chord, thickness, camber, pitch_angle = self.get_blade_properties(r_norm)
            
            # Velocities
            V_axial = V_ship * (1 - 0.1 * r_norm)  # Wake factor for marine
            V_tangential = omega * r * (1 + 0.05 * r_norm)  # Induced velocity
            V_rel = np.sqrt(V_axial**2 + V_tangential**2)
            
            # Flow angles
            phi = np.arctan2(V_axial, V_tangential) if V_tangential > 0 else 0
            alpha = pitch_angle - phi
            
            # Reynolds number for water
            Re = V_rel * chord / self.kinematic_viscosity
            Re = max(Re, 5e4)  # Higher minimum for marine
            
            # Forces
            Cl, Cd = self.lift_drag_coefficients(alpha, Re)
            
            dr = r_values[1] - r_values[0] if i < n_elements-1 else r_values[-1] - r_values[-2]
            dL = 0.5 * self.water_density * V_rel**2 * chord * Cl * dr
            dD = 0.5 * self.water_density * V_rel**2 * chord * Cd * dr
            
            dT = dL * np.cos(phi) - dD * np.sin(phi)
            dQ = (dL * np.sin(phi) + dD * np.cos(phi)) * r
            
            total_thrust += self.params.n_blades * dT
            total_torque += self.params.n_blades * dQ
            
            # Cavitation
            sigma = self.calculate_cavitation_number(r, V_ship, rpm)
            cavitation_indices.append(1.0 / (sigma + 0.1))
            
            # Noise (marine-specific)
            tip_speed = V_tangential
            loading_noise = 20 * np.log10(abs(dT / (chord * dr)) + 1)
            thickness_noise = 10 * np.log10(tip_speed / 30 + 0.1)  # Marine noise model
            noise_levels.append(loading_noise + thickness_noise)
        
        # Efficiency
        if total_torque > 0 and omega > 0:
            efficiency = (total_thrust * V_ship) / (omega * total_torque) * 0.98  # Hull efficiency factor
        else:
            efficiency = 0
        
        # Performance metrics
        max_cavitation_index = max(cavitation_indices) if cavitation_indices else 1.0
        overall_noise = 10 * np.log10(sum(10**(nl/10) for nl in noise_levels)) if noise_levels else 0
        
        return {
            'thrust': total_thrust,
            'torque': total_torque,
            'efficiency': efficiency,
            'cavitation_index': max_cavitation_index,
            'expanded_area_ratio': self.calculate_expanded_area_ratio(),
            'noise_level': overall_noise,
            'KT': total_thrust / (self.water_density * n**2 * (self.params.diameter/1000)**4) if n > 0 else 0,
            'KQ': total_torque / (self.water_density * n**2 * (self.params.diameter/1000)**5) if n > 0 else 0,
            'J': J
        }

class MarinePropellerOptimizationProblem(Problem):
    """Multi-objective optimization problem for MARINE propeller design"""
    
    def __init__(self):
        # Design variables for MARINE propeller
        n_vars = 35  # Added EAR as design variable
        
        # MARINE-SPECIFIC bounds
        xl = np.array(
            [0.35]*10 +  # Chord: wide for marine (0.35-0.55)
            [0.08]*10 +  # Thickness: thick for marine (0.08-0.20)
            [0.01]*10 +  # Camber: moderate (0.01-0.06)
            [0.7,        # Pitch ratio: typical marine range
             10,         # Skew: positive for marine
             0,          # Rake: typically aft (positive)
             3,          # n_blades: 3-6 typical for marine
             0.4]        # Expanded area ratio: 0.4-1.0 for marine
        )
        xu = np.array(
            [0.55]*10 +  # Chord upper bound
            [0.20]*10 +  # Thickness upper bound
            [0.06]*10 +  # Camber upper bound
            [1.4,        # Pitch ratio upper
             35,         # Skew upper
             15,         # Rake upper
             6,          # n_blades upper
             1.0]        # EAR upper
        )
        
        super().__init__(n_var=n_vars, n_obj=3, n_constr=2, xl=xl, xu=xu)
        
        self.eval_count = 0
        
        # Marine operating conditions
        self.operating_conditions = {
            'V_ship': 5.0,  # m/s (10 knots) - typical small boat speed
            'rpm': 2000,  # RPM - typical for 180mm marine prop
            'rho': 1025.0,  # kg/m³ (seawater)
            'nu': 1.35e-6,  # m²/s (seawater)
            'depth': 3.0,  # m - shallow water
            'p_inf': 101325,  # Pa
            'pv': 2340  # Pa
        }
    
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate objective functions and constraints"""
        n_pop = x.shape[0]
        
        f1 = np.zeros(n_pop)  # Cavitation index
        f2 = np.zeros(n_pop)  # Negative efficiency
        f3 = np.zeros(n_pop)  # Noise level
        
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
                params.expanded_area_ratio = x[i, 34]
                
                # Run BEMT analysis
                solver = BEMTSolver(params, self.operating_conditions)
                results = solver.solve(
                    V_ship=self.operating_conditions['V_ship'],
                    rpm=self.operating_conditions['rpm']
                )
                
                # Objectives
                f1[i] = results['cavitation_index']
                f2[i] = -results['efficiency'] if results['efficiency'] > 0 else 1.0
                f3[i] = results['noise_level']
                
                # Constraints for marine propeller
                g1[i] = 50 - results['thrust']  # Higher thrust requirement for marine
                g2[i] = 0.08 - np.min(params.thickness_distribution)  # Thicker minimum for marine
                
            except Exception as e:
                # Penalty values
                f1[i] = 10.0
                f2[i] = 1.0
                f3[i] = 100.0
                g1[i] = 100.0
                g2[i] = 100.0
            
            self.eval_count += 1
            
            if self.eval_count % 50 == 0:
                print(f"Eval {self.eval_count}: Cav={f1[i]:.3f}, Eff={-f2[i]:.3f}, Noise={f3[i]:.1f}dB")
                print(f"  Marine specs: Blades={int(x[i, 33])}, P/D={x[i, 30]:.2f}, EAR={x[i, 34]:.2f}")
        
        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack([g1, g2])

def optimize_marine_propeller(n_generations: int = 40, population_size: int = 80):
    """Run NSGA-II optimization for MARINE propeller"""
    
    problem = MarinePropellerOptimizationProblem()
    
    algorithm = NSGA2(
        pop_size=population_size,
        n_offsprings=25,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=12),
        mutation=PM(eta=15, prob=0.25),
        eliminate_duplicates=True
    )
    
    termination = get_termination("n_gen", n_generations)
    
    print(f"Starting MARINE propeller optimization")
    print(f"Population: {population_size}, Generations: {n_generations}")
    print("Objectives: Minimize [Cavitation, -Efficiency, Noise]")
    print("Marine-specific geometry with wide blades and proper EAR")
    print("-" * 60)
    
    try:
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=None,
                       save_history=True,
                       verbose=True)
        return res
    except Exception as e:
        print(f"Optimization error: {e}")
        return None

def visualize_marine_propeller(params: PropellerParams, save_path: str = None):
    """Visualize MARINE propeller with performance metrics"""
    geom = MarinePropellerGeometry(params)
    prop_mesh = geom.create_propeller()
    
    # Calculate performance
    solver = BEMTSolver(params, {
        'V_ship': 5.0,
        'rpm': 2000,
        'rho': 1025.0,
        'nu': 1.35e-6,
        'depth': 3.0,
        'p_inf': 101325,
        'pv': 2340
    })
    perf = solver.solve(V_ship=5.0, rpm=2000)
    
    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    
    # 3D view
    ax1 = fig.add_subplot(221, projection='3d')
    
    vertices = prop_mesh.vertices
    ax1.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    triangles=prop_mesh.faces,
                    cmap='ocean', alpha=0.9, edgecolor='none')
    
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'Marine Propeller - {params.n_blades} Blades, EAR={params.expanded_area_ratio:.2f}')
    
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
    ax1.view_init(elev=25, azim=45)
    
    # Top view - shows blade width clearly
    ax2 = fig.add_subplot(222)
    ax2.triplot(vertices[:, 0], vertices[:, 1], prop_mesh.faces, 'b-', alpha=0.2, linewidth=0.3)
    ax2.fill(vertices[:, 0], vertices[:, 1], 'navy', alpha=0.3)
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title(f'Top View - Marine Blade Pattern')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Performance metrics
    ax3 = fig.add_subplot(223)
    ax3.axis('off')
    metrics_text = f"""
    MARINE PROPELLER PERFORMANCE
    Operating: 2000 RPM, 10 knots (5 m/s)
    
    Thrust: {perf['thrust']:.1f} N
    Torque: {perf['torque']:.2f} Nm
    Efficiency: {perf['efficiency']:.3f} ({perf['efficiency']*100:.1f}%)
    
    Cavitation Metrics:
    Cavitation Index: {perf['cavitation_index']:.3f}
    Expanded Area Ratio: {perf['expanded_area_ratio']:.3f}
    Noise Level: {perf['noise_level']:.1f} dB
    
    Marine Geometry:
    Number of Blades: {params.n_blades}
    Diameter: {params.diameter:.1f} mm
    Pitch Ratio (P/D): {params.pitch_ratio:.3f}
    Skew: {params.skew:.1f}°
    Rake: {params.rake:.1f}°
    EAR: {params.expanded_area_ratio:.3f}
    Cup: {params.cup:.3f}
    
    Coefficients:
    KT: {perf['KT']:.4f}
    KQ: {perf['KQ']:.4f}
    J: {perf['J']:.3f}
    """
    ax3.text(0.1, 0.9, metrics_text, transform=ax3.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    # Blade distributions
    ax4 = fig.add_subplot(224)
    r_norm = np.linspace(0, 1, len(params.chord_distribution))
    ax4.plot(r_norm, params.chord_distribution, 'b-', label='Chord', linewidth=2)
    ax4.plot(r_norm, params.thickness_distribution, 'r-', label='Thickness', linewidth=2)
    ax4.plot(r_norm, params.camber_distribution * 5, 'g-', label='Camber (×5)', linewidth=2)
    ax4.set_xlabel('Normalized Radius (r/R)')
    ax4.set_ylabel('Normalized Value')
    ax4.set_title('Marine Blade Distributions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 0.6)
    
    plt.suptitle('MARINE PROPELLER DESIGN', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    try:
        plt.show()
    except:
        plt.close()
    
    return prop_mesh

def save_marine_design(params: PropellerParams, filepath: str, performance: dict = None):
    """Save MARINE propeller design"""
    geom = MarinePropellerGeometry(params)
    prop_mesh = geom.create_propeller()
    
    # Export
    if filepath.endswith('.stl'):
        prop_mesh.export(filepath, file_type='stl')
    elif filepath.endswith('.3mf'):
        prop_mesh.export(filepath, file_type='3mf')
    else:
        raise ValueError("Unsupported file format")
    
    print(f"Marine propeller saved to: {filepath}")
    
    # Save parameters
    params_dict = {
        'type': 'MARINE_PROPELLER',
        'geometry': {
            'n_blades': params.n_blades,
            'diameter': params.diameter,
            'hub_dia_base': params.hub_dia_base,
            'hub_dia_top': params.hub_dia_top,
            'hub_height': params.hub_height,
            'pitch_ratio': params.pitch_ratio,
            'rake': params.rake,
            'skew': params.skew,
            'expanded_area_ratio': params.expanded_area_ratio,
            'cup': params.cup,
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
    """Main execution function for MARINE propeller optimization"""
    
    # Output directory
    output_dir = "/HPC/matthew.barry/propeller_output"
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    except Exception as e:
        print(f"Warning: Could not create output directory: {e}")
        output_dir = "."
    
    print("\n" + "="*60)
    print("MARINE PROPELLER OPTIMIZATION SYSTEM")
    print("Boat/Ship Propeller Design with Proper Marine Geometry")
    print("="*60)
    print("\nMarine-Specific Features:")
    print("✓ Wide blade design (high EAR)")
    print("✓ Thick blade sections for strength")
    print("✓ Marine airfoil profiles")
    print("✓ Proper skew and rake for vibration reduction")
    print("✓ Cup effect for improved thrust")
    print("✓ Ogival hub design")
    print("\nOptimization Objectives:")
    print("1. Minimize cavitation")
    print("2. Maximize efficiency")
    print("3. Minimize underwater noise")
    print("\n" + "-"*60 + "\n")
    
    # Run optimization
    print("Starting marine propeller optimization...\n")
    
    results = optimize_marine_propeller(n_generations=30, population_size=60)
    
    print("\n" + "-"*60)
    print("Optimization Complete!")
    print("-"*60 + "\n")
    
    # Process results
    if results is not None and hasattr(results, 'X') and results.X is not None and len(results.X) > 0:
        # Get best solution
        if hasattr(results, 'F') and results.F is not None:
            # Weight objectives
            F_norm = results.F.copy()
            for i in range(3):
                if F_norm[:, i].max() > F_norm[:, i].min():
                    F_norm[:, i] = (F_norm[:, i] - F_norm[:, i].min()) / (F_norm[:, i].max() - F_norm[:, i].min())
            
            scores = np.sum(F_norm * [0.4, 0.4, 0.2], axis=1)
            best_idx = np.argmin(scores)
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
            optimized_params.expanded_area_ratio = best_solution[34]
            optimized_params.cup = 0.02  # Default cup value
    else:
        # Create default marine propeller
        print("Using default marine propeller design...")
        optimized_params = PropellerParams()
        optimized_params.n_blades = 4
        optimized_params.pitch_ratio = 1.0
        optimized_params.skew = 20.0
        optimized_params.rake = 5.0
        optimized_params.expanded_area_ratio = 0.65
        optimized_params.cup = 0.02
    
    # Calculate performance
    solver = BEMTSolver(optimized_params, {
        'V_ship': 5.0,
        'rpm': 2000,
        'rho': 1025.0,
        'nu': 1.35e-6,
        'depth': 3.0,
        'p_inf': 101325,
        'pv': 2340
    })
    
    try:
        final_performance = solver.solve(V_ship=5.0, rpm=2000)
        
        print("\nBest Marine Propeller Design:")
        print(f"  Number of blades: {optimized_params.n_blades}")
        print(f"  Pitch ratio (P/D): {optimized_params.pitch_ratio:.3f}")
        print(f"  Skew angle: {optimized_params.skew:.1f}°")
        print(f"  Rake angle: {optimized_params.rake:.1f}°")
        print(f"  Expanded Area Ratio: {optimized_params.expanded_area_ratio:.3f}")
        print(f"\nPerformance (@ 2000 RPM, 10 knots):")
        print(f"  Thrust: {final_performance['thrust']:.1f} N")
        print(f"  Efficiency: {final_performance['efficiency']*100:.1f}%")
        print(f"  Cavitation Index: {final_performance['cavitation_index']:.3f}")
        print(f"  Noise Level: {final_performance['noise_level']:.1f} dB")
        
    except Exception as e:
        print(f"Error calculating performance: {e}")
        final_performance = None
    
    # Save files
    stl_path = os.path.join(output_dir, "marine_propeller.stl")
    threedf_path = os.path.join(output_dir, "marine_propeller.3mf")
    png_path = os.path.join(output_dir, "marine_propeller.png")
    
    # Visualize
    print("\n" + "-"*60)
    print("Generating marine propeller visualization...")
    print("-"*60)
    
    try:
        mesh = visualize_marine_propeller(optimized_params, save_path=png_path)
        print(f"Visualization saved to: {png_path}")
    except Exception as e:
        print(f"Warning: Could not generate visualization: {e}")
    
    # Save designs
    print("\nSaving marine propeller files...")
    
    try:
        save_marine_design(optimized_params, stl_path, final_performance)
        save_marine_design(optimized_params, threedf_path, final_performance)
        
        print("\n" + "="*60)
        print("MARINE PROPELLER OPTIMIZATION COMPLETE!")
        print("="*60)
        print(f"✓ {stl_path}")
        print(f"✓ {threedf_path}")
        print(f"✓ {png_path}")
        print(f"\nAll files saved in: {output_dir}")
        print("\nThis is a proper MARINE propeller design with:")
        print("- Wide blades for water operation")
        print("- Thick sections for strength")
        print("- Marine-specific geometry")
        
    except Exception as e:
        print(f"Error saving files: {e}")

if __name__ == "__main__":
    main()