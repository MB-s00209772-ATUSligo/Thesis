import numpy as np
from deap import base, creator, tools, algorithms
import random
from scipy.spatial.distance import euclidean
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PropellerConstraints:
    """
    Defines all geometric and physical constraints for the propeller design.
    All dimensions are in meters unless otherwise specified.
    """
    def __init__(self):
        # Primary constraint: fixed diameter of 180mm
        self.DIAMETER = 0.18  # meters
        self.MAX_LENGTH = 0.4  # 400mm maximum length
        
        # Hub constraints (typically 20-25% of diameter for marine propellers)
        self.MIN_HUB_DIAMETER = 0.2 * self.DIAMETER
        self.MAX_HUB_DIAMETER = 0.25 * self.DIAMETER
        
        # Chord length constraints (typically 10-40% of diameter)
        self.MAX_CHORD = 0.99 * self.DIAMETER
        self.MIN_CHORD = 0.01 * self.DIAMETER
        
        # Thickness constraints relative to chord length
        self.MIN_THICKNESS_RATIO = 0.01
        self.MAX_THICKNESS_RATIO = 0.99
        
        # Pitch angle constraints (degrees)
        self.MIN_PITCH_ANGLE = 5
        self.MAX_PITCH_ANGLE = 90

        self.MIN_BLADES = 2
        self.MAX_BLADES = 20
        
        # Operating conditions
        self.DESIGN_SPEED = 30  # knots
        self.NOMINAL_RPM = 3000  # typical for 180mm propeller

class Propeller:
    """
    Represents a complete propeller design with all geometric and performance parameters.
    Handles initialization and validation of propeller geometry.
    """
    def __init__(self, num_sections=10):
        self.num_sections = num_sections
        self.constraints = PropellerConstraints()
        
        # Fixed diameter as per requirement
        self.diameter = self.constraints.DIAMETER
        self.max_length = self.constraints.MAX_LENGTH  # 400mm maximum length

        # Parameters to be optimized
        self.chord_lengths = None
        self.pitch_angles = None
        self.thickness = None
        self.num_blades = None
        self.hub_ratio = None
        
        # Performance metrics (calculated later)
        self.efficiency = None
        self.noise_level = None
        self.cavitation_index = None
    
    def initialize_random(self):
        """
        Initialize a random propeller design within the geometric constraints,
        ensuring the length constraint is met.
        """
        # Generate radial positions for blade sections
        radial_positions = np.linspace(0.2, 1.0, self.num_sections)
        
        # Calculate maximum allowable pitch angle at each radius to meet length constraint
        max_pitch_angles = np.degrees(np.arctan(self.max_length / (radial_positions * self.diameter)))
        
        # Initialize pitch angles with length constraint
        base_pitch = random.uniform(self.constraints.MIN_PITCH_ANGLE,
                                  min(max_pitch_angles.min(), self.constraints.MAX_PITCH_ANGLE))
        pitch_reduction = np.linspace(0, 0.2, self.num_sections)
        self.pitch_angles = base_pitch * (1 - pitch_reduction)
        
        # Ensure pitch angles don't violate length constraint
        total_length = np.max(radial_positions * self.diameter * np.tan(np.radians(self.pitch_angles)))
        if total_length > self.max_length:
            scaling_factor = self.max_length / total_length
            self.pitch_angles *= scaling_factor
        
        # Initialize chord distribution (elliptical for efficiency)
        max_chord_distribution = self.constraints.MAX_CHORD * np.sqrt(1 - radial_positions**2)
        min_chord_distribution = self.constraints.MIN_CHORD * np.ones_like(radial_positions)
        self.chord_lengths = np.array([
            random.uniform(min_chord_distribution[i], max_chord_distribution[i])
            for i in range(self.num_sections)
        ])
        
        # Initialize thickness with hydrodynamic profile
        max_thickness = self.chord_lengths * self.constraints.MAX_THICKNESS_RATIO
        min_thickness = self.chord_lengths * self.constraints.MIN_THICKNESS_RATIO
        thickness_distribution = np.exp(-radial_positions)
        self.thickness = min_thickness + (max_thickness - min_thickness) * thickness_distribution
        
        # Initialize number of blades within expanded range
        self.num_blades = random.randint(self.constraints.MIN_BLADES, self.constraints.MAX_BLADES)
        
        # Initialize hub ratio
        self.hub_ratio = random.uniform(
            self.constraints.MIN_HUB_DIAMETER / self.diameter,
            self.constraints.MAX_HUB_DIAMETER / self.diameter
        )

def calculate_efficiency(propeller):
    """
    Calculate propeller efficiency using a combination of theoretical and empirical methods.
    Returns a value between 0 and 1, where 1 is perfect efficiency.
    """
    # Calculate basic geometric efficiency factors
    mean_pitch_angle = np.mean(propeller.pitch_angles)
    pitch_efficiency = np.cos(np.radians(mean_pitch_angle)) * 0.9
    
    # Calculate blade area ratio (BAR)
    mean_chord = np.mean(propeller.chord_lengths)
    blade_area = mean_chord * (propeller.diameter/2) * propeller.num_blades
    disk_area = np.pi * (propeller.diameter/2)**2
    BAR = blade_area / disk_area
    
    # Efficiency factors with blade number consideration
    area_efficiency = 1 - (propeller.hub_ratio ** 2)
    
    # Modified blade efficiency calculation for expanded blade range
    if propeller.num_blades < 3:
        blade_efficiency = 0.85  # Penalty for very few blades
    elif propeller.num_blades <= 7:
        blade_efficiency = 0.95 - (0.05 * abs(5 - propeller.num_blades))
    else:
        blade_efficiency = 0.85 - (0.02 * (propeller.num_blades - 7))  # Declining efficiency for many blades
    
    # Length-based efficiency factor
    max_length = np.max(propeller.diameter/2 * np.tan(np.radians(propeller.pitch_angles)))
    length_efficiency = 1.0 - (max_length / propeller.max_length) ** 2
    
    return pitch_efficiency * area_efficiency * blade_efficiency * length_efficiency


def estimate_noise(propeller):
    """
    Estimate propeller noise levels using semi-empirical methods.
    Considers various noise sources including cavitation, blade passage, and vortex shedding.
    """
    # Calculate tip speed
    tip_speed = (propeller.diameter * np.pi * propeller.constraints.NOMINAL_RPM) / 60
    
    # Calculate blade passing frequency
    blade_passing_freq = propeller.constraints.NOMINAL_RPM * propeller.num_blades / 60
    
    # Noise components
    tip_vortex_noise = (tip_speed / 50) ** 3  # Normalized to typical speed
    blade_loading_noise = np.mean(propeller.chord_lengths) * propeller.num_blades
    thickness_noise = np.mean(propeller.thickness) * 10

    # Additional noise factor for high blade counts
    blade_interaction_noise = 1.0 + (max(0, propeller.num_blades - 7) * 0.1)

    # Length-based noise factor (longer propellers generally produce less noise)
    max_length = np.max(propeller.diameter/2 * np.tan(np.radians(propeller.pitch_angles)))
    length_factor = 1.0 + (1.0 - max_length / propeller.max_length)
    
    # Calculate cavitation number (simplified)
    water_density = 1025  # kg/m³
    vapor_pressure = 2000  # Pa
    atmospheric_pressure = 101325  # Pa
    tip_pressure = 0.5 * water_density * tip_speed ** 2
    cavitation_number = (atmospheric_pressure - vapor_pressure) / tip_pressure
    
    # Additional noise due to cavitation
    cavitation_noise = max(0, 1 - cavitation_number) * 2
    
    total_noise = (tip_vortex_noise * blade_loading_noise * thickness_noise * 
                  (1 + cavitation_noise) * blade_passing_freq * blade_interaction_noise * length_factor)
    
    return total_noise

def evaluate_propeller(individual):
    """
    Evaluate the fitness of a propeller design.
    Returns a tuple of (noise_level, efficiency) for multi-objective optimization.
    """
    # Create propeller instance from individual's genes
    propeller = Propeller()
    
    # Decode the individual's genes into propeller parameters
    genes = np.array(individual)
    propeller.chord_lengths = genes[0:10]
    propeller.pitch_angles = genes[10:20]
    propeller.thickness = genes[20:30]
    propeller.num_blades = int(genes[30])
    propeller.hub_ratio = genes[31]

    try:

        # Calculate performance metrics
        efficiency = calculate_efficiency(propeller)
        noise = estimate_noise(propeller)

        max_radius = propeller.diameter / 2
        total_length = max_radius * np.max(np.tan(np.radians(propeller.pitch_angles)))

        # Apply penalties if length constraint is violated
        if total_length > propeller.max_length:
            efficiency *= 0.5  # Penalty for exceeding length
            noise *= 2.0      # Increase noise penalty
    
        # Apply penalties for constraint violations
        if any(propeller.chord_lengths > propeller.constraints.MAX_CHORD) or \
        any(propeller.chord_lengths < propeller.constraints.MIN_CHORD):
         efficiency *= 0.5
         noise *= 2
    
        if any(propeller.thickness/propeller.chord_lengths > propeller.constraints.MAX_THICKNESS_RATIO) or \
           any(propeller.thickness/propeller.chord_lengths < propeller.constraints.MIN_THICKNESS_RATIO):
            efficiency *= 0.5
            noise *= 2
    
        return -noise, efficiency  # Negative noise because we want to minimize it
    
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        return -float('inf'), 0.0  # Return worst possible fitness in case of error

def export_to_stl(propeller, filename="optimized_propeller.stl"):
    """
    Export the propeller design to STL format.
    Generates a complete 3D model including hub, blades, and fillets.
    """
    def generate_blade_vertices(propeller, blade_number):
        vertices = []
        normals = []
        
        # Generate points along the blade
        for i in range(propeller.num_sections):
            radius_ratio = (i + 1) / propeller.num_sections
            radius = (propeller.diameter / 2) * radius_ratio
            
            # Add rake (backward tilt of blade)
            rake_angle = 15 * (1 - radius_ratio)  # Progressive rake
            rake_offset = radius * np.tan(np.radians(rake_angle))
            
            # Add skew (circumferential displacement)
            skew_angle = 10 * radius_ratio  # Progressive skew
            skew_offset = radius * np.tan(np.radians(skew_angle))
            
            # Calculate blade section coordinates
            chord = propeller.chord_lengths[i]
            pitch = propeller.pitch_angles[i]
            thickness = propeller.thickness[i]
            
            # Basic angle for this blade
            angle = (blade_number * 360 / propeller.num_blades)
            
            # Generate blade section with NACA profile
            num_points = 20
            for j in range(num_points):
                # Parameter along chord
                t = j / (num_points - 1)
                
                # NACA 66 thickness distribution (simplified)
                y_thick = thickness * (1.98 * t - 2.49 * t**2 + 1.51 * t**3)
                
                # Calculate point coordinates
                x = radius * np.cos(np.radians(angle)) - skew_offset * np.sin(np.radians(angle))
                y = radius * np.sin(np.radians(angle)) + skew_offset * np.cos(np.radians(angle))
                z = radius * np.tan(np.radians(pitch)) + rake_offset + y_thick
                
                vertices.append([x, y, z])
                
                # Calculate normal vector (simplified)
                if j < num_points - 1:
                    normal = np.cross(
                        np.array([chord * np.cos(np.radians(angle)), 
                                 chord * np.sin(np.radians(angle)), 0]),
                        np.array([0, 0, thickness])
                    )
                    normal = normal / np.linalg.norm(normal)
                    normals.append(normal)
        
        return np.array(vertices), np.array(normals)

    # Create STL file
    with open(filename, 'wb') as f:
        # Write STL header
        f.write(struct.pack('80s', b'Propeller design optimized for 180mm diameter'))
        
        # Calculate total number of triangles
        points_per_section = 1000
        triangles_per_blade = (propeller.num_sections - 1) * (points_per_section - 1) * 2
        total_triangles = triangles_per_blade * propeller.num_blades
        f.write(struct.pack('I', total_triangles))
        
        # Generate and write triangles for each blade
        for blade in range(propeller.num_blades):
            vertices, normals = generate_blade_vertices(propeller, blade)
            
            # Create triangles from vertices
            for i in range(len(vertices) - points_per_section):
                if (i + 1) % points_per_section != 0:
                    # First triangle
                    normal = normals[i // (points_per_section - 1)]
                    v1 = vertices[i]
                    v2 = vertices[i + 1]
                    v3 = vertices[i + points_per_section]
                    
                    # Write triangle to STL file
                    f.write(struct.pack('12f', *normal, *v1, *v2, *v3))
                    f.write(struct.pack('H', 0))  # Attribute byte count
                    
                    # Second triangle
                    v1 = vertices[i + 1]
                    v2 = vertices[i + points_per_section]
                    v3 = vertices[i + points_per_section + 1]
                    
                    f.write(struct.pack('12f', *normal, *v1, *v2, *v3))
                    f.write(struct.pack('H', 0))

def plot_propeller(propeller, show_3d=True):
    """
    Create visualization of the propeller design.
    Generates both 2D blade profiles and 3D visualization.
    """
    if show_3d:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each blade
        for blade in range(propeller.num_blades):
            vertices, _ = generate_blade_vertices(propeller, blade)
            vertices = np.array(vertices)
            
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Propeller Visualization')
        plt.show()
    
    # Plot 2D blade profile
    plt.figure(figsize=(10, 5))
    radial_positions = np.linspace(0.2, 1.0, propeller.num_sections)
    
    plt.subplot(131)
    plt.plot(radial_positions, propeller.chord_lengths)
    plt.title('Chord Distribution')
    plt.xlabel('Radial Position (r/R)')
    plt.ylabel('Chord Length (m)')
    
    plt.subplot(132)
    plt.plot(radial_positions, propeller.pitch_angles)
    plt.title('Pitch Distribution')
    plt.xlabel('Radial Position (r/R)')
    plt.ylabel('Pitch Angle (deg)')
    
    plt.subplot(133)
    plt.plot(radial_positions, propeller.thickness)
    plt.title('Thickness Distribution')
    plt.xlabel('Radial Position (r/R)')
    plt.ylabel('Thickness (m)')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main optimization routine using NSGA-II algorithm for multi-objective optimization
    of the propeller design.
    """
    # Problem Constants
    NGEN = 1000        # Number of generations
    POPSIZE = 10000      # Population size20

    
    # Clear any existing DEAP declarations to avoid conflicts
    if hasattr(creator, "FitnessMulti"):
        del creator.FitnessMulti
    if hasattr(creator, "Individual"):
        del creator.Individual
    
    # Create fitness and individual classes for multi-objective optimization
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))  # Minimize noise, maximize efficiency
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    # Initialize toolbox with constrained parameter ranges
    toolbox = base.Toolbox()
    constraints = PropellerConstraints()
    
    # Register genes with appropriate constraints
    # Initialize chord lengths with elliptical distribution
    def init_chord():
        r = random.random()
        return constraints.MIN_CHORD + (constraints.MAX_CHORD - constraints.MIN_CHORD) * np.sqrt(1 - r**2)
    
    # Initialize pitch with gradual reduction towards tip
    def init_pitch():
        base_pitch = random.uniform(constraints.MIN_PITCH_ANGLE, constraints.MAX_PITCH_ANGLE)
        reduction = random.uniform(0, 0.2)  # Up to 20% reduction
        return base_pitch * (1 - reduction)
    
    # Initialize thickness with hydrodynamic profile
    def init_thickness():
        chord = init_chord()
        return random.uniform(
            chord * constraints.MIN_THICKNESS_RATIO,
            chord * constraints.MAX_THICKNESS_RATIO
        )
    
    # Register all genetic operators
    #toolbox.register("attr_chord", init_chord)
    #toolbox.register("attr_pitch", init_pitch)
    #toolbox.register("attr_thickness", init_thickness)
    #toolbox.register("attr_numblades", random.choice, [3, 5])
    #toolbox.register("attr_hubratio", random.uniform,
                     #constraints.MIN_HUB_DIAMETER/constraints.DIAMETER,
                     #constraints.MAX_HUB_DIAMETER/constraints.DIAMETER)
    
    # Register genes with updated constraints
    toolbox.register("attr_chord", random.uniform, 
                     constraints.MIN_CHORD, constraints.MAX_CHORD)
    toolbox.register("attr_pitch", random.uniform, 
                     constraints.MIN_PITCH_ANGLE, constraints.MAX_PITCH_ANGLE)
    toolbox.register("attr_thickness", random.uniform, 
                     constraints.MIN_THICKNESS_RATIO, constraints.MAX_THICKNESS_RATIO)
    toolbox.register("attr_numblades", random.randint,
                     constraints.MIN_BLADES, constraints.MAX_BLADES)
    toolbox.register("attr_hubratio", random.uniform,
                     constraints.MIN_HUB_DIAMETER/constraints.DIAMETER,
                     constraints.MAX_HUB_DIAMETER/constraints.DIAMETER)
    


    # Structure initializers for individual and population
    N_PARAMS = 32  # Total number of parameters to optimize
    
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_chord,) * 10 +      # 10 chord lengths
                     (toolbox.attr_pitch,) * 10 +      # 10 pitch angles
                     (toolbox.attr_thickness,) * 10 +  # 10 thickness values
                     (toolbox.attr_numblades,) +       # 1 number of blades
                     (toolbox.attr_hubratio,),         # 1 hub ratio
                     n=1)
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Genetic operators
    toolbox.register("evaluate", evaluate_propeller)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)

    # Statistics setup
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # Create initial population
    pop = toolbox.population(n=POPSIZE)
    
    # Create Hall of Fame to keep track of best individuals
    hof = tools.ParetoFront()
    
    # Run the optimization
    print("Starting optimization...")
    final_pop, logbook = algorithms.eaMuPlusLambda(
        pop, 
        toolbox,
        mu=POPSIZE,          # Number of individuals to select for next generation
        lambda_=POPSIZE,     # Number of children to produce at each generation
        cxpb=0.7,           # Crossover probability
        mutpb=0.3,          # Mutation probability
        ngen=NGEN,          # Number of generations
        stats=stats,
        halloffame=hof,
        verbose=True
    )
    
    # Get best individual
    best_individual = tools.selBest(final_pop, k=1)[0]
    
    # Convert best individual to propeller
    best_propeller = Propeller()
    best_propeller.chord_lengths = np.array(best_individual[0:10])
    best_propeller.pitch_angles = np.array(best_individual[10:20])
    best_propeller.thickness = np.array(best_individual[20:30])
    best_propeller.num_blades = int(best_individual[30])
    best_propeller.hub_ratio = best_individual[31]
    
    # Calculate final performance metrics
    final_efficiency = calculate_efficiency(best_propeller)
    final_noise = estimate_noise(best_propeller)
    
    # Print results
    print("\nOptimization Results:")
    print(f"Number of blades: {best_propeller.num_blades}")
    print(f"Hub ratio: {best_propeller.hub_ratio:.3f}")
    print(f"Efficiency: {final_efficiency:.3f}")
    print(f"Noise level: {final_noise:.3f}")
    
    # Export the design
    export_to_stl(best_propeller, "optimized_propeller_180mm.stl")
    
    # Create visualizations
    plot_propeller(best_propeller)
    
    # Save optimization history
    save_optimization_history(logbook, "optimization_history.txt")
    
    return best_propeller, logbook, hof

def save_optimization_history(logbook, filename):
    """
    Save the optimization history to a file for later analysis.
    """
    with open(filename, 'w') as f:
        f.write("Generation,Avg_Efficiency,Avg_Noise,Best_Efficiency,Best_Noise\n")
        for gen, entry in enumerate(logbook):
            avg_values = entry['avg']
            min_values = entry['min']
            f.write(f"{gen},{avg_values[1]},{-avg_values[0]},{min_values[1]},{-min_values[0]}\n")

def analyze_results(propeller, logbook):
    """
    Analyze and visualize the optimization results.
    """
    # Plot convergence history
    gen = range(len(logbook))
    avg_eff = [entry['avg'][1] for entry in logbook]
    avg_noise = [-entry['avg'][0] for entry in logbook]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.plot(gen, avg_eff, 'b-', label='Average Efficiency')
    plt.xlabel('Generation')
    plt.ylabel('Efficiency')
    plt.title('Efficiency Convergence')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(122)
    plt.plot(gen, avg_noise, 'r-', label='Average Noise')
    plt.xlabel('Generation')
    plt.ylabel('Noise Level')
    plt.title('Noise Convergence')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print("\nDetailed Analysis:")
    print("Blade Geometry:")
    print(f"- Average chord length: {np.mean(propeller.chord_lengths):.3f} m")
    print(f"- Average pitch angle: {np.mean(propeller.pitch_angles):.1f} degrees")
    print(f"- Average thickness: {np.mean(propeller.thickness):.3f} m")
    
    # Calculate performance metrics
    tip_speed = (propeller.diameter * np.pi * propeller.constraints.NOMINAL_RPM) / 60
    advance_speed = propeller.constraints.DESIGN_SPEED * 0.514444
    
    print("\nPerformance Metrics:")
    print(f"- Tip speed: {tip_speed:.1f} m/s")
    print(f"- Advance ratio: {advance_speed/tip_speed:.3f}")
    print(f"- Blade loading coefficient: {np.mean(propeller.chord_lengths) * propeller.num_blades / propeller.diameter:.3f}")

def generate_blade_vertices(propeller, blade_number):
    """
    Generate the 3D vertices and normals for a single propeller blade.
    Uses advanced marine propeller design principles to create realistic blade geometry.
    
    Parameters:
        propeller: Propeller object containing the design parameters
        blade_number: Integer indicating which blade to generate (0 to num_blades-1)
    
    Returns:
        vertices: Array of 3D points defining the blade surface
        normals: Array of normal vectors for each vertex
    """
    vertices = []
    normals = []
    
    # Number of points along chord direction for detailed blade surface
    num_chord_points = 1000
    
    # Generate cross-sections along the blade radius
    for i in range(propeller.num_sections):
        # Calculate radial position (r/R)
        radius_ratio = (i + 1) / propeller.num_sections
        radius = (propeller.diameter / 2) * radius_ratio
        
        # Get section parameters
        chord = propeller.chord_lengths[i]
        pitch = propeller.pitch_angles[i]
        thickness = propeller.thickness[i]
        
        # Apply rake (backward tilt of blade)
        # Rake increases towards root for structural strength
        rake_angle = 15 * (1 - radius_ratio)  # degrees
        rake_offset = radius * np.tan(np.radians(rake_angle))
        
        # Apply skew (circumferential displacement)
        # Skew increases towards tip for cavitation reduction
        skew_angle = 10 * radius_ratio  # degrees
        skew_offset = radius * np.tan(np.radians(skew_angle))
        
        # Calculate base angle for this blade
        blade_angle = (blade_number * 360 / propeller.num_blades)
        
        # Generate points along the chord at this radius
        for j in range(num_chord_points):
            # Parameter along chord (0 to 1)
            chord_param = j / (num_chord_points - 1)
            
            # Generate NACA 66 thickness distribution (modified for propellers)
            # This profile is commonly used in marine propellers for good cavitation characteristics
            y_thick = thickness * (
                1.98 * chord_param -
                2.49 * chord_param**2 +
                1.51 * chord_param**3 -
                0.15 * chord_param**4
            )
            
            # Add meanline camber (modified for propeller applications)
            # Uses a parabolic meanline distribution typical for marine propellers
            camber = chord * 0.06 * (2 * chord_param - chord_param**2)
            
            # Calculate point coordinates in 3D space
            x = radius * np.cos(np.radians(blade_angle)) - \
                skew_offset * np.sin(np.radians(blade_angle)) + \
                chord_param * chord * np.cos(np.radians(blade_angle))
                
            y = radius * np.sin(np.radians(blade_angle)) + \
                skew_offset * np.cos(np.radians(blade_angle)) + \
                chord_param * chord * np.sin(np.radians(blade_angle))
                
            z = radius * np.tan(np.radians(pitch)) + \
                rake_offset + \
                y_thick + camber
            
            vertices.append([x, y, z])
            
            # Calculate normal vector using cross product of tangent vectors
            if j < num_chord_points - 1 and i < propeller.num_sections - 1:
                # Vectors along chord and radius directions
                chord_vector = np.array([
                    chord * np.cos(np.radians(blade_angle)),
                    chord * np.sin(np.radians(blade_angle)),
                    thickness
                ])
                
                radius_vector = np.array([
                    radius * np.cos(np.radians(blade_angle + 1)) - x,
                    radius * np.sin(np.radians(blade_angle + 1)) - y,
                    radius * np.tan(np.radians(pitch + 1)) - z
                ])
                
                # Calculate normal as cross product
                normal = np.cross(chord_vector, radius_vector)
                normal = normal / np.linalg.norm(normal)  # Normalize
                normals.append(normal)
    
    return np.array(vertices), np.array(normals)

if __name__ == "__main__":
    # Run the optimization
    best_propeller, logbook, hof = main()
    
    # Analyze and visualize results
    analyze_results(best_propeller, logbook)
    

    print("\nOptimization complete. Results have been saved to:")
    print("- optimized_propeller_180mm.stl (3D model)")
    print("- optimization_history.txt (convergence history)")