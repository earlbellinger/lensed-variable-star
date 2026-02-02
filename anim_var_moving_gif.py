import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from scipy import signal

# --- Configuration ---
OUTPUT_FILENAME = "moving_variable_source.gif"
FPS = 20
DURATION_SEC = 10
TOTAL_FRAMES = FPS * DURATION_SEC
RESOLUTION = 150  

# Physics Parameters
GRID_SIZE = 2.0
LENS_DIST = 4.0
SOURCE_DIST = 8.0
SRC_Y = 0.1          # Y stays fixed
SRC_X_START = -2.05   # X moves from here...
SRC_X_END = 2.05      # ...to here
SHAPE_PARAM = 0.7 
AMPLITUDE = 1.5
CYCLES = 10.0

# Setup Plot Style
plt.style.use('dark_background')

# --- Physics Functions ---

def get_lightcurve_flux(phase, shape_param, amplitude=1.0):
    """Vectorized flux calculation based on phase."""
    t = 2 * np.pi * phase
    wave_sin = np.sin(t)
    wave_saw = signal.sawtooth(t, width=0.1)
    combined_wave = (1 - shape_param) * wave_sin + shape_param * wave_saw
    return 1.0 + (amplitude * 0.5 * combined_wave)

def lens_model_sis(x, y, theta_E):
    """Singular Isothermal Sphere Model."""
    r = np.sqrt(x**2 + y**2) + 1e-6 
    alpha_x = theta_E * (x / r)
    alpha_y = theta_E * (y / r)
    return alpha_x, alpha_y

# --- Static Pre-Computation ---
# We only pre-compute the Lens properties (grid and deflection angles)
# because the Lens itself doesn't move.

print("--- Initializing Static Geometry ---")

# 1. Distances & Constants
dist_ls = SOURCE_DIST - LENS_DIST
mass_scale = 5.0
theta_E = (mass_scale * np.sqrt(dist_ls / (LENS_DIST * SOURCE_DIST))) if dist_ls > 0 else 0

# Time delay scaling factor
td_scale = 0.0
if LENS_DIST < SOURCE_DIST:
    td_scale = ((LENS_DIST * SOURCE_DIST) / dist_ls) * 0.05

# 2. Grid Generation
x_grid = np.linspace(-GRID_SIZE, GRID_SIZE, RESOLUTION)
y_grid = np.linspace(-GRID_SIZE, GRID_SIZE, RESOLUTION)
xx, yy = np.meshgrid(x_grid, y_grid)

# 3. Ray Tracing (The Lens Equation)
# These beta values tell us where a light ray hitting pixel (x,y) came from on the source plane
alpha_x, alpha_y = lens_model_sis(xx, yy, theta_E)
beta_x = xx - alpha_x
beta_y = yy - alpha_y

# --- Plot Initialization ---

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [1.5, 1]})

# Left Panel: Lightcurve
lc_phases = np.linspace(0, CYCLES, 300)
lc_fluxes = get_lightcurve_flux(lc_phases, SHAPE_PARAM, AMPLITUDE)
ax1.plot(lc_phases, lc_fluxes, color='#FF4B4B', lw=2, label='Intrinsic Source')
ax1.set_xlim(0, CYCLES)
ax1.set_ylim(0, 2.0)
ax1.set_xlabel("Pulsation Phase")
ax1.set_ylabel("Relative Flux")

# Dynamic elements for Ax1
line_curr_phase = ax1.axvline(0, color='white', linestyle='--', alpha=0.9, label='Current Phase')
line_delayed = ax1.axvline(0, color='#4B4BFF', linestyle=':', lw=2, label='Observed (Avg Delay)')
ax1.legend(loc='upper right', frameon=False, fontsize=8)

# Right Panel: Lensed Image
im_plot = ax2.imshow(np.zeros_like(xx), extent=[-GRID_SIZE, GRID_SIZE, -GRID_SIZE, GRID_SIZE], 
                   origin='lower', cmap='magma', vmin=0, vmax=1.8)
ax2.plot(0, 0, 'w+', markersize=10, alpha=0.5) # Lens Center

# The source marker (Green circle) needs to be a variable now
source_marker, = ax2.plot([], [], 'go', markersize=5, fillstyle='none') 

ax2.set_xlabel("")
ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

fig.tight_layout()

# --- Animation Loop ---

def update(frame):
    # 1. Calculate Progression
    progress = frame / TOTAL_FRAMES
    
    # Move the Source X from Start to End
    current_src_x = SRC_X_START + (SRC_X_END - SRC_X_START) * progress
    
    # Global Phase (0 to CYCLES cycles)
    global_phase = progress * CYCLES 

    # 2. Recalculate Source Profile (The shape of the ring changes as source moves)
    source_radius = 0.15
    # Distance from ray-traced coordinate (beta) to the CURRENT source position
    dist_to_source = np.sqrt((beta_x - current_src_x)**2 + (beta_y - SRC_Y)**2)
    base_intensity = np.exp(-(dist_to_source**2) / (2 * (source_radius/2)**2))
    base_intensity[dist_to_source > source_radius] = 0

    # 3. Recalculate Time Delay Map
    # The geometric delay depends on where the source is relative to the lens center
    norm_factor = np.sqrt(current_src_x**2 + SRC_Y**2 + 1e-9)
    # Projection of image position onto source axis
    time_delay_map = td_scale * (xx * (current_src_x/norm_factor) + yy * (SRC_Y/norm_factor))

    # 4. Apply Pulsation (Variable Star Physics)
    # Phase at pixel = Global Time - Time it took light to get here
    pixel_phases = (global_phase - time_delay_map) 
    flux_map = get_lightcurve_flux(pixel_phases, SHAPE_PARAM, AMPLITUDE)
    
    # Combine shape with flux
    observed_image = base_intensity * flux_map
    
    # 5. Update Visuals
    im_plot.set_data(observed_image)
    source_marker.set_data([current_src_x], [SRC_Y]) # Update green circle pos

    # Update Graph Lines
    current_x = global_phase % CYCLES
    line_curr_phase.set_xdata([current_x])
    
    # Calculate average delay for the active pixels to move the blue line
    if np.any(base_intensity > 0.01):
        avg_delay = np.mean(time_delay_map[base_intensity > 0.01])
        delayed_x = (global_phase - avg_delay) % CYCLES
        line_delayed.set_xdata([delayed_x])
    
    return im_plot, line_curr_phase, line_delayed, source_marker

# --- Rendering ---

print(f"--- Rendering {TOTAL_FRAMES} frames (Source moving {SRC_X_START} -> {SRC_X_END}) ---")

anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=1000/FPS, blit=True)

try:
    if OUTPUT_FILENAME.endswith('.mp4'):
        writer = FFMpegWriter(fps=FPS, metadata=dict(artist='Sim'), bitrate=3000)
        anim.save(OUTPUT_FILENAME, writer=writer)
    else:
        writer = PillowWriter(fps=FPS)
        anim.save(OUTPUT_FILENAME, writer=writer)
        
    print(f"✅ Success! Saved to {OUTPUT_FILENAME}")
except Exception as e:
    print(f"❌ Error saving animation: {e}")
