import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from scipy import signal
import sys

# --- Configuration ---
# Output settings
OUTPUT_FILENAME = "lensed_constant_star.gif" # Change to .gif if you don't have FFMpeg
FPS = 20
DURATION_SEC = 10
TOTAL_FRAMES = FPS * DURATION_SEC
RESOLUTION = 150  # Higher = sharper image, slower render

# Physics Parameters
GRID_SIZE = 2.0
LENS_DIST = 4.0
SOURCE_DIST = 8.0
SRC_X, SRC_Y = 0.1, 0.1
SHAPE_PARAM = 0.7  # 0 = Sine, 1 = Sawtooth
AMPLITUDE = 0

# Setup Plot Style (Dark Mode)
plt.style.use('dark_background')

# --- Physics Functions ---

def get_lightcurve_flux(phase, shape_param, amplitude=1.0):
    """Vectorized flux calculation based on phase (0.0 to 1.0)."""
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

# --- Pre-Computation (The "Fast" Part) ---
print("--- Initializing Geometry ---")

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

# 3. Ray Tracing (Lens Equation)
alpha_x, alpha_y = lens_model_sis(xx, yy, theta_E)
beta_x = xx - alpha_x
beta_y = yy - alpha_y

# 4. Source Profile (Static Intensity - The shape of the ring)
source_radius = 0.15
dist_to_source = np.sqrt((beta_x - SRC_X)**2 + (beta_y - SRC_Y)**2)
base_intensity = np.exp(-(dist_to_source**2) / (2 * (source_radius/2)**2))
base_intensity[dist_to_source > source_radius] = 0

# 5. Time Delay Map
# Approximate geometric time delay projected onto the image plane
# (Flux arrives at different pixels at different times)
lens_pot_term = theta_E * np.sqrt(xx**2 + yy**2) # Simplified potential
geom_term = 0.5 * ((xx - beta_x)**2 + (yy - beta_y)**2) # Fermat potential approx
# We use the simplified geometric projection from the original script for consistency
# delay ~ projection of position onto source axis
time_delay_map = td_scale * (xx * (SRC_X/np.sqrt(SRC_X**2 + SRC_Y**2+1e-9)) + yy * (SRC_Y/np.sqrt(SRC_X**2 + SRC_Y**2+1e-9)))

# --- Plot Initialization ---

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [1.5, 1]})
#fig.suptitle(f"Gravitational Lensing Time Delays ($D_L={LENS_DIST}, D_S={SOURCE_DIST}$)", color='white', fontsize=14)

# Left Panel: Lightcurve (Static Background)
lc_phases = np.linspace(0, 3, 300) # Show 3 cycles
lc_fluxes = get_lightcurve_flux(lc_phases, SHAPE_PARAM, AMPLITUDE)
ax1.plot(lc_phases, lc_fluxes, color='#FF4B4B', lw=2, label='Intrinsic Source')
ax1.set_xlim(0, 3)
ax1.set_ylim(0, 2.0)
#ax1.set_title("Source Lightcurve", fontsize=10)
ax1.set_xlabel("Pulsation Phase")
ax1.set_ylabel("Relative Flux")
#ax1.grid(True, alpha=0.2)

# Dynamic elements for Ax1
line_curr_phase = ax1.axvline(0, color='white', linestyle='--', alpha=0.9, label='Current Phase')
# Create a marker for the "delayed" image arrival (heuristic average delay)
avg_delay = np.mean(time_delay_map[base_intensity > 0.01]) if np.any(base_intensity > 0.01) else 0
line_delayed = ax1.axvline(0, color='#4B4BFF', linestyle=':', lw=2, label='Observed (Avg Delay)')
ax1.legend(loc='upper right', frameon=False, fontsize=8)

# Right Panel: Lensed Image
# We normalize vmax to handle the peak flux
im_plot = ax2.imshow(np.zeros_like(base_intensity), extent=[-GRID_SIZE, GRID_SIZE, -GRID_SIZE, GRID_SIZE], 
                   origin='lower', cmap='magma', vmin=0, vmax=1.8)
ax2.plot(0, 0, 'w+', markersize=10, alpha=0.5) # Lens
ax2.plot(SRC_X, SRC_Y, 'go', markersize=5, fillstyle='none') # Source
#ax2.set_title("Observed Image Plane", fontsize=10)
ax2.set_xlabel("")

# Remove axes ticks for cleanliness on the image
ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

fig.tight_layout()

# --- Animation Loop ---

def update(frame):
    # Calculate global phase (0 to 3 cycles over the duration)
    # This creates a looping effect
    global_phase = (frame / TOTAL_FRAMES) * 3.0 
    
    # 1. Update Image Plane
    # Calculate local phase for every pixel: Global Phase - Time Delay at that pixel
    pixel_phases = (global_phase - time_delay_map) 
    
    # Vectorized flux map lookup
    flux_map = get_lightcurve_flux(pixel_phases, SHAPE_PARAM, AMPLITUDE)
    
    # Apply flux modulation to the static lensed shape
    observed_image = base_intensity * flux_map
    
    im_plot.set_data(observed_image)
    
    # 2. Update Lightcurve Indicators
    # Modulo 3 to keep it within the x-axis limits
    current_x = global_phase % 3.0
    line_curr_phase.set_xdata([current_x])
    
    delayed_x = (global_phase - avg_delay) % 3.0
    line_delayed.set_xdata([delayed_x])
    
    return im_plot, line_curr_phase, line_delayed

# --- Rendering ---

print(f"--- Rendering {TOTAL_FRAMES} frames ---")

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
    print("If you lack FFMpeg, change OUTPUT_FILENAME to end in .gif")
