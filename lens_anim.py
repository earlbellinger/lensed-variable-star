import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

# --- Configuration & Styling ---
st.set_page_config(layout="wide", page_title="Lensed Variable Star Simulator")

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1 {
    color: #fafafa;
}
div.stButton > button {
    width: 100%;
    background-color: #FF4B4B;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# --- Physics Functions ---

def get_lightcurve_flux(phase, shape_param, amplitude=1.0):
    """
    Generates a lightcurve flux combining Sine and Sawtooth waves.
    phase: Can be scalar or numpy array.
    """
    # Ensure phase is within 0-1 cycle for calculation, but input can be anything
    t = 2 * np.pi * phase
    
    # Pure Sine Wave
    wave_sin = np.sin(t)
    
    # Sawtooth Wave (width=1.0 is pure saw)
    wave_saw = signal.sawtooth(t, width=0.1)
    
    # Linear interpolation
    combined_wave = (1 - shape_param) * wave_sin + shape_param * wave_saw
    
    # Base flux = 1.0, Amplitude varies
    flux = 1.0 + (amplitude * 0.5 * combined_wave)
    return flux

def lens_model_sis(x, y, theta_E):
    r = np.sqrt(x**2 + y**2) + 1e-6 
    alpha_x = theta_E * (x / r)
    alpha_y = theta_E * (y / r)
    return alpha_x, alpha_y

def geometric_time_delay_scale(D_l, D_s):
    if D_l >= D_s: return 0.0
    D_ls = D_s - D_l
    td_scale = (D_l * D_s) / D_ls
    return td_scale * 0.05 

# --- App Layout ---

st.title("Gravitationally Lensed Variable Star")

# --- Sidebar Controls ---
#st.sidebar.header("Controls")

# 1. Animation Control
#st.sidebar.subheader("Animation")
if "animate" not in st.session_state:
    st.session_state.animate = False

def toggle_animation():
    st.session_state.animate = not st.session_state.animate

btn_label = "⏹ Stop Animation" if st.session_state.animate else "▶ Play Animation"
st.sidebar.button(btn_label, on_click=toggle_animation)

animation_speed = st.sidebar.slider("Animation Speed", 0.01, 0.1, 0.05)

# 2. Physics Parameters
st.sidebar.subheader("Stellar Parameters")
variable = st.sidebar.checkbox("Variable", value=True, help="Uncheck to remove stellar variability")
amplitude = 1 if variable else 0
shape_param = st.sidebar.slider("Lightcurve Shape", 0.0, 1.0, 0.84, help="0=Sine, 1=Sawtooth")
#amplitude = st.sidebar.slider("Amplitude", 0.0, 1.0, 0.8)

# If not animating, allow manual phase control
if not st.session_state.animate:
    manual_phase = st.sidebar.slider("Phase", 0.0, 1.0, 0.7)
    current_phase_start = manual_phase
else:
    st.sidebar.info("Phase control disabled during animation.")
    if "phase_counter" not in st.session_state:
        st.session_state.phase_counter = 0.0
    current_phase_start = st.session_state.phase_counter

# 3. Geometry
st.sidebar.subheader("Geometry")
src_x = st.sidebar.slider("Source X", -2.0, 2.0, 0.1)
src_y = st.sidebar.slider("Source Y", -2.0, 2.0, 0.1)
dist_lens = st.sidebar.slider("Lens Distance (D_L)", 0.1, 10.0, 4.0)
dist_source = st.sidebar.slider("Star Distance (D_S)", dist_lens + 0.1, 15.0, 8.0)

# Derived Constants
dist_ls = dist_source - dist_lens
mass_scale = 5.0
theta_E = (mass_scale * np.sqrt(dist_ls / (dist_lens * dist_source))) if dist_ls > 0 else 0
td_scale = geometric_time_delay_scale(dist_lens, dist_source)
st.sidebar.metric("Einstein Radius (θ_E)", f"{theta_E:.2f}")

# Pre-calculate Grid (Static per frame)
grid_size = 3.0
res = 100
x_grid = np.linspace(-grid_size, grid_size, res)
y_grid = np.linspace(-grid_size, grid_size, res)
xx, yy = np.meshgrid(x_grid, y_grid)

# Lens Equation
alpha_x, alpha_y = lens_model_sis(xx, yy, theta_E)
beta_x = xx - alpha_x
beta_y = yy - alpha_y

# Source Profile (Static geometry)
source_radius = 0.15
dist_to_source = np.sqrt((beta_x - src_x)**2 + (beta_y - src_y)**2)
base_intensity = np.exp(-(dist_to_source**2) / (2 * (source_radius/2)**2))
base_intensity[dist_to_source > source_radius] = 0

# Time Delay Map (Static geometry)
# Uses simple geometric projection for delay approximation
time_delay_map = td_scale * (xx * (src_x/np.sqrt(src_x**2 + src_y**2+1e-9)) + yy * (src_y/np.sqrt(src_x**2 + src_y**2+1e-9)))

# --- Main Plotting Loop ---

# Create a placeholder for the entire main content area
plot_placeholder = st.empty()

def render_plots(phase):
    """
    Helper function to generate the Matplotlib figures
    """
    # --- 1. Calculate Frame Physics ---
    
    # Image Plane Flux
    pixel_phases = (phase - time_delay_map) % 1.0
    flux_map = get_lightcurve_flux(pixel_phases, shape_param, amplitude)
    observed_image = base_intensity * flux_map

    # Lightcurve Plot Data
    lc_phases = np.linspace(0, 3, 200)
    lc_fluxes = get_lightcurve_flux(lc_phases, shape_param, amplitude)
    
    # Counter-image delay (heuristic for plotting the blue line)
    delay_diff = td_scale * 2.0
    delayed_phase = (phase - delay_diff) % 1.0

    # --- 2. Visualization ---
    # Create one figure with 2 subplots to ensure synchronization and layout stability
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1]})
    fig.patch.set_facecolor('#0e1117')

    # -- Left Panel: Lightcurve --
    ax1.plot(lc_phases, lc_fluxes, color='#FF4B4B', lw=2, label='Intrinsic Lightcurve')
    ax1.axvline(phase % 1.0, color='white', linestyle='--', alpha=0.8, label='Source Phase')
    ax1.axvline((phase % 1.0) + 1.0, color='white', linestyle='--', alpha=0.8) # Repeat for 2nd cycle visual
    ax1.axvline((phase % 1.0) + 2.0, color='white', linestyle='--', alpha=0.8) # Repeat for 3rd cycle visual

    # Counter image line
    if theta_E > 0:
         # Shift to 0-1 range for plotting
         dp_plot = delayed_phase % 1.0
         ax1.axvline(dp_plot, color='#4B4BFF', linestyle=':', lw=2, label='Delayed Image')
         ax1.axvline(dp_plot + 1.0, color='#4B4BFF', linestyle=':', lw=2)
         ax1.axvline(dp_plot + 2.0, color='#4B4BFF', linestyle=':', lw=2)

    ax1.set_xlim([0, 3])
    ax1.set_title("Intrinsic Lightcurve", color='white')
    ax1.set_facecolor('#0e1117')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values(): spine.set_edgecolor('white')
    ax1.legend(facecolor='#0e1117', labelcolor='white', loc='upper right')

    # -- Right Panel: Image --
    ax2.imshow(observed_image, extent=[-grid_size, grid_size, -grid_size, grid_size], 
               origin='lower', cmap='magma', vmin=0, vmax=1.5)
    ax2.plot(0, 0, 'w+', markersize=10, alpha=0.5, label='Lens Center') # Lens Center
    ax2.plot(src_x, src_y, 'go', markersize=5, fillstyle='none', label='True Source Pos')
    ax2.legend(facecolor='#0e1117', labelcolor='white', loc='upper right')
    
    # Styling
    ax2.set_title("Observed Image", color='white')
    ax2.set_xlabel("Arcsec", color='white')
    ax2.set_facecolor('black')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values(): spine.set_edgecolor('white')

    return fig

# --- Execution ---

if st.session_state.animate:
    # Loop indefinitely until state changes
    # We initialize a localized counter to avoid modifying session state inside the loop constantly
    # (which triggers full reruns).
    curr_p = st.session_state.get('phase_counter', 0.0)
    
    while st.session_state.animate:
        # Update phase
        curr_p += animation_speed
        if curr_p > 1.0: curr_p -= 1.0
        
        # Render
        fig = render_plots(curr_p)
        
        # Push to UI
        with plot_placeholder.container():
            st.pyplot(fig)
            
        # Clean up memory
        plt.close(fig)
        
        # Small sleep to control frame rate
        #time.sleep(0.01)
        
    # Save state when stopped so we don't snap back to 0
    st.session_state.phase_counter = curr_p

else:
    # Static Mode (Single Render)
    fig = render_plots(current_phase_start)
    with plot_placeholder.container():
        st.pyplot(fig)
