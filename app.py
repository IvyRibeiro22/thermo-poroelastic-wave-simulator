
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Thermo-Poroelastic Wave Simulator",
    page_icon="🌊",
    layout="wide"
)

# Main header
st.title("Thermo-Poroelastic Wave Simulator")
st.caption("1D Wave Dispersion and Attenuation Analysis")

st.divider()

# Sidebar
with st.sidebar:
    st.header("Simulation Settings")

    model = st.selectbox(
        "Physical Model",
        ["Darcy", "Biot-JKD"]
    )

    st.info("Physical parameters will be added soon.")

# Main content
st.subheader("Simulation Overview")

st.write(
    "Scientific simulation of wave propagation "
    "in fluid-saturated porous media."
)

st.write(
    "The simulator will analyze the phase velocity "
    "and attenuation of P1, P2, and thermal T waves."
)

st.info(f"Selected model: {model}")

st.warning(
    "Prototype interface: numerical calculations "
    "have not been implemented yet."
)
