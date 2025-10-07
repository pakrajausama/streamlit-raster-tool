import os
os.environ['PROJ_LIB'] = r"E:\Jupyter nt\webap\env\Lib\site-packages\pyproj\proj_dir\share\proj"

import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
from io import BytesIO
from rasterio.warp import calculate_default_transform, reproject, Resampling





# Set page config first (must be first Streamlit command)
st.set_page_config(
    page_title="PakGeoHub - GIS Converter",
    page_icon="🌍", 
    layout="centered"
)

# --- Complete UI Cleanup ---
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display:none;}
#stAppViewChip {display:none;}
#stAppViewAvatar {display:none;}
.stAppViewHeader {display:none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

WEB_MERCATOR_CRS = "EPSG:3857"

st.set_page_config(page_title="Raster Map Designer", layout="centered")
st.title("🗺️ Create Stunning Raster Maps in Seconds")

st.markdown("""
    <style>
        body { background-color: #f0f8ff; font-family: 'Arial', sans-serif; }
        .main { padding: 20px; }
        .sidebar .sidebar-content { background-color: #e6f7ff; }
        .stButton>button {
            background-color: #1e90ff; color: white; border-radius: 5px; font-size: 16px; font-weight: bold;
        }
        .stButton>button:hover { background-color: #4682b4; }
        .stTextInput input { font-size: 16px; }
    </style>
""", unsafe_allow_html=True)

st.header("📂 Upload Your Rasters")
uploaded_rasters = st.file_uploader("Upload one or more GeoTIFF files:", type=["tif"], accept_multiple_files=True)

if uploaded_rasters:
    total = len(uploaded_rasters)

    with st.sidebar.expander("🎨 Raster Style", expanded=False):
        color_map = st.selectbox("Color Ramp", plt.colormaps())
        raster_opacity = st.slider("Opacity", 0.0, 1.0, 1.0)

    with st.sidebar.expander("🎚️ Colorbar", expanded=False):
        colorbar_title = st.text_input("Colorbar Title", "Pixel Value")
        colorbar_orientation = st.radio("Orientation", ["Vertical", "Horizontal"], index=0)
        colorbar_fraction = st.slider("Size", 0.01, 0.2, 0.03)
        colorbar_pad = st.slider("Padding", 0.01, 0.2, 0.04)
        colorbar_fontsize = st.slider("Font Size", 6, 30, 10)  # ➕ New: Font size control

    with st.sidebar.expander("🗂️ Axis & Grid", expanded=False):
        show_axis = st.checkbox("Show Coordinate Frame", True)
        show_gridlines = st.checkbox("Show Gridlines", True)
        grid_color = st.color_picker("Gridline Color", "#808080")
        grid_style = st.selectbox("Gridline Style", ["--", "-.", ":", "-"])

    with st.sidebar.expander("📐 Layout", expanded=False):
        title_font_size = st.slider("Title Font Size", 10, 40, 16)
        font_family = st.selectbox("Font Family", ["Arial", "Times New Roman", "Courier New", "Verdana", "Georgia", "Comic Sans MS"])

        # ➕ Allow any figure size using float inputs (removed limit of 20)
        figsize_width = st.number_input("Width (inches)", value=10.0, min_value=1.0)
        figsize_height = st.number_input("Height (inches)", value=10.0, min_value=1.0)

        cols = st.number_input("Columns", 1, 6, 2)
        rows = st.number_input("Rows", 1, 10, 1)
        hspace = st.slider("Horizontal Space", 0.0, 1.0, 0.05)
        vspace = st.slider("Vertical Space", 0.0, 1.0, 0.05)

    with st.sidebar.expander("🗺️ Basemap", expanded=False):
        basemap_choice = st.radio("Basemap", ["None", "OpenStreetMap", "Satellite"])
        basemap_opacity = st.slider("Basemap Opacity", 0.0, 1.0, 1.0)

    with st.sidebar.expander("📝 Raster Titles", expanded=False):
        custom_titles = []
        for i, file in enumerate(uploaded_rasters):
            name = st.text_input(f"Title for Raster {i+1}", file.name)
            custom_titles.append(name)

    with st.sidebar.expander("📸 Export", expanded=False):
        dpi_resolution = st.slider("DPI (Resolution)", 72, 600, 300)  # ➕ New: DPI control

    fig, axes = plt.subplots(rows, cols, figsize=(figsize_width, figsize_height), constrained_layout=False)
    fig.subplots_adjust(hspace=vspace, wspace=hspace)

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    else:
        axes = axes.tolist()

    flat_axes = [ax for row in axes for ax in row]

    for idx in range(total):
        ax = flat_axes[idx]

        with rasterio.open(uploaded_rasters[idx]) as src:
            raster = src.read(1, masked=True)
            bounds = src.bounds
            crs = src.crs

            if basemap_choice != "None":
                transform, width, height = calculate_default_transform(
                    crs, WEB_MERCATOR_CRS, src.width, src.height, *src.bounds
                )
                reprojected = np.empty((height, width), dtype=raster.dtype)

                reproject(
                    source=raster,
                    destination=reprojected,
                    src_transform=src.transform,
                    src_crs=crs,
                    dst_transform=transform,
                    dst_crs=WEB_MERCATOR_CRS,
                    resampling=Resampling.nearest
                )

                raster = reprojected
                bounds = rasterio.transform.array_bounds(height, width, transform)
                crs = WEB_MERCATOR_CRS

        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.ticklabel_format(useOffset=False, style='plain')

        if basemap_choice != "None":
            try:
                if basemap_choice == "OpenStreetMap":
                    ctx.add_basemap(ax, crs=crs, source=ctx.providers.OpenStreetMap.Mapnik, alpha=basemap_opacity)
                elif basemap_choice == "Satellite":
                    ctx.add_basemap(ax, crs=crs, source=ctx.providers.Esri.WorldImagery, alpha=basemap_opacity)
            except Exception as e:
                st.warning(f"⚠️ Basemap failed: {e}")

        img = ax.imshow(
            raster,
            cmap=color_map,
            extent=[bounds[0], bounds[2], bounds[1], bounds[3]],
            alpha=raster_opacity,
        )

        ax.set_title(custom_titles[idx], fontsize=title_font_size, fontname=font_family, pad=4)

        if show_axis:
            ax.set_xlabel("", fontname=font_family)
            ax.set_ylabel("", rotation=90, fontname=font_family)
            for label in ax.get_xticklabels():
                label.set_fontname(font_family)
            for label in ax.get_yticklabels():
                label.set_rotation(90)
                label.set_fontname(font_family)
        else:
            ax.set_axis_off()

        if show_gridlines and show_axis:
            ax.grid(True, color=grid_color, linestyle=grid_style)

        cbar = fig.colorbar(
            img,
            ax=ax,
            orientation="horizontal" if colorbar_orientation == "Horizontal" else "vertical",
            fraction=colorbar_fraction,
            pad=colorbar_pad,
            shrink=0.8,
            anchor=(0.5, 0.5)
        )
        cbar.set_label(colorbar_title, fontsize=colorbar_fontsize)
        if colorbar_orientation == "Horizontal":
            cbar.ax.xaxis.label.set_fontname(font_family)
            for label in cbar.ax.get_xticklabels():
                label.set_fontsize(colorbar_fontsize)
                label.set_fontname(font_family)
        else:
            cbar.ax.yaxis.label.set_fontname(font_family)
            for label in cbar.ax.get_yticklabels():
                label.set_fontsize(colorbar_fontsize)
                label.set_fontname(font_family)

    for ax in flat_axes[total:]:
        ax.set_visible(False)

    st.pyplot(fig)

    def save_plot(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi_resolution)
        buf.seek(0)
        return buf

    st.download_button(
        label="💾 Download Map as PNG",
        data=save_plot(fig),
        file_name="raster_map_grid.png",
        mime="image/png"
    )
