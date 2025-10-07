import streamlit as st
import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from tempfile import NamedTemporaryFile
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import mapclassify
from rasterio.warp import reproject, Resampling
import matplotlib.colors as mcolors





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

# ==============================
# APP TITLE
# ==============================
st.set_page_config(layout="wide")
st.title("🌍 PakGeoHub Quick Bivariate Map Tool")
st.sidebar.header("⚙️ Workflow")

# ------------------------------
# Helpers
# ------------------------------
def sample_colormap(cmap_name, t):
    """Return RGB tuple sampled from matplotlib colormap name at position t (0..1)."""
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(t)
    return np.array(rgba[:3])  # RGB only

def make_bivariate_palette(cmap_x, cmap_y, bins=3):
    """
    Create a bins x bins list of hex colors by blending samples from two colormaps.
    Simple blend: average RGB of color from x and y ramps.
    Returns flat list and 2D array of colors.
    """
    palette = []
    grid = []
    for j in range(bins):  # y axis (rows)
        row = []
        for i in range(bins):  # x axis (cols)
            # sampling positions: 0 -> low, 1 -> high
            tx = i / (bins - 1) if bins > 1 else 0.5
            ty = j / (bins - 1) if bins > 1 else 0.5
            cx = sample_colormap(cmap_x, tx)
            cy = sample_colormap(cmap_y, ty)
            # Blend: simple average (you can change weighting)
            blended = (cx + cy) / 2.0
            hexcol = mcolors.to_hex(blended)
            row.append(hexcol)
            palette.append(hexcol)
        grid.append(row)
    return palette, grid

def create_listed_cmap_from_flat(flat_colors):
    return mcolors.ListedColormap(flat_colors)

def save_fig_to_buf(fig, dpi=300, bbox_inches='tight'):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches=bbox_inches)
    buf.seek(0)
    return buf

# ------------------------------
# Step 1: Choose map type
# ------------------------------
map_type = st.sidebar.radio("Choose Map Type", ["Vector", "Raster"])

# ------------------------------
# VECTOR WORKFLOW
# ------------------------------
if map_type == "Vector":
    with st.sidebar.expander("📂 Step 1: Upload Vector", expanded=True):
        uploaded_file = st.file_uploader(
            "Upload vector file (GeoJSON or Shapefile in .zip)",
            type=["geojson", "zip"]
        )

    if uploaded_file:
        try:
            # read file
            if uploaded_file.name.lower().endswith(".geojson"):
                gdf = gpd.read_file(uploaded_file)
            else:
                # zip: save temp then read via zip://
                with NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                gdf = gpd.read_file(f"zip://{tmp_path}")

            st.sidebar.success(f"✅ File loaded ({len(gdf)} features)")

            # step 2 fields & title
            with st.sidebar.expander("🔢 Step 2: Choose Fields & Map Title", expanded=True):
                numeric_cols = gdf.select_dtypes(include="number").columns.tolist()
                if len(numeric_cols) < 2:
                    st.error("❌ Need at least 2 numeric columns for a bivariate map.")
                    st.stop()
                col1 = st.selectbox("First variable (X-axis)", numeric_cols, index=0)
                col2 = st.selectbox("Second variable (Y-axis)", numeric_cols, index=1)
                map_title = st.text_input("📝 Map Title", "Bivariate Choropleth Map")
                bins = st.slider("Number of classes per axis (bins)", 2, 5, 3, step=1)

            # step 3 legend / colors: choose two ramps for the two axes
            with st.sidebar.expander("🎨 Step 3: Legend & Color Ramps", expanded=False):
                cmap_list = sorted(m for m in plt.colormaps())
                cmap_x = st.selectbox("Color ramp for X (low→high)", cmap_list, index=cmap_list.index("Blues"))
                cmap_y = st.selectbox("Color ramp for Y (low→high)", cmap_list, index=cmap_list.index("Reds"))
                legend_size = st.slider("Legend size (relative %)", 0.08, 0.4, 0.18, step=0.02)
                margin_x = st.slider("Legend Margin X (pad)", 0, 30, 6)
                margin_y = st.slider("Legend Margin Y (pad)", 0, 30, 6)
                legend_pos = st.selectbox("Legend Position",
                                          ["Top Left (TL)", "Top Right (TR)", "Bottom Left (BL)", "Bottom Right (BR)"])
                x_label = st.text_input("Rename X-axis label", col1)
                y_label = st.text_input("Rename Y-axis label", col2)
                loc_map = {"Bottom Left (BL)": 3, "Bottom Right (BR)": 4, "Top Left (TL)": 2, "Top Right (TR)": 1}

            # step 4 grid
            with st.sidebar.expander("📐 Step 4: Grid Options", expanded=False):
                show_grid = st.checkbox("Show Grids?", False)
                grid_fontsize = st.slider("Grid Label Size", 6, 14, 9)

            # compute classes for vector
            # mask NaNs
            vals_x = gdf[col1].to_numpy()
            vals_y = gdf[col2].to_numpy()
            valid_mask = (~np.isnan(vals_x)) & (~np.isnan(vals_y))

            if valid_mask.sum() == 0:
                st.error("No valid numeric pairs to classify.")
                st.stop()

            classifier_x = mapclassify.Quantiles(vals_x[valid_mask], k=bins)
            classifier_y = mapclassify.Quantiles(vals_y[valid_mask], k=bins)

            x_class = np.full(len(gdf), np.nan)
            y_class = np.full(len(gdf), np.nan)
            x_class[valid_mask] = classifier_x.yb
            y_class[valid_mask] = classifier_y.yb
            numeric_index = (x_class * bins + y_class).astype(float)  # floats with nan where invalid

            gdf["bivar_index"] = numeric_index

            # build bivariate palette
            flat_colors, grid_colors = make_bivariate_palette(cmap_x, cmap_y, bins=bins)
            listed_cmap = create_listed_cmap_from_flat(flat_colors)

            # Map integer indices (0..bins*bins-1) to region categories for plotting
            # For geopandas, we will plot numeric categories using the ListedColormap and categorical colormap mapping
            # create an integer category column where invalid = NaN
            gdf["bivar_int"] = np.where(np.isnan(gdf["bivar_index"]), -1, gdf["bivar_index"].astype(int))

            # Plot
            figure_width = st.sidebar.number_input("Figure Width (inches)", 4.0, 20.0, 7.0, key="fig_w")
            figure_height = st.sidebar.number_input("Figure Height (inches)", 4.0, 20.0, 5.0, key="fig_h")
            fig, ax = plt.subplots(figsize=(figure_width, figure_height))
            # Plot valid categories with colormap: use categorical mapping by supplying 'cmap' and specifying vmin/vmax
            # Create a color list per feature using bivar_int
            colors_for_plot = []
            for val in gdf["bivar_int"]:
                if val == -1:
                    colors_for_plot.append("none")
                else:
                    colors_for_plot.append(flat_colors[int(val)])
            gdf.plot(color=colors_for_plot, linewidth=0.3, ax=ax, edgecolor="0.6")

            if show_grid:
                ax.grid(True, linewidth=0.3, linestyle="--")
                ax.tick_params(labelsize=grid_fontsize)
            else:
                ax.set_axis_off()

            ax.set_title(map_title, fontsize=14, pad=15)

            legend_ax = inset_axes(ax,
                                   width=f"{legend_size*100}%",
                                   height=f"{legend_size*100}%",
                                   loc=loc_map[legend_pos],
                                   borderpad=1.5)
            legend_matrix = np.arange(bins * bins).reshape(bins, bins)
            legend_ax.imshow(legend_matrix, cmap=listed_cmap, origin="lower")
            legend_ax.set_xticks([])
            legend_ax.set_yticks([])
            legend_ax.set_xlabel(x_label, fontsize=9, labelpad=margin_x)
            legend_ax.set_ylabel(y_label, fontsize=9, labelpad=margin_y)

            st.pyplot(fig, use_container_width=False)

            # download options
            with st.sidebar.expander("💾 Download Map", expanded=False):
                dpi = st.slider("DPI (Resolution)", 72, 600, 300, step=30)
                fig_dl, ax_dl = plt.subplots(figsize=(figure_width, figure_height))
                gdf.plot(color=colors_for_plot, linewidth=0.3, ax=ax_dl, edgecolor="0.6")

                # Title
                ax_dl.set_title(map_title, fontsize=14, pad=15)

                # Grid
                if show_grid:
                    ax_dl.grid(True, linewidth=0.3, linestyle="--")
                    ax_dl.tick_params(labelsize=grid_fontsize)
                else:
                    ax_dl.set_axis_off()

                # Legend
                legend_ax = inset_axes(ax_dl,
                                    width=f"{legend_size*100}%",
                                    height=f"{legend_size*100}%",
                                    loc=loc_map[legend_pos],
                                    borderpad=1.5)
                legend_matrix = np.arange(bins * bins).reshape(bins, bins)
                legend_ax.imshow(legend_matrix, cmap=listed_cmap, origin="lower")
                legend_ax.set_xticks([])
                legend_ax.set_yticks([])
                legend_ax.set_xlabel(x_label, fontsize=9, labelpad=margin_x)
                legend_ax.set_ylabel(y_label, fontsize=9, labelpad=margin_y)

                fig_dl.tight_layout()
                buf = save_fig_to_buf(fig_dl, dpi=dpi)

                st.download_button("📥 Download Map as PNG", data=buf, file_name="bivariate_map_vector.png", mime="image/png")

        except Exception as e:
            st.error(f"Error loading vector: {e}")

    else:
        st.info("📂 Upload your GeoJSON or Shapefile (.zip) file to get started.")

# ------------------------------
# RASTER WORKFLOW
# ------------------------------
# ------------------------------
# RASTER WORKFLOW
# ------------------------------
elif map_type == "Raster":
    with st.sidebar.expander("📂 Step 1: Upload Rasters", expanded=True):
        raster1_file = st.file_uploader("Raster 1 (Base) — GeoTIFF", type=["tif"])
        raster2_file = st.file_uploader("Raster 2 (to compare) — GeoTIFF", type=["tif"])

    if raster1_file and raster2_file:
        try:
            # Step 2: Map Settings
            with st.sidebar.expander("🎨 Step 2: Map Settings", expanded=True):
                cmap_list = sorted(m for m in plt.colormaps())
                cmap_x = st.selectbox("Color ramp for Raster 1 (X)", cmap_list, index=cmap_list.index("Blues"))
                cmap_y = st.selectbox("Color ramp for Raster 2 (Y)", cmap_list, index=cmap_list.index("Reds"))
                bins = st.slider("Number of classes per axis (bins)", 2, 6, 3, step=1)
                map_title = st.text_input("📝 Map Title", "Bivariate Raster Map")
                legend_size = st.slider("Legend size (relative %)", 0.08, 0.4, 0.18, step=0.02)
                margin_x = st.slider("Legend Margin X (pad)", 0, 30, 6)
                margin_y = st.slider("Legend Margin Y (pad)", 0, 30, 6)
                x_label = st.text_input("Rename X-axis label", "Raster 1")
                y_label = st.text_input("Rename Y-axis label", "Raster 2")
                show_grid = st.checkbox("Show Grids?", False)
                grid_fontsize = st.slider("Grid Label Size", 6, 14, 9)
                legend_pos = st.selectbox("Legend Position",
                                          ["Top Left (TL)", "Top Right (TR)", "Bottom Left (BL)", "Bottom Right (BR)"])
                loc_map = {"Bottom Left (BL)": 3, "Bottom Right (BR)": 4, "Top Left (TL)": 2, "Top Right (TR)": 1}

            # Step 3: Open rasters
            with rasterio.open(raster1_file) as src1, rasterio.open(raster2_file) as src2:
                arr1 = src1.read(1).astype(np.float32)
                arr2 = src2.read(1).astype(np.float32)
                t1, nodata1 = src1.transform, src1.nodata
                t2, nodata2 = src2.transform, src2.nodata

                res1_area = abs(t1.a * t1.e)
                res2_area = abs(t2.a * t2.e)

                if res1_area <= res2_area:
                    high_src, low_src = src1, src2
                    high_arr, low_arr = arr1.copy(), arr2.copy()
                    high_nodata, low_nodata = nodata1, nodata2
                else:
                    high_src, low_src = src2, src1
                    high_arr, low_arr = arr2.copy(), arr1.copy()
                    high_nodata, low_nodata = nodata2, nodata1

                dst_shape = (high_src.height, high_src.width)
                dst_transform, dst_crs = high_src.transform, high_src.crs

                resampled_low = np.full(dst_shape, np.nan, dtype=np.float32)
                reproject(
                    source=low_src.read(1, masked=True).astype(np.float32),
                    destination=resampled_low,
                    src_transform=low_src.transform,
                    src_crs=low_src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear
                )

                data1, data2 = high_arr, resampled_low
                mask = np.zeros_like(data1, dtype=bool)
                if high_nodata is not None: mask |= (data1 == high_nodata)
                if low_nodata is not None: mask |= (data2 == low_nodata)
                mask |= np.isnan(data1) | np.isnan(data2)
                data1, data2 = np.where(mask, np.nan, data1), np.where(mask, np.nan, data2)

            # Step 4: Classification
            valid_mask = (~np.isnan(data1)) & (~np.isnan(data2))
            if valid_mask.sum() == 0:
                st.error("No overlapping valid data between the two rasters.")
                st.stop()

            classifier_x = mapclassify.Quantiles(data1[valid_mask].flatten(), k=bins)
            classifier_y = mapclassify.Quantiles(data2[valid_mask].flatten(), k=bins)

            x_class = np.full(data1.shape, np.nan)
            y_class = np.full(data2.shape, np.nan)
            x_class[valid_mask], y_class[valid_mask] = classifier_x.yb, classifier_y.yb
            bivariate_index = (x_class * bins + y_class)

            flat_colors, _ = make_bivariate_palette(cmap_x, cmap_y, bins=bins)
            listed_cmap = create_listed_cmap_from_flat(flat_colors)

            # Step 5: Plot preview
            figure_width = st.sidebar.number_input("Figure Width (inches)", 4.0, 20.0, 7.0, key="fig_w")
            figure_height = st.sidebar.number_input("Figure Height (inches)", 4.0, 20.0, 5.0, key="fig_h")
            fig, ax = plt.subplots(figsize=(figure_width, figure_height))
            bivar_masked = np.ma.masked_where(np.isnan(bivariate_index), bivariate_index)
            ax.imshow(bivar_masked, cmap=listed_cmap, origin="upper", vmin=0, vmax=bins*bins - 1)
            if show_grid:
                ax.grid(True, linewidth=0.3, linestyle="--")
                ax.tick_params(labelsize=grid_fontsize)
            else:
                ax.set_axis_off()

            ax.set_title(map_title, fontsize=13, pad=10)

            legend_ax = inset_axes(ax,
                                   width=f"{legend_size*100}%",
                                   height=f"{legend_size*100}%",
                                   loc=loc_map.get(legend_pos, 1),
                                   borderpad=1.5)
            legend_matrix = np.arange(bins * bins).reshape(bins, bins)
            legend_ax.imshow(legend_matrix, cmap=listed_cmap, origin="lower")
            legend_ax.set_xticks([])
            legend_ax.set_yticks([])
            legend_ax.set_xlabel(x_label, fontsize=9, labelpad=margin_x)
            legend_ax.set_ylabel(y_label, fontsize=9, labelpad=margin_y)

            st.pyplot(fig, use_container_width=False)


            # Step 6: Download options
            with st.sidebar.expander("💾 Download Map", expanded=False):
                dpi = st.slider("DPI (Resolution)", 72, 600, 300, step=30)


                fig_dl, ax_dl = plt.subplots(figsize=(figure_width, figure_height))
                bivar_masked_dl = np.ma.masked_where(np.isnan(bivariate_index), bivariate_index)
                ax_dl.imshow(bivar_masked_dl, cmap=listed_cmap, origin="upper", vmin=0, vmax=bins*bins - 1)

                ax_dl.set_title(map_title, fontsize=13, pad=10)
                if show_grid:
                    ax_dl.grid(True, linewidth=0.3, linestyle="--")
                    ax_dl.tick_params(labelsize=grid_fontsize)
                else:
                    ax_dl.set_axis_off()

                legend_ax = inset_axes(ax_dl,
                                       width=f"{legend_size*100}%",
                                       height=f"{legend_size*100}%",
                                       loc=loc_map.get(legend_pos, 1),
                                       borderpad=1.5)
                legend_matrix = np.arange(bins * bins).reshape(bins, bins)
                legend_ax.imshow(legend_matrix, cmap=listed_cmap, origin="lower")
                legend_ax.set_xticks([])
                legend_ax.set_yticks([])
                legend_ax.set_xlabel(x_label, fontsize=9, labelpad=margin_x)
                legend_ax.set_ylabel(y_label, fontsize=9, labelpad=margin_y)

                fig_dl.tight_layout()
                buf = save_fig_to_buf(fig_dl, dpi=dpi)

                st.download_button("📥 Download Map as PNG", data=buf,
                                   file_name="bivariate_map_raster.png", mime="image/png")

        except Exception as e:
            st.error(f"Error processing raster: {e}")

    else:
        st.info("Upload both Raster 1 and Raster 2 (GeoTIFF) to create a bivariate raster map.")

