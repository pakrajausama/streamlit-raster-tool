import os
import zipfile
import uuid
import geopandas as gpd
import tempfile
import streamlit as st



# --- Complete UI Cleanup ---

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


# --- Helper Functions ---

def extract_kmz(uploaded_file, temp_dir: str) -> str:
    """Extract KMZ to temp dir and return KML path."""
    kmz_path = os.path.join(temp_dir, uploaded_file.name)
    with open(kmz_path, 'wb') as f:
        f.write(uploaded_file.read())

    with zipfile.ZipFile(kmz_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    for root, _, files in os.walk(temp_dir):
        for f in files:
            if f.lower().endswith('.kml'):
                return os.path.join(root, f)
    raise Exception("No KML found in KMZ.")


def extract_zip_shapefile(uploaded_file, temp_dir: str) -> str:
    """Extract ZIP containing shapefile and return .shp path."""
    zip_path = os.path.join(temp_dir, uploaded_file.name)
    with open(zip_path, 'wb') as f:
        f.write(uploaded_file.read())

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    for root, _, files in os.walk(temp_dir):
        for f in files:
            if f.lower().endswith('.shp'):
                return os.path.join(root, f)
    raise Exception("No .shp found in ZIP. Include .shp, .shx, .dbf, .prj, .cpg.")


def convert_single_file(uploaded_file, output_path, output_format):
    """Convert one uploaded file to requested format."""
    file_temp_dir = tempfile.mkdtemp()

    # Detect input format
    if uploaded_file.name.lower().endswith('.kmz'):
        input_path = extract_kmz(uploaded_file, file_temp_dir)
    elif uploaded_file.name.lower().endswith('.zip'):
        input_path = extract_zip_shapefile(uploaded_file, file_temp_dir)
    elif uploaded_file.name.lower().endswith('.shp'):
        raise Exception("Upload shapefiles as a ZIP with all related files.")
    else:
        input_path = os.path.join(file_temp_dir, uploaded_file.name)
        with open(input_path, 'wb') as f:
            f.write(uploaded_file.read())

    gdf = gpd.read_file(input_path)
    if gdf.empty:
        raise Exception(f"{uploaded_file.name} contains no features.")

    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)

    output_ext = output_format.lower()

    if output_ext == 'geojson':
        gdf.to_file(output_path, driver='GeoJSON', index=False)
        return output_path

    elif output_ext == 'kml':
        gdf.to_file(output_path, driver='KML', index=False)
        return output_path

    elif output_ext == 'shapefile':
        shp_dir = os.path.splitext(output_path)[0] + "_shp"
        os.makedirs(shp_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(output_path))[0].replace('.shp', '')
        shp_full_path = os.path.join(shp_dir, base_name + '.shp')

        gdf = gdf[gdf.geometry.notnull()].copy()
        gdf = gdf[gdf.is_valid]

        if gdf.empty:
            raise Exception("No valid geometries. Cannot write shapefile.")

        gdf.to_file(shp_full_path, driver='ESRI Shapefile', index=False)

        # Zip shapefile components
        zip_path = os.path.join(os.path.dirname(output_path), f"{base_name}_shapefile.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for f in os.listdir(shp_dir):
                file_path = os.path.join(shp_dir, f)
                zipf.write(file_path, arcname=f)
        return zip_path

    else:
        raise Exception("Unsupported output format.")


# --- Streamlit App ---
st.title("🌍 Geospatial File Converter")

uploaded_files = st.file_uploader(
    "Upload geospatial files (.zip with shapefile, .kml, .kmz, .geojson)",
    type=["zip", "kml", "kmz", "geojson"],
    accept_multiple_files=True
)

output_format = st.selectbox(
    "Select output format:",
    ["GeoJSON", "KML", "Shapefile"]
)

if uploaded_files and st.button("Convert"):
    st.write("Processing files... ⏳")
    results = []
    for uploaded_file in uploaded_files:
        try:
            # Generate safe unique filename
            base_name = os.path.splitext(uploaded_file.name)[0]
            safe_base_name = "".join(c for c in base_name if c.isalnum() or c in ('_', '-'))
            unique_suffix = uuid.uuid4().hex[:6]

            if output_format.lower() == 'shapefile':
                out_filename = f"{safe_base_name}_{unique_suffix}.shp"
            else:
                out_filename = f"{safe_base_name}_{unique_suffix}.{output_format.lower()}"

            output_path = os.path.join(tempfile.mkdtemp(), out_filename)
            result_path = convert_single_file(uploaded_file, output_path, output_format)
            results.append(result_path)

        except Exception as e:
            st.error(f"❌ {uploaded_file.name}: {str(e)}")

    # Provide download links
    if results:
        st.success("✅ Conversion completed!")
        for res in results:
            with open(res, "rb") as f:
                st.download_button(
                    label=f"⬇️ Download {os.path.basename(res)}",
                    data=f,
                    file_name=os.path.basename(res),
                )
