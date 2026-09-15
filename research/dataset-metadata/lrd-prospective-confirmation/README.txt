Landslide Reference Data (Sentinel-1 & Sentinel-2)

This folder contains the data used for landslide detection and analysis. The directory is organized into three main subfolders:

1. original_scenes/
   - Contains the original georeferenced input data for each landslide event.
   - Includes Sentinel-1 (SAR), Sentinel-2 (optical), and Copernicus DEM features.
   - Data is organized per event, with all relevant raster features included.

2. reference_data/
   - Contains pre-processed and labeled data patches in .h5 format.
   - Split into three subsets: train/, val/, and test/.
   - Each patch is 125x125 pixels and includes multiple feature layers.

3. vector_data/
   - Contains vector-format landslide inventory in GeoPackage format.
   - Includes polygon features with attribute data (e.g., event ID, date, location, source).

This structure supports machine learning workflows, geospatial analysis, and manual inspection.

References:
Orynbaikyzy, A., Albrecht, F., Yao, W., Motagh, M., Wang, W., Martinis, S., & Plank, S. (2025). Landslide mapping with deep learning: the role of pre-/post-event SAR features and multi-sensor data fusion. GIScience & Remote Sensing, 62(1). https://doi.org/10.1080/15481603.2025.2502214
