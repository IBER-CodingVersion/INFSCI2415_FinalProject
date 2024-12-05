# Import necessary libraries
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import matplotlib.colors as colors
import matplotlib.cm as cmx
from shapely.geometry import LineString
import numpy as np

# -----------------------------
# Part 1: Global Population Density Choropleth Map with Latitude and Longitude Lines and Labels
# -----------------------------

# Step 1: Read the population density data
population_density_df = pd.read_csv('PopulationDensitydata.csv')

# Step 2: Rename '2021' to 'Population Density' for clarity
population_density_df = population_density_df.rename(columns={'2021': 'Population Density'})

# Step 3: Convert 'Population Density' to numeric, coercing errors to NaN
population_density_df['Population Density'] = pd.to_numeric(population_density_df['Population Density'], errors='coerce')

# Step 4: Drop rows with NaN in 'Population Density'
population_density_df = population_density_df.dropna(subset=['Population Density'])

# Step 5: Read the world map data
world = gpd.read_file('ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')

# Step 6: Prepare the world countries DataFrame
world_countries = world[['ADMIN', 'geometry']].rename(columns={'ADMIN': 'Country Name'})

# Step 7: Check unmatched countries
countries_in_population = set(population_density_df['Country Name'])
countries_in_world = set(world_countries['Country Name'])
unmatched_countries = countries_in_population - countries_in_world
print(f"Unmatched countries: {unmatched_countries}")

# Step 8: Replace country names for matching
country_replacements = {
    'Bahamas, The': 'The Bahamas',
    'Brunei Darussalam': 'Brunei',
    'Congo, Dem. Rep.': 'Democratic Republic of the Congo',
    'Congo, Rep.': 'Republic of the Congo',
    'Egypt, Arab Rep.': 'Egypt',
    'Gambia, The': 'The Gambia',
    'Hong Kong SAR, China': 'Hong Kong S.A.R.',
    'Iran, Islamic Rep.': 'Iran',
    'Korea, Dem. People’s Rep.': 'North Korea',
    'Korea, Rep.': 'South Korea',
    'Kyrgyz Republic': 'Kyrgyzstan',
    'Lao PDR': 'Laos',
    'Macedonia, FYR': 'North Macedonia',
    'Russian Federation': 'Russia',
    'Slovak Republic': 'Slovakia',
    'Syrian Arab Republic': 'Syria',
    'Venezuela, RB': 'Venezuela',
    'Vietnam': 'Vietnam',
    'Yemen, Rep.': 'Yemen',
    'United States': 'United States of America',
    'Czech Republic': 'Czechia',
    'Türkiye': 'Turkey',
    'St. Kitts and Nevis': 'Saint Kitts and Nevis',
    'St. Lucia': 'Saint Lucia',
    'St. Vincent and the Grenadines': 'Saint Vincent and the Grenadines',
    'Antigua and Barbuda': 'Antigua and Barbuda',
    'Cabo Verde': 'Cape Verde',
    'Micronesia, Fed. Sts.': 'Federated States of Micronesia',
    'Marshall Islands': 'Marshall Islands',
    'São Tomé and Príncipe': 'Sao Tome and Principe',
    'Eswatini': 'Swaziland',
    'Timor-Leste': 'East Timor',
    'Kosovo': 'Kosovo'  # Added Kosovo
}

population_density_df['Country Name'] = population_density_df['Country Name'].replace(country_replacements)

# Step 9: Check unmatched countries again
unmatched_countries_after = set(population_density_df['Country Name']) - set(world_countries['Country Name'])
print(f"Still unmatched countries: {unmatched_countries_after}")
merged_data = population_density_df[~population_density_df['Country Name'].isin(unmatched_countries_after)]

# Step 10: Merge population density data with world map data
merged = world_countries.merge(merged_data, on='Country Name', how='left')

# Step 11: Create Latitude and Longitude Lines

# Define desired latitudes and longitudes
latitudes = list(range(-60, 61, 30))  # -60, -30, 0, 30, 60
longitudes = list(range(-180, 181, 60))  # -180, -120, -60, 0, 60, 120, 180

# Create LineStrings for latitude lines
latitude_lines = [LineString([(-180, lat), (180, lat)]) for lat in latitudes]

# Create LineStrings for longitude lines
longitude_lines = [LineString([(lon, -90), (lon, 90)]) for lon in longitudes]

# Combine all lines
all_lines = latitude_lines + longitude_lines

# Create a GeoDataFrame for the lines
lines_gdf = gpd.GeoDataFrame({'geometry': all_lines}, crs='EPSG:4326')

# Step 12: Plot the choropleth map with Latitude and Longitude Lines and Labels
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Plot the population density choropleth map
merged_plot = merged.plot(
    column='Population Density',
    cmap='YlGnBu',  
    linewidth=0.8,
    ax=ax,
    edgecolor='0.8',
    legend=False,  
    missing_kwds={
        "color": "lightgrey",
        "edgecolor": "red",
        "hatch": "///",
        "label": "No data",
    }
)

# Plot the latitude and longitude lines
lines_gdf.plot(
    ax=ax,
    color='black',
    linewidth=0.5,
    linestyle='--',
    alpha=0.7
)

# Step 13: Add Latitude and Longitude Labels

# Function to add latitude labels
def add_latitude_labels(ax, latitudes):
    for lat in latitudes:
        # Place the label at longitude -180 (left edge)
        ax.text(-180, lat, f'{lat}°', verticalalignment='bottom' if lat > 0 else 'top',
                horizontalalignment='left', fontsize=8, color='black',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

# Function to add longitude labels
def add_longitude_labels(ax, longitudes):
    for lon in longitudes:
        # Place the label at latitude 90 (top edge)
        ax.text(lon, 90, f'{lon}°', horizontalalignment='center', verticalalignment='bottom',
                fontsize=8, color='black',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
        # Place the label at latitude -90 (bottom edge)
        ax.text(lon, -90, f'{lon}°', horizontalalignment='center', verticalalignment='top',
                fontsize=8, color='black',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

# Add latitude labels
add_latitude_labels(ax, latitudes)

# Add longitude labels
add_longitude_labels(ax, longitudes)

# Add title
latest_year = '2021'
ax.set_title(f'Global Population Density in {latest_year}', fontdict={'fontsize': 20}, pad=20)

# Remove axis
ax.set_axis_off()

# Create a colorbar as a legend
cmin = merged['Population Density'].min()
cmax = merged['Population Density'].max()
norm = Normalize(vmin=cmin, vmax=cmax)

cmap = plt.get_cmap('YlGnBu')
sm = cmx.ScalarMappable(norm=norm, cmap=cmap)
sm._A = []

cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label('Population Density (people per sq. km)', rotation=270, labelpad=15, fontsize=12)

# Format the colorbar tick labels
def format_population_density(x, pos):
    if x >= 1e3:
        return f'{x*1e-3:.0f}K'
    elif x >= 1:
        return f'{x:.0f}'
    else:
        return f'{x:.1f}'

cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(format_population_density))

# Add legend for missing data
missing_patch = mpatches.Patch(facecolor='lightgrey', hatch='///', edgecolor='red', label='No data')
ax.legend(handles=[missing_patch], loc='lower left', fontsize=12)

# Display the choropleth map
plt.show()

# -----------------------------
# Part 2: Global Population Density Scatter Plot with Log Transformation
# -----------------------------

# Step 14: Calculate centroids to get longitude and latitude
merged['centroid'] = merged['geometry'].representative_point()
merged['Longitude'] = merged['centroid'].x
merged['Latitude'] = merged['centroid'].y

# Step 15: Apply logarithmic transformation to Population Density
merged['Log Population Density'] = np.log10(merged['Population Density'].replace(0, np.nan))
merged = merged.dropna(subset=['Log Population Density'])

# Step 16: Plotting the Population Density Scatter Plot with Log Transformation
plt.figure(figsize=(15, 10))

# Scatter plot with Longitude and Latitude as axes, Log Population Density as circle size and color
scatter = plt.scatter(
    merged['Longitude'],
    merged['Latitude'],
    s=merged['Log Population Density'] * 50,
    alpha=0.6,
    c=merged['Log Population Density'],
    cmap='YlGnBu',
    edgecolors='k'
)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Log Population Density (log10 people per sq. km)', fontsize=12)

# Add title and labels
plt.title(f'Global Population Density Scatter Plot in {latest_year} (Log Scale)', fontsize=20)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)

# Add grid
plt.grid(True, linestyle='--', alpha=0.5)

# Set limits for better visualization
plt.xlim(-180, 180)
plt.ylim(-90, 90)

# Show plot
plt.show()
