import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# 读取Excel文件，假设文件名为 data.xlsx
df = pd.read_excel('info.xlsx')

# 构造几何点
geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

# 载入世界地图底图
world = gpd.read_file('./ne_110m_admin_0_countries.shp')

# 绘图
fig, ax = plt.subplots(figsize=(12,8))
world.plot(ax=ax, color='lightgray', edgecolor='white')

scatter = gdf.plot(
    ax=ax,
    column='altitudey',
    cmap='viridis',
    markersize=50,
    alpha=0.7,
    legend=True
)

for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf['Station']):
    ax.text(x, y, label, fontsize=8, ha='right')

ax.set_title('Geographic Distribution of Stations with Altitude')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.savefig('map.svg', format='svg', bbox_inches='tight')
plt.show()
