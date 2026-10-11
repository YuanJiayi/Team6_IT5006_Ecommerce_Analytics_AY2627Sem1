import json

import folium
from branca.element import Element
from map_layers.order_density import add_order_density_layer #Add new layers to map like this
from map_layers.seller_density import add_seller_density_layer
from map_layers.seller_buyer_routes import add_seller_buyer_routes_layer
from map_layers.state_order_density import add_state_order_density_layer

with open("data/brazil_boundary.geojson", encoding="utf-8") as boundary_file:
    brazil_boundary = json.load(boundary_file)
with open("data/brazil_states.geojson", encoding="utf-8") as states_file:
    brazil_states = json.load(states_file)

# Keep the initial view and panning focused on Brazil.
brazil_bounds = [[-33.75, -73.99], [5.27, -34.79]]
brazil_map = folium.Map(
    location=[-14.2, -51.9],
    zoom_start=4.5,
    min_zoom=4,
    max_bounds=True,
    tiles=None,
    control_scale=True
)

# Base-map examples can be compared from the layer control in the top-right.
tile_layers = [
    {
        "name": "OpenTopoMap (terrain)",
        "tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "attr": "Map data &copy; OpenStreetMap contributors, SRTM | Map style &copy; OpenTopoMap",
        "control": True,
        "show": False,
    },
    {
        "name": "Esri World Street Map",
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        "attr": "Tiles &copy; Esri",
        "control": True,
        "show": True,
    },
    {
        "name": "Esri World Imagery (satellite)",
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Tiles &copy; Esri",
        "control": True,
        "show": False,
    },
]

for tile_layer in tile_layers:
    folium.TileLayer(**tile_layer).add_to(brazil_map)

folium.GeoJson(
    brazil_boundary,
    control=False,
    style_function=lambda feature: {
        "color": "#4a4a4a",
        "weight": 1.5,
        "fillColor": "#f5f5f5",
        "fillOpacity": 0.08
    }
).add_to(brazil_map)

folium.GeoJson(
    brazil_states,
    control=False,
    style_function=lambda feature: {
        "color": "#374151",
        "weight": 1.1,
        "fillOpacity": 0
    },
    tooltip=folium.GeoJsonTooltip(fields=["name", "sigla"], aliases=["State", "Code"])
).add_to(brazil_map)

# Calculated metrics are an independent overlay layer.
add_order_density_layer(brazil_map)
add_seller_density_layer(brazil_map)
add_seller_buyer_routes_layer(brazil_map)
add_state_order_density_layer(brazil_map)

folium.LayerControl(collapsed=False).add_to(brazil_map)

map_style = """
<style>
.leaflet-control-layers {
    background: rgba(255, 255, 255, 0.97);
    border: 1px solid #d7dee8;
    border-radius: 10px;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.18);
    color: #1f2937;
    font: 13px/1.35 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    padding: 7px 9px;
}
.leaflet-control-layers-list {
    margin: 2px 0;
}
.leaflet-control-layers-base label,
.leaflet-control-layers-overlays label {
    border-radius: 6px;
    display: flex;
    gap: 7px;
    margin: 1px 0;
    padding: 5px 6px;
}
.leaflet-control-layers-base label:hover,
.leaflet-control-layers-overlays label:hover {
    background: #eef4fb;
}
.leaflet-control-layers input {
    accent-color: #2563a6;
    margin: 2px 0 0;
}
</style>
"""
brazil_map.get_root().header.add_child(Element(map_style))

# Fit the map to Brazil rather than only the order locations.
brazil_map.fit_bounds(brazil_bounds)

brazil_map.save("brazil_order_density_map.html")
print("Saved map to brazil_order_density_map.html")


# The Streamlit can probably use the Folium object directly rather than rely on the html.
# We can keep adding layers to this same map so thats nice lol