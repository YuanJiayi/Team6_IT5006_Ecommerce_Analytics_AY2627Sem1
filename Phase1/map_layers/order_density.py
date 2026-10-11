import pandas as pd
import folium
from folium.plugins import HeatMap


GRADIENT = {
    0.00: "#d9f0ff",
    0.10: "#66c2ff",
    0.25: "#20b2aa",
    0.40: "#ffd166",
    0.60: "#f08a24",
    0.80: "#d62828",
    1.00: "#8b0000"
}


def _calculate_order_locations(orders, customers, geo):
    orders = orders.copy()
    customers = customers.copy()
    geo = geo.copy()

    orders.columns = orders.columns.str.strip().str.lower()
    customers.columns = customers.columns.str.strip().str.lower()
    geo.columns = geo.columns.str.strip().str.lower()

    orders = orders.merge(
        customers[["customer_id", "customer_zip_code_prefix"]],
        on="customer_id",
        how="left"
    )
    orders["zip_prefix"] = (
        orders["customer_zip_code_prefix"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .str.zfill(5)
        .str[:5]
    )
    geo["zip_prefix"] = (
        geo["geolocation_zip_code_prefix"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .str.zfill(5)
        .str[:5]
    )

    geo_by_zip = (
        geo.groupby("zip_prefix", as_index=False)
           .agg(latitude=("geolocation_lat", "mean"),
                longitude=("geolocation_lng", "mean"))
    )
    locations = (
        orders.groupby("zip_prefix")
              .size()
              .rename("order_count")
              .reset_index()
              .merge(geo_by_zip, on="zip_prefix", how="inner")
              .dropna(subset=["latitude", "longitude"])
    )
    locations = locations[
        locations["latitude"].between(-35, 6) &
        locations["longitude"].between(-75, -30)
    ].copy()
    locations["heat_intensity"] = locations["order_count"].rank(pct=True)
    return locations


def add_order_density_layer(map_object, data_directory="data"):
    orders = pd.read_csv(f"{data_directory}/smartcommerce_consolidated.csv", low_memory=False)
    customers = pd.read_csv(f"{data_directory}/olist_customers_dataset.csv", low_memory=False)
    geo = pd.read_csv(f"{data_directory}/olist_geolocation_dataset.csv", low_memory=False)
    locations = _calculate_order_locations(orders, customers, geo)

    density_layer = folium.FeatureGroup(name="Order density", show=True)
    density_layer.add_to(map_object)
    HeatMap(
        locations[["latitude", "longitude", "heat_intensity"]].values.tolist(),
        radius=10,
        blur=14,
        min_opacity=0.25,
        max_zoom=12,
        gradient=GRADIENT,
        use_local_extrema=False
    ).add_to(density_layer)