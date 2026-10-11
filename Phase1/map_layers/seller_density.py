import pandas as pd
import folium
from folium.plugins import HeatMap


SELLER_GRADIENT = {
    0.00: "#e9d5ff",
    0.15: "#c084fc",
    0.35: "#818cf8",
    0.55: "#2563eb",
    0.75: "#0f766e",
    1.00: "#064e3b"
}


def _calculate_seller_locations(sellers, geo):
    sellers = sellers.copy()
    geo = geo.copy()
    sellers.columns = sellers.columns.str.strip().str.lower()
    geo.columns = geo.columns.str.strip().str.lower()

    sellers["zip_prefix"] = (
        sellers["seller_zip_code_prefix"]
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
        sellers.groupby("zip_prefix")["seller_id"]
               .nunique()
               .rename("seller_count")
               .reset_index()
               .merge(geo_by_zip, on="zip_prefix", how="inner")
               .dropna(subset=["latitude", "longitude"])
    )
    locations = locations[
        locations["latitude"].between(-35, 6) &
        locations["longitude"].between(-75, -30)
    ].copy()
    locations["heat_intensity"] = locations["seller_count"].rank(pct=True)
    return locations


def add_seller_density_layer(map_object, data_directory="data"):
    sellers = pd.read_csv(f"{data_directory}/olist_sellers_dataset.csv", low_memory=False)
    geo = pd.read_csv(f"{data_directory}/olist_geolocation_dataset.csv", low_memory=False)
    locations = _calculate_seller_locations(sellers, geo)

    density_layer = folium.FeatureGroup(name="Seller density", show=False)
    density_layer.add_to(map_object)
    HeatMap(
        locations[["latitude", "longitude", "heat_intensity"]].values.tolist(),
        radius=10,
        blur=14,
        min_opacity=0.25,
        max_zoom=12,
        gradient=SELLER_GRADIENT,
        use_local_extrema=False
    ).add_to(density_layer)