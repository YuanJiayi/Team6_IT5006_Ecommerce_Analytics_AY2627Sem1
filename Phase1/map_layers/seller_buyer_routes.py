import json
import math
import re
import unicodedata

import folium
import pandas as pd


MIN_ROUTE_ORDERS = 50


def _city_key(city, state):
    value = f"{city} {state}".casefold()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(character for character in value if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def _city_coordinates(geo):
    geo = geo.copy()
    geo.columns = geo.columns.str.strip().str.lower()
    geo["city_key"] = [_city_key(city, state) for city, state in zip(
        geo["geolocation_city"], geo["geolocation_state"]
    )]
    return geo.groupby("city_key", as_index=False).agg(
        city=("geolocation_city", "first"),
        state=("geolocation_state", "first"),
        latitude=("geolocation_lat", "mean"),
        longitude=("geolocation_lng", "mean")
    )


def _calculate_routes(orders, customers, sellers, geo):
    orders = orders[["order_id", "customer_id", "seller_id"]].copy()
    customers = customers[["customer_id", "customer_city", "customer_state"]].copy()
    sellers = sellers[["seller_id", "seller_city", "seller_state"]].copy()
    orders.columns = orders.columns.str.strip().str.lower()
    customers.columns = customers.columns.str.strip().str.lower()
    sellers.columns = sellers.columns.str.strip().str.lower()

    routes = (
        orders.merge(customers, on="customer_id", how="left")
              .merge(sellers, on="seller_id", how="left")
              .rename(columns={
                  "customer_city": "buyer_city",
                  "customer_state": "buyer_state",
                  "seller_city": "seller_city",
                  "seller_state": "seller_state"
              })
    )
    routes["buyer_city_key"] = [
        _city_key(city, state)
        for city, state in zip(routes["buyer_city"], routes["buyer_state"])
    ]
    routes["seller_city_key"] = [
        _city_key(city, state)
        for city, state in zip(routes["seller_city"], routes["seller_state"])
    ]

    routes = (
        routes.groupby(["seller_city_key", "buyer_city_key"])
              .agg(
                  route_orders=("order_id", "nunique"),
                  seller_city=("seller_city", "first"),
                  buyer_city=("buyer_city", "first")
              )
              .reset_index()
    )
    coordinates = _city_coordinates(geo)
    routes = (
        routes.merge(
            coordinates.rename(columns={
                "city_key": "seller_city_key",
                "latitude": "seller_latitude",
                "longitude": "seller_longitude",
                "city": "seller_geo_city"
            }),
            on="seller_city_key",
            how="inner"
        )
        .merge(
            coordinates.rename(columns={
                "city_key": "buyer_city_key",
                "latitude": "buyer_latitude",
                "longitude": "buyer_longitude",
                "city": "buyer_geo_city"
            }),
            on="buyer_city_key",
            how="inner"
        )
    )
    routes = routes[
        routes["seller_latitude"].between(-35, 6) &
        routes["seller_longitude"].between(-75, -30) &
        routes["buyer_latitude"].between(-35, 6) &
        routes["buyer_longitude"].between(-75, -30)
    ].copy()
    # One-off routes create noise; retain recurring routes for a readable flow layer.
    routes = routes[routes["route_orders"] >= MIN_ROUTE_ORDERS]
    return routes


def add_seller_buyer_routes_layer(map_object, data_directory="data"):
    orders = pd.read_csv(
        f"{data_directory}/smartcommerce_consolidated.csv",
        usecols=["order_id", "customer_id", "seller_id"]
    )
    customers = pd.read_csv(f"{data_directory}/olist_customers_dataset.csv", low_memory=False)
    sellers = pd.read_csv(f"{data_directory}/olist_sellers_dataset.csv", low_memory=False)
    geo = pd.read_csv(f"{data_directory}/olist_geolocation_dataset.csv", low_memory=False)
    routes = _calculate_routes(orders, customers, sellers, geo)

    max_route_weight = max(math.log1p(routes["route_orders"].max()), 1)
    features = []
    for route in routes.itertuples(index=False):
        intensity = math.log1p(route.route_orders) / max_route_weight
        features.append({
            "type": "Feature",
            "properties": {
                "seller_city": route.seller_city,
                "buyer_city": route.buyer_city,
                "route_orders": int(route.route_orders),
                "route_weight": 0.5 + 1.8 * intensity,
                "route_opacity": 0.10 + 0.28 * intensity
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [route.seller_longitude, route.seller_latitude],
                    [route.buyer_longitude, route.buyer_latitude]
                ]
            }
        })

    route_data = {"type": "FeatureCollection", "features": features}
    route_layer = folium.FeatureGroup(name="Seller-buyer routes", show=False)
    route_layer.add_to(map_object)
    folium.GeoJson(
        route_data,
        style_function=lambda feature: {
            "color": "#2563a6",
            "weight": feature["properties"]["route_weight"],
            "opacity": feature["properties"]["route_opacity"],
            "lineCap": "round",
            "lineJoin": "round"
        },
        highlight_function=lambda feature: {
            "color": "#f59e0b",
            "weight": feature["properties"]["route_weight"] + 2,
            "opacity": 0.95,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["seller_city", "buyer_city", "route_orders"],
            aliases=["Seller city", "Buyer city", "Orders on route"]
        )
    ).add_to(route_layer)