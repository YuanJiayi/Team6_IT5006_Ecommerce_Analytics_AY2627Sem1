import copy
import json
import math

import folium
import pandas as pd
from branca.colormap import linear
from branca.element import MacroElement
from jinja2 import Template


class _StateLegendToggle(MacroElement):
    _template = Template(
        """
        {% macro script(this, kwargs) %}
        const legend = document.querySelector('.legend.leaflet-control');
        if (legend) legend.style.display = 'none';
        {{ this._parent.get_name() }}.on('overlayadd overlayremove', function (event) {
            if (event.name !== 'State order density') return;
            const stateLegend = document.querySelector('.legend.leaflet-control');
            if (stateLegend) stateLegend.style.display = event.type === 'overlayadd' ? '' : 'none';
        });
        {% endmacro %}
        """
    )

    def __init__(self, legend_name):
        super().__init__()
        self.legend_name = legend_name


def add_state_order_density_layer(map_object, data_directory="data"):
    orders = pd.read_csv(
        f"{data_directory}/smartcommerce_consolidated.csv",
        usecols=["order_id", "customer_state"]
    )
    with open(f"{data_directory}/brazil_states.geojson", encoding="utf-8") as states_file:
        states = json.load(states_file)

    orders["state_code"] = orders["customer_state"].astype(str).str.strip().str.upper()
    counts = orders.groupby("state_code")["order_id"].nunique()
    state_values = [math.log1p(int(value)) for value in counts.values]
    color_scale = linear.YlOrRd_09.scale(min(state_values), max(state_values))
    color_scale.caption = "Log-scaled orders by state"

    states_with_counts = copy.deepcopy(states)
    for feature in states_with_counts["features"]:
        state_code = feature["properties"].get("sigla", "").upper()
        feature["properties"]["order_count"] = int(counts.get(state_code, 0))

    def style_function(feature):
        order_count = feature["properties"]["order_count"]
        return {
            "fillColor": color_scale(math.log1p(order_count)) if order_count else "#f2f2f2",
            "color": "#555555",
            "weight": 1,
            "fillOpacity": 0.55
        }

    state_layer_group = folium.FeatureGroup(name="State order density", show=False)
    state_layer_group.add_to(map_object)
    state_layer = folium.GeoJson(
        states_with_counts,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["name", "sigla", "order_count"],
            aliases=["State", "Code", "Orders"]
        )
    )
    state_layer.add_to(state_layer_group)
    color_scale.add_to(map_object)

    _StateLegendToggle(color_scale.get_name()).add_to(map_object)