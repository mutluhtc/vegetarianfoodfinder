import streamlit as st
import json
import pandas as pd
import pydeck as pdk
import requests
import math
import os
import glob

st.set_page_config(page_title="VegFinder | Discover Vegetarian-Friendly Restaurants", layout="wide")

# --- Helper: Haversine distance (in miles) ---
def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# --- Cuisine mapping ---
CUISINE_MAP = {
    "american": "American", "new american": "American", "burgers": "American", "bbq": "American",
    "thai": "Thai",
    "chinese": "Chinese", "szechuan": "Chinese",
    "mediterranean": "Mediterranean", "greek": "Mediterranean", "middle eastern": "Mediterranean",
    "lebanese": "Mediterranean", "turkish": "Mediterranean",
    "indian": "Indian", "pakistani": "Indian", "nepalese": "Indian",
    "italian": "Italian", "pizza": "Italian", "pasta": "Italian",
    "mexican": "Mexican", "latin": "Mexican", "taco": "Mexican",
    "vegan": "Vegan/Vegetarian", "vegetarian": "Vegan/Vegetarian", "plant based": "Vegan/Vegetarian"
}
DEFAULT_CATEGORY = "Vegan/Vegetarian"

def normalize_cuisine(categories_str):
    """Map detailed Yelp categories to one of the 8 main cuisines."""
    if not categories_str or not isinstance(categories_str, str):
        return DEFAULT_CATEGORY
    categories = [c.strip().lower() for c in categories_str.split(",")]
    for c in categories:
        for key, mapped in CUISINE_MAP.items():
            if key in c:
                return mapped
    return DEFAULT_CATEGORY


# --- Load all JSONs dynamically ---
@st.cache_data
def load_all_jsons(folder_path):
    """Load and merge all JSON files from a folder into one list."""
    all_data = []
    json_files = glob.glob(os.path.join(folder_path, "*.json"))

    for file in json_files:
        try:
            with open(file, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
        except Exception as e:
            st.warning(f"⚠️ Error reading {file}: {e}")

    valid_data = [r for r in all_data if r.get("menu") and len(r["menu"]) > 0]
    for r in valid_data:
        r["normalized_cuisine"] = normalize_cuisine(r.get("categories", ""))
        if r.get("location"):
            r["city"] = r["location"].split(",")[0].strip()
        else:
            r["city"] = "San Diego"

    st.success(f"✅ Loaded {len(json_files)} JSON files — {len(valid_data)} valid restaurant entries")
    return valid_data


# --- Geocode ZIP code using OpenStreetMap Nominatim ---
def get_coordinates_from_zip(zip_code):
    """Return latitude and longitude for a given ZIP code."""
    try:
        url = f"https://nominatim.openstreetmap.org/search?postalcode={zip_code}&country=USA&format=json"
        response = requests.get(url, headers={"User-Agent": "VegFinderApp"}, timeout=10)
        data = response.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        st.warning(f"Could not locate ZIP code: {e}")
    return None, None


# --- Load data (folder with all 28 JSONs) ---
data = load_all_jsons("data_jsons")   # <-- your folder containing the JSONs

# --- Header ---
st.title("🥗 Veggie Food Finder — Your Personalized Vegetarian Food Map")
st.markdown("""
### Discover Vegetarian-Friendly Restaurants Near You 🌎  
Enter your **ZIP code** to see nearby restaurants, filter by **cuisine** or **diet type**,  
and explore detailed **menus and locations** on an interactive map.
""")

# --- Sidebar filters ---
st.sidebar.header("Filter Options")

# ZIP code input
zip_code = st.sidebar.text_input("Enter your ZIP code:")
user_lat, user_lon = (None, None)
if zip_code:
    user_lat, user_lon = get_coordinates_from_zip(zip_code)
    if user_lat and user_lon:
        st.sidebar.success(f"📍 Located at ({round(user_lat, 4)}, {round(user_lon, 4)})")
    else:
        st.sidebar.warning("Could not find that ZIP code.")

# Distance filter
distance_filter = st.sidebar.selectbox(
    "Show restaurants within:",
    ["All", 5, 10, 20, 50],
    help="Distance is measured from your ZIP code (in miles)."
)

# City and cuisine filters
all_cities = sorted(set(r["city"] for r in data))
selected_city = st.sidebar.selectbox("Select City", ["All"] + all_cities)

CATEGORIES = ["All", "American", "Thai", "Chinese", "Mediterranean", "Indian", "Italian", "Mexican", "Vegan/Vegetarian"]
selected_cuisine = st.sidebar.selectbox("Filter by Cuisine", CATEGORIES)
veg_filter = st.sidebar.selectbox("Show Dishes", ["All", "Vegetarian", "Non-Vegetarian"])

# --- Filtering logic ---
filtered_data = []
for r in data:
    city_match = (selected_city == "All") or (r["city"] == selected_city)
    cuisine_match = (selected_cuisine == "All") or (r["normalized_cuisine"] == selected_cuisine)
    if not (city_match and cuisine_match):
        continue

    dishes = r["menu"]
    if veg_filter != "All":
        dishes = [d for d in dishes if d.get("vegetarian_label") == veg_filter]

    if dishes:
        r_copy = r.copy()
        r_copy["menu"] = dishes

        if user_lat and user_lon and r.get("latitude") and r.get("longitude"):
            r_copy["distance_miles"] = round(haversine(user_lat, user_lon, r["latitude"], r["longitude"]), 2)
        else:
            r_copy["distance_miles"] = None

        filtered_data.append(r_copy)

# --- Apply distance filter ---
if user_lat and user_lon and distance_filter != "All":
    filtered_data = [r for r in filtered_data if r["distance_miles"] and r["distance_miles"] <= distance_filter]

# --- Sort by distance then rating ---
filtered_data = sorted(
    filtered_data,
    key=lambda x: (x["distance_miles"] if x["distance_miles"] is not None else 9999, -x.get("rating", 0))
)

st.write(f"### Showing {len(filtered_data)} restaurants that match your filters")

# --- Map visualization ---
if filtered_data:
    map_df = pd.DataFrame([
        {
            "Restaurant": r["restaurant_name"],
            "City": r["city"],
            "Cuisine": r["normalized_cuisine"],
            "Rating": r.get("rating", None),
            "Distance (mi)": r.get("distance_miles", None),
            "lat": r.get("latitude", None),
            "lon": r.get("longitude", None),
        }
        for r in filtered_data if r.get("latitude") and r.get("longitude")
    ])

    st.subheader("🗺️ Interactive Map of Restaurants")
    st.caption("Hover or click on any point to view restaurant details.")

    avg_lat = user_lat if user_lat else map_df["lat"].mean()
    avg_lon = user_lon if user_lon else map_df["lon"].mean()

    view_state = pdk.ViewState(latitude=avg_lat, longitude=avg_lon, zoom=10, pitch=0)

    restaurant_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["lon", "lat"],
        get_fill_color=[255, 100, 100, 180],
        get_radius=300,
        pickable=True,
    )

    layers = [restaurant_layer]

    # Add user marker
    if user_lat and user_lon:
        user_df = pd.DataFrame([{"lat": user_lat, "lon": user_lon}])
        user_layer = pdk.Layer(
            "ScatterplotLayer",
            data=user_df,
            get_position=["lon", "lat"],
            get_fill_color=[0, 150, 255, 255],
            get_radius=800,
            pickable=False,
        )
        layers.append(user_layer)

    tooltip = {
        "html": "<b>{Restaurant}</b><br/>Cuisine: {Cuisine}<br/>Rating: ⭐ {Rating}<br/>Distance: {Distance (mi)} miles",
        "style": {"backgroundColor": "steelblue", "color": "white"},
    }

    rmap = pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=tooltip)
    st.pydeck_chart(rmap)

# --- Restaurant list ---
if not filtered_data:
    st.info("No restaurants match your filters.")
else:
    for r in filtered_data:
        st.markdown("---")
        st.subheader(f"{r['restaurant_name']} ⭐ {r.get('rating', 'N/A')}")
        if r.get("distance_miles"):
            st.write(f"📍 **Distance:** {r['distance_miles']} miles")
        st.write(f"**City:** {r['city']}")
        st.write(f"**Cuisine:** {r['normalized_cuisine']}")
        st.write(f"**Address:** {r.get('address', 'N/A')}")

        with st.expander("Show Menu Items"):
            dishes = pd.DataFrame(r["menu"])
            if not dishes.empty:
                dishes_display = dishes[["item_name", "price", "description", "vegetarian_label"]].copy()
                dishes_display["price"] = dishes_display["price"].apply(
                    lambda x: f"${int(x)}" if pd.notnull(x) and isinstance(x, (int, float)) else x
                )
                dishes_display.columns = ["Dish Name", "Price", "Description", "Vegetarian Label"]
                st.dataframe(dishes_display, use_container_width=True)
