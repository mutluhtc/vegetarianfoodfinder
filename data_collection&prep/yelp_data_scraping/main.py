import pandas as pd
import os
import time
from yelp_api import fetch_restaurants, fetch_business_details, deduplicate_restaurants
from scraper import get_website_text
import argparse

CATEGORIES = ["italian", "mexican", "indian", "chinese", "thai",
              "american", "mediterranean", "vegan", "vegetarian"]

def scrape_city(city="San Diego, CA", max_per_category=200, sample_only=False):
    all_restaurants = []

    # Step 1: Fetch restaurants by category
    for cat in CATEGORIES:
        results = fetch_restaurants(location=city, term=f"restaurants,{cat}", max_fetch=max_per_category)
        all_restaurants.extend(results)

    # Deduplicate
    restaurants = deduplicate_restaurants(all_restaurants)
    print(f"Total unique restaurants fetched: {len(restaurants)}")

    # Step 2: Fetch business details
    all_data = []
    sample_restaurants = restaurants[:5] if sample_only else restaurants
    for i, biz in enumerate(sample_restaurants):
        details = fetch_business_details(biz['id'])
        if not details:
            continue

        website_url = details.get("attributes", {}).get("menu_url", details.get("url"))

        all_data.append({
            "id": details.get("id"),
            "name": details.get("name"),
            "rating": details.get("rating"),
            "review_count": details.get("review_count"),
            "categories": ", ".join([c['title'] for c in details.get("categories", [])]),
            "price": details.get("price", ""),
            "address": ", ".join(details["location"].get("display_address", [])),
            "latitude": details["coordinates"].get("latitude"),
            "longitude": details["coordinates"].get("longitude"),
            "url": details.get("url"),
            "website_url": website_url
        })

        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{len(sample_restaurants)} restaurants")
        time.sleep(0.5)

    df = pd.DataFrame(all_data)
    df["city"] = city

    # Step 3: Scrape website text
    print("Fetching website/menu text...")
    df["raw_text"] = df["website_url"].apply(get_website_text)

   # Ensure data folder exists
    os.makedirs("data", exist_ok=True)
    filename = f"data/{city.replace(' ', '_').lower()}_restaurants_full_with_rawtext.csv"
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape restaurants for a city.")
    parser.add_argument("--city", type=str, default="San Diego, CA", help="City to scrape")
    parser.add_argument("--sample_only", type=lambda x: x.lower() in ['true','1','yes'], default=False, help="If True, only scrape a sample of 5 restaurants")
    args = parser.parse_args()

    scrape_city(city=args.city, sample_only=args.sample_only)
