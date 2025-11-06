import requests
import time
from config import SEARCH_URL, DETAILS_URL, HEADERS

def fetch_restaurants(location="San Diego, CA", term="restaurants", max_fetch=200):
    """
    Fetch restaurants from Yelp by location and term.
    Matches notebook behavior: logs progress, uses 50-limit offsets.
    """
    results = []
    limit = 50  # Yelp API max per request
    for offset in range(0, max_fetch, limit):
        params = {"location": location, "term": term, "limit": limit, "offset": offset}
        try:
            r = requests.get(SEARCH_URL, headers=HEADERS, params=params)
            if r.status_code != 200:
                print(f"Error {r.status_code}: {r.text}")
                break
            chunk = r.json().get("businesses", [])
            if not chunk:
                break
            results.extend(chunk)
            print(f"Fetched {len(results)} for term '{term}' so far...")
            time.sleep(0.5)
        except Exception as e:
            print(f"Request error: {e}")
            break
    return results

def fetch_business_details(business_id):
    """
    Fetch detailed info for a Yelp business.
    Matches notebook: captures URL and menu_url if available.
    """
    try:
        r = requests.get(DETAILS_URL.format(business_id), headers=HEADERS)
        if r.status_code != 200:
            print(f"Error fetching details for {business_id}: {r.status_code}")
            return None
        return r.json()
    except Exception as e:
        print(f"Request exception for {business_id}: {e}")
        return None

def deduplicate_restaurants(restaurants):
    """
    Remove duplicates based on Yelp business ID.
    Keeps the first occurrence (matches notebook).
    """
    unique = {}
    for biz in restaurants:
        if biz['id'] not in unique:
            unique[biz['id']] = biz
    return list(unique.values())
