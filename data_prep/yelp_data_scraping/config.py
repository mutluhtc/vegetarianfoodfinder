# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Yelp API key
YELP_API_KEY = os.getenv("YELP_API_KEY")

# Yelp API endpoints
SEARCH_URL = "https://api.yelp.com/v3/businesses/search"
DETAILS_URL = "https://api.yelp.com/v3/businesses/{}"
REVIEWS_URL = "https://api.yelp.com/v3/businesses/{}/reviews"

# Request headers
HEADERS = {"Authorization": f"Bearer {YELP_API_KEY}"}