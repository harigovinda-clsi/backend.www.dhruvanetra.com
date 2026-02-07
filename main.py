"""
DhruvaNetra Backend API
A disciplined observer of near-Earth space.
Built with patience, precision, and purpose.
"""

from fastapi import FastAPI, HTTPException, Depends, status, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import httpx
import os
from passlib.context import CryptContext
import jwt
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import asyncio
from functools import lru_cache
import logging

# Configure logging with measured verbosity
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dhruva_netra")

# === CONFIGURATION ===
# Every system needs boundaries. Here are ours.

class Settings:
    """Configuration that adapts to environment without complaint."""
    
    def __init__(self):
        self.nasa_api_key = os.getenv("NASA_API_KEY", "DEMO_KEY")
        self.mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
        self.database_name = os.getenv("DATABASE_NAME", "dhruva_netra")
        self.jwt_secret = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
        self.jwt_algorithm = "HS256"
        self.access_token_expire_minutes = 1440  # 24 hours - patience is virtue
        self.nasa_base_url = "https://api.nasa.gov/neo/rest/v1"
        
        # Rate limiting - discipline in API consumption
        self.request_delay = 0.1  # Small pause between requests
        self.cache_ttl = 300  # 5 minutes - balance freshness with courtesy

@lru_cache()
def get_settings() -> Settings:
    """Single source of truth, cached for efficiency."""
    return Settings()

# === DATABASE CONNECTION ===
# Persistent memory across the void

class Database:
    """MongoDB connection manager. Patient and reliable."""
    
    client: Optional[AsyncIOMotorClient] = None
    
    @classmethod
    async def connect(cls):
        """Establish connection when application awakens."""
        settings = get_settings()
        cls.client = AsyncIOMotorClient(settings.mongodb_url)
        logger.info("Database connection established. Memory active.")
    
    @classmethod
    async def disconnect(cls):
        """Close connection gracefully when work concludes."""
        if cls.client:
            cls.client.close()
            logger.info("Database connection closed. Rest well.")
    
    @classmethod
    def get_database(cls):
        """Access the knowledge repository."""
        if not cls.client:
            raise RuntimeError("Database not connected. Cannot access memory.")
        settings = get_settings()
        return cls.client[settings.database_name]

# === SECURITY ===
# Protection without paranoia

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def hash_password(password: str) -> str:
    """Transform plaintext into secure form. One-way journey."""
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    """Compare without revealing. Truth emerges."""
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Generate JWT with appropriate lifespan."""
    settings = get_settings()
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)

def decode_access_token(token: str) -> dict:
    """Verify and extract claims from token."""
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired. Time waits for no one.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token. Authentication failed.")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Extract user from bearer token. Trust but verify."""
    token = credentials.credentials
    payload = decode_access_token(token)
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload.")
    
    db = Database.get_database()
    user = await db.users.find_one({"_id": ObjectId(user_id)})
    
    if not user:
        raise HTTPException(status_code=401, detail="User not found. Ghost in the machine.")
    
    return {
        "id": str(user["_id"]),
        "email": user["email"],
        "username": user["username"]
    }

# === DATA MODELS ===
# Structure brings clarity

class UserCreate(BaseModel):
    """Registration data. First step of the journey."""
    username: str = Field(..., min_length=3, max_length=30)
    email: EmailStr
    password: str = Field(..., min_length=8)
    
    @validator('username')
    def username_alphanumeric(cls, v):
        """Usernames should be simple and clear."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must be alphanumeric (with _ or - allowed)')
        return v

class UserLogin(BaseModel):
    """Authentication credentials. Identity verification."""
    email: EmailStr
    password: str

class Token(BaseModel):
    """Access token response. Key to the observatory."""
    access_token: str
    token_type: str = "bearer"
    username: str
    email: str

class WatchListItem(BaseModel):
    """Object to monitor. Chosen for observation."""
    neo_id: str
    designation: str
    added_at: Optional[datetime] = None

class AlertPreference(BaseModel):
    """User's notification preferences. Customized vigilance."""
    distance_threshold_au: float = Field(default=0.05, ge=0.001, le=1.0)
    velocity_threshold_kms: float = Field(default=20.0, ge=1.0, le=100.0)
    hazardous_only: bool = False
    email_notifications: bool = True

# === NASA NEO API CLIENT ===
# Bridge to external knowledge

class NASANeoClient:
    """Client for NASA's Near Earth Object Web Service.
    Respectful consumer of public data."""
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.nasa_base_url
        self.api_key = self.settings.nasa_api_key
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still fresh."""
        if key not in self._cache_timestamps:
            return False
        
        age = (datetime.utcnow() - self._cache_timestamps[key]).total_seconds()
        return age < self.settings.cache_ttl
    
    async def _get(self, endpoint: str, params: dict = None) -> dict:
        """Make GET request to NASA API with courtesy delay."""
        if params is None:
            params = {}
        
        params["api_key"] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        
        # Check cache first
        cache_key = f"{endpoint}:{str(params)}"
        if self._is_cache_valid(cache_key):
            logger.info(f"Cache hit for {endpoint}")
            return self._cache[cache_key]
        
        # Courtesy delay - we are patient observers
        await asyncio.sleep(self.settings.request_delay)
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Cache the result
                self._cache[cache_key] = data
                self._cache_timestamps[cache_key] = datetime.utcnow()
                
                return data
            except httpx.HTTPStatusError as e:
                logger.error(f"NASA API error: {e.response.status_code} - {e.response.text}")
                raise HTTPException(
                    status_code=502,
                    detail=f"External data source unavailable. Sky remains observable, patience required."
                )
            except Exception as e:
                logger.error(f"Unexpected error calling NASA API: {str(e)}")
                raise HTTPException(
                    status_code=503,
                    detail="Temporary difficulty accessing external data. Try again shortly."
                )
    
    async def get_feed(self, start_date: str, end_date: str) -> dict:
        """Retrieve NEO feed for date range (max 7 days)."""
        return await self._get("feed", {"start_date": start_date, "end_date": end_date})
    
    async def get_neo_lookup(self, neo_id: str) -> dict:
        """Lookup specific NEO by ID."""
        return await self._get(f"neo/{neo_id}")
    
    async def get_browse(self) -> dict:
        """Browse paginated list of NEOs."""
        return await self._get("neo/browse")

# Global client instance
nasa_client = NASANeoClient()

# === UTILITY FUNCTIONS ===
# Helper logic for transformations

def calculate_risk_score(neo_data: dict) -> float:
    """Calculate normalized risk score based on multiple factors.
    
    Considers:
    - Hazardous classification
    - Miss distance
    - Velocity
    - Diameter
    
    Returns score from 0.0 (negligible) to 10.0 (significant concern)
    """
    is_hazardous = neo_data.get("is_potentially_hazardous_asteroid", False)
    
    # Extract approach data (use closest approach)
    approaches = neo_data.get("close_approach_data", [])
    if not approaches:
        return 0.0
    
    closest = min(approaches, key=lambda x: float(x["miss_distance"]["astronomical"]))
    
    miss_distance_au = float(closest["miss_distance"]["astronomical"])
    velocity_kms = float(closest["relative_velocity"]["kilometers_per_second"])
    
    # Diameter estimation (use average of min and max)
    diameter_info = neo_data.get("estimated_diameter", {}).get("meters", {})
    diameter_min = diameter_info.get("estimated_diameter_min", 0)
    diameter_max = diameter_info.get("estimated_diameter_max", 0)
    diameter_avg = (diameter_min + diameter_max) / 2 if diameter_max > 0 else 0
    
    # Base score from hazardous classification
    score = 5.0 if is_hazardous else 2.0
    
    # Distance factor (closer = higher score)
    if miss_distance_au < 0.05:
        score += 3.0
    elif miss_distance_au < 0.1:
        score += 2.0
    elif miss_distance_au < 0.2:
        score += 1.0
    
    # Velocity factor (faster = higher concern)
    if velocity_kms > 30:
        score += 1.5
    elif velocity_kms > 20:
        score += 1.0
    elif velocity_kms > 15:
        score += 0.5
    
    # Size factor (larger = higher impact)
    if diameter_avg > 500:
        score += 1.5
    elif diameter_avg > 300:
        score += 1.0
    elif diameter_avg > 100:
        score += 0.5
    
    # Cap at 10.0
    return min(score, 10.0)

def format_neo_summary(neo_data: dict) -> dict:
    """Transform NASA API response into clean summary format."""
    approaches = neo_data.get("close_approach_data", [])
    closest = min(approaches, key=lambda x: float(x["miss_distance"]["astronomical"])) if approaches else None
    
    diameter_info = neo_data.get("estimated_diameter", {}).get("meters", {})
    
    return {
        "id": neo_data.get("id"),
        "neo_reference_id": neo_data.get("neo_reference_id"),
        "name": neo_data.get("name"),
        "designation": neo_data.get("designation", neo_data.get("name")),
        "is_hazardous": neo_data.get("is_potentially_hazardous_asteroid", False),
        "absolute_magnitude": neo_data.get("absolute_magnitude_h"),
        "estimated_diameter_m": {
            "min": diameter_info.get("estimated_diameter_min"),
            "max": diameter_info.get("estimated_diameter_max")
        },
        "closest_approach": {
            "date": closest.get("close_approach_date") if closest else None,
            "date_full": closest.get("close_approach_date_full") if closest else None,
            "velocity_kms": float(closest["relative_velocity"]["kilometers_per_second"]) if closest else None,
            "miss_distance_au": float(closest["miss_distance"]["astronomical"]) if closest else None,
            "miss_distance_km": float(closest["miss_distance"]["kilometers"]) if closest else None,
            "orbiting_body": closest.get("orbiting_body") if closest else None
        } if closest else None,
        "risk_score": calculate_risk_score(neo_data),
        "nasa_jpl_url": neo_data.get("nasa_jpl_url")
    }

# === FASTAPI APPLICATION ===
# The observatory opens its doors

app = FastAPI(
    title="DhruvaNetra API",
    description="Near-Earth Object monitoring with discipline and precision",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS - controlled openness
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === LIFECYCLE EVENTS ===

@app.on_event("startup")
async def startup_event():
    """Initialize connections when service awakens."""
    logger.info("DhruvaNetra backend initializing...")
    await Database.connect()
    
    # Create indexes for efficient queries
    db = Database.get_database()
    await db.users.create_index("email", unique=True)
    await db.users.create_index("username", unique=True)
    await db.watch_lists.create_index([("user_id", 1), ("neo_id", 1)], unique=True)
    
    logger.info("Observatory is operational. Observation begins.")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown when service rests."""
    logger.info("DhruvaNetra backend shutting down...")
    await Database.disconnect()
    logger.info("Observatory closed. Until next observation cycle.")

# === HEALTH CHECK ===

@app.get("/api/health", tags=["System"])
async def health_check():
    """Verify system responsiveness. Simple heartbeat."""
    return {
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "message": "The eye remains steady."
    }

# === AUTHENTICATION ENDPOINTS ===

@app.post("/api/auth/register", response_model=Token, status_code=status.HTTP_201_CREATED, tags=["Authentication"])
async def register_user(user: UserCreate):
    """Create new user account. Begin observation journey."""
    db = Database.get_database()
    
    # Check for existing user
    existing_email = await db.users.find_one({"email": user.email})
    if existing_email:
        raise HTTPException(
            status_code=400,
            detail="Email already registered. Each observer needs unique identity."
        )
    
    existing_username = await db.users.find_one({"username": user.username})
    if existing_username:
        raise HTTPException(
            status_code=400,
            detail="Username already taken. Choose another designation."
        )
    
    # Create user
    user_doc = {
        "username": user.username,
        "email": user.email,
        "password_hash": hash_password(user.password),
        "created_at": datetime.utcnow(),
        "alert_preferences": {
            "distance_threshold_au": 0.05,
            "velocity_threshold_kms": 20.0,
            "hazardous_only": False,
            "email_notifications": True
        }
    }
    
    result = await db.users.insert_one(user_doc)
    user_id = str(result.inserted_id)
    
    # Generate token
    access_token = create_access_token({"sub": user_id})
    
    logger.info(f"New observer registered: {user.username}")
    
    return Token(
        access_token=access_token,
        username=user.username,
        email=user.email
    )

@app.post("/api/auth/login", response_model=Token, tags=["Authentication"])
async def login_user(credentials: UserLogin):
    """Authenticate existing user. Verify identity."""
    db = Database.get_database()
    
    user = await db.users.find_one({"email": credentials.email})
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials. Identity not verified."
        )
    
    if not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials. Authentication failed."
        )
    
    # Generate token
    access_token = create_access_token({"sub": str(user["_id"])})
    
    logger.info(f"Observer authenticated: {user['username']}")
    
    return Token(
        access_token=access_token,
        username=user["username"],
        email=user["email"]
    )

@app.get("/api/auth/me", tags=["Authentication"])
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Retrieve current user information. Know thyself."""
    db = Database.get_database()
    user = await db.users.find_one({"_id": ObjectId(current_user["id"])})
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found in records.")
    
    return {
        "id": str(user["_id"]),
        "username": user["username"],
        "email": user["email"],
        "created_at": user["created_at"].isoformat(),
        "alert_preferences": user.get("alert_preferences", {})
    }

# === NEO DATA ENDPOINTS ===

@app.get("/api/neo/feed", tags=["Near-Earth Objects"])
async def get_neo_feed(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    current_user: dict = Depends(get_current_user)
):
    """Retrieve NEO feed for date range.
    
    If dates not provided, defaults to today through next 7 days.
    Maximum range is 7 days.
    """
    
    # Default to today + 7 days if not specified
    if not start_date:
        start_date = datetime.utcnow().strftime("%Y-%m-%d")
    if not end_date:
        end_date = (datetime.utcnow() + timedelta(days=7)).strftime("%Y-%m-%d")
    
    # Validate date range
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        if (end - start).days > 7:
            raise HTTPException(
                status_code=400,
                detail="Date range cannot exceed 7 days. Patience in observation."
            )
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use YYYY-MM-DD."
        )
    
    data = await nasa_client.get_feed(start_date, end_date)
    
    # Transform and enrich the data
    near_earth_objects = data.get("near_earth_objects", {})
    processed_neos = []
    
    for date, neos in near_earth_objects.items():
        for neo in neos:
            processed_neos.append(format_neo_summary(neo))
    
    # Sort by risk score descending
    processed_neos.sort(key=lambda x: x["risk_score"], reverse=True)
    
    logger.info(f"Feed retrieved: {len(processed_neos)} objects for {start_date} to {end_date}")
    
    return {
        "start_date": start_date,
        "end_date": end_date,
        "element_count": len(processed_neos),
        "near_earth_objects": processed_neos
    }

@app.get("/api/neo/{neo_id}", tags=["Near-Earth Objects"])
async def get_neo_details(
    neo_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Retrieve detailed information for specific NEO.
    
    Provides comprehensive orbital parameters, physical characteristics,
    and close approach data.
    """
    
    try:
        data = await nasa_client.get_neo_lookup(neo_id)
    except HTTPException as e:
        if e.status_code == 502:
            raise HTTPException(
                status_code=404,
                detail=f"Object {neo_id} not found in catalog. Verify designation."
            )
        raise
    
    # Format and enrich
    summary = format_neo_summary(data)
    
    # Add orbital elements
    orbital_data = data.get("orbital_data", {})
    summary["orbital_elements"] = {
        "orbit_id": orbital_data.get("orbit_id"),
        "orbit_determination_date": orbital_data.get("orbit_determination_date"),
        "first_observation_date": orbital_data.get("first_observation_date"),
        "last_observation_date": orbital_data.get("last_observation_date"),
        "data_arc_in_days": orbital_data.get("data_arc_in_days"),
        "observations_used": orbital_data.get("observations_used"),
        "orbit_uncertainty": orbital_data.get("orbit_uncertainty"),
        "minimum_orbit_intersection": orbital_data.get("minimum_orbit_intersection"),
        "jupiter_tisserand_invariant": orbital_data.get("jupiter_tisserand_invariant"),
        "epoch_osculation": orbital_data.get("epoch_osculation"),
        "eccentricity": orbital_data.get("eccentricity"),
        "semi_major_axis": orbital_data.get("semi_major_axis"),
        "inclination": orbital_data.get("inclination"),
        "ascending_node_longitude": orbital_data.get("ascending_node_longitude"),
        "orbital_period": orbital_data.get("orbital_period"),
        "perihelion_distance": orbital_data.get("perihelion_distance"),
        "perihelion_argument": orbital_data.get("perihelion_argument"),
        "aphelion_distance": orbital_data.get("aphelion_distance"),
        "perihelion_time": orbital_data.get("perihelion_time"),
        "mean_anomaly": orbital_data.get("mean_anomaly"),
        "mean_motion": orbital_data.get("mean_motion"),
        "equinox": orbital_data.get("equinox"),
        "orbit_class": orbital_data.get("orbit_class", {})
    }
    
    # Add all close approaches
    summary["all_close_approaches"] = data.get("close_approach_data", [])
    
    logger.info(f"Details retrieved for object: {neo_id}")
    
    return summary

# === WATCH LIST ENDPOINTS ===

@app.get("/api/watch-list", tags=["Watch List"])
async def get_watch_list(current_user: dict = Depends(get_current_user)):
    """Retrieve user's watch list. Objects under continuous observation."""
    db = Database.get_database()
    
    watch_items = await db.watch_lists.find({"user_id": current_user["id"]}).to_list(length=100)
    
    # Enrich with latest data from NASA
    enriched_items = []
    for item in watch_items:
        try:
            neo_data = await nasa_client.get_neo_lookup(item["neo_id"])
            enriched = format_neo_summary(neo_data)
            enriched["watched_since"] = item["added_at"].isoformat()
            enriched_items.append(enriched)
        except Exception as e:
            logger.warning(f"Could not enrich watch item {item['neo_id']}: {str(e)}")
            # Include basic info even if NASA API fails
            enriched_items.append({
                "neo_id": item["neo_id"],
                "designation": item["designation"],
                "watched_since": item["added_at"].isoformat(),
                "error": "Current data unavailable"
            })
    
    logger.info(f"Watch list retrieved for user {current_user['username']}: {len(enriched_items)} objects")
    
    return {
        "count": len(enriched_items),
        "items": enriched_items
    }

@app.post("/api/watch-list", status_code=status.HTTP_201_CREATED, tags=["Watch List"])
async def add_to_watch_list(
    item: WatchListItem,
    current_user: dict = Depends(get_current_user)
):
    """Add NEO to watch list. Begin dedicated observation."""
    db = Database.get_database()
    
    # Verify NEO exists
    try:
        await nasa_client.get_neo_lookup(item.neo_id)
    except:
        raise HTTPException(
            status_code=404,
            detail=f"Object {item.neo_id} not found. Verify designation."
        )
    
    # Check if already in watch list
    existing = await db.watch_lists.find_one({
        "user_id": current_user["id"],
        "neo_id": item.neo_id
    })
    
    if existing:
        raise HTTPException(
            status_code=400,
            detail="Object already in watch list. Observation continues."
        )
    
    # Add to watch list
    watch_doc = {
        "user_id": current_user["id"],
        "neo_id": item.neo_id,
        "designation": item.designation,
        "added_at": datetime.utcnow()
    }
    
    await db.watch_lists.insert_one(watch_doc)
    
    logger.info(f"Object {item.neo_id} added to watch list for user {current_user['username']}")
    
    return {
        "message": "Object added to watch list successfully",
        "neo_id": item.neo_id,
        "designation": item.designation
    }

@app.delete("/api/watch-list/{neo_id}", tags=["Watch List"])
async def remove_from_watch_list(
    neo_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Remove NEO from watch list. Conclude observation."""
    db = Database.get_database()
    
    result = await db.watch_lists.delete_one({
        "user_id": current_user["id"],
        "neo_id": neo_id
    })
    
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail="Object not found in watch list."
        )
    
    logger.info(f"Object {neo_id} removed from watch list for user {current_user['username']}")
    
    return {
        "message": "Object removed from watch list",
        "neo_id": neo_id
    }

# === ALERT PREFERENCES ===

@app.put("/api/preferences/alerts", tags=["User Preferences"])
async def update_alert_preferences(
    preferences: AlertPreference,
    current_user: dict = Depends(get_current_user)
):
    """Update user's alert notification preferences. Customize vigilance."""
    db = Database.get_database()
    
    result = await db.users.update_one(
        {"_id": ObjectId(current_user["id"])},
        {"$set": {"alert_preferences": preferences.dict()}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(
            status_code=500,
            detail="Failed to update preferences. Try again."
        )
    
    logger.info(f"Alert preferences updated for user {current_user['username']}")
    
    return {
        "message": "Alert preferences updated successfully",
        "preferences": preferences.dict()
    }

# === ANALYTICS ENDPOINTS ===

@app.get("/api/analytics/summary", tags=["Analytics"])
async def get_analytics_summary(current_user: dict = Depends(get_current_user)):
    """Retrieve summary analytics. Understand the patterns."""
    
    # Get current week's data
    start_date = datetime.utcnow().strftime("%Y-%m-%d")
    end_date = (datetime.utcnow() + timedelta(days=7)).strftime("%Y-%m-%d")
    
    feed_data = await nasa_client.get_feed(start_date, end_date)
    
    all_neos = []
    for date, neos in feed_data.get("near_earth_objects", {}).items():
        all_neos.extend(neos)
    
    # Calculate statistics
    total_count = len(all_neos)
    hazardous_count = sum(1 for neo in all_neos if neo.get("is_potentially_hazardous_asteroid", False))
    
    # Close approaches (within 0.1 AU)
    close_approaches = sum(
        1 for neo in all_neos
        for approach in neo.get("close_approach_data", [])
        if float(approach["miss_distance"]["astronomical"]) < 0.1
    )
    
    # Average miss distance
    all_distances = [
        float(approach["miss_distance"]["astronomical"])
        for neo in all_neos
        for approach in neo.get("close_approach_data", [])
    ]
    avg_distance = sum(all_distances) / len(all_distances) if all_distances else 0
    
    # Get user's watch list count
    db = Database.get_database()
    watch_count = await db.watch_lists.count_documents({"user_id": current_user["id"]})
    
    return {
        "period": {
            "start": start_date,
            "end": end_date
        },
        "totals": {
            "objects_tracked": total_count,
            "hazardous_objects": hazardous_count,
            "close_approaches": close_approaches,
            "in_watch_list": watch_count
        },
        "averages": {
            "miss_distance_au": round(avg_distance, 4)
        },
        "generated_at": datetime.utcnow().isoformat()
    }

# === ROOT ENDPOINT ===

@app.get("/", tags=["System"])
async def root():
    """API root. Gateway to observation."""
    return {
        "service": "DhruvaNetra Backend API",
        "version": "1.0.0",
        "status": "operational",
        "motto": "The steady eye on near-Earth space",
        "documentation": "/api/docs",
        "health": "/api/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)