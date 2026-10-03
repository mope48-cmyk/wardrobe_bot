"""Получение погоды и геокодирование через OpenWeatherMap."""

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

OWM_BASE = "https://api.openweathermap.org"


async def geocode_city(city: str) -> dict | None:
    """Найти координаты по названию города.

    Возвращает {"lat": ..., "lon": ..., "city": "..."} или None.
    """
    if not settings.OWM_API_KEY:
        logger.warning("OWM_API_KEY не задан")
        return None

    url = f"{OWM_BASE}/geo/1.0/direct"
    params = {"q": city, "limit": 1, "appid": settings.OWM_API_KEY}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception:
        logger.exception("Ошибка геокодирования для %r", city)
        return None

    if not data:
        return None

    first = data[0]
    # Локальное название на русском, если есть
    local_names = first.get("local_names") or {}
    display_name = local_names.get("ru") or first.get("name") or city

    return {
        "lat": first["lat"],
        "lon": first["lon"],
        "city": display_name,
    }


async def get_weather(lat: float, lon: float) -> dict | None:
    """Получить текущую погоду по координатам.

    Возвращает:
    {
        "temp": float,             # °C
        "feels_like": float,       # °C
        "description": str,        # "легкий дождь"
        "main": str,               # "Rain", "Snow", "Clear", "Clouds", ...
        "wind_speed": float,       # м/с
        "has_precipitation": bool,
    }
    """
    if not settings.OWM_API_KEY:
        return None

    url = f"{OWM_BASE}/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": settings.OWM_API_KEY,
        "units": "metric",
        "lang": "ru",
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception:
        logger.exception("Ошибка получения погоды lat=%s lon=%s", lat, lon)
        return None

    main = data.get("main", {})
    weather_list = data.get("weather") or [{}]
    weather = weather_list[0]
    wind = data.get("wind", {})

    weather_main = weather.get("main", "")

    return {
        "temp": float(main.get("temp", 0)),
        "feels_like": float(main.get("feels_like", 0)),
        "description": weather.get("description", ""),
        "main": weather_main,
        "wind_speed": float(wind.get("speed", 0)),
        "has_precipitation": weather_main in (
            "Rain", "Snow", "Drizzle", "Thunderstorm"
        ),
    }
