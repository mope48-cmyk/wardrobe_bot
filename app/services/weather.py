"""Получение погоды и геокодирование через Open-Meteo.

Open-Meteo — бесплатный API без ключа и без жёстких лимитов, в отличие
от OpenWeatherMap (OWM_API_KEY + ~1000 запросов/день в бесплатном тире).
Контракт функций сохранён:

    geocode_city(city) -> {"lat", "lon", "city"} | None
    get_weather(lat, lon) -> {"temp", "feels_like", "description", "main",
                               "wind_speed", "has_precipitation"} | None
"""

import logging

import httpx

logger = logging.getLogger(__name__)

GEOCODING_API = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"

# Коды погоды WMO → (короткий код, русское описание)
_WMO = {
    0: ("Clear", "ясно"),
    1: ("Clear", "преимущественно ясно"),
    2: ("Clouds", "переменная облачность"),
    3: ("Clouds", "пасмурно"),
    45: ("Fog", "туман"),
    48: ("Fog", "изморозь"),
    51: ("Drizzle", "лёгкая морось"),
    53: ("Drizzle", "морось"),
    55: ("Drizzle", "сильная морось"),
    56: ("Drizzle", "ледяная морось"),
    57: ("Drizzle", "сильная ледяная морось"),
    61: ("Rain", "небольшой дождь"),
    63: ("Rain", "дождь"),
    65: ("Rain", "сильный дождь"),
    66: ("Rain", "ледяной дождь"),
    67: ("Rain", "сильный ледяной дождь"),
    71: ("Snow", "небольшой снег"),
    73: ("Snow", "снег"),
    75: ("Snow", "сильный снег"),
    77: ("Snow", "снежная крупа"),
    80: ("Rain", "ливень"),
    81: ("Rain", "сильный ливень"),
    82: ("Rain", "экстремальный ливень"),
    85: ("Snow", "снежный ливень"),
    86: ("Snow", "сильный снежный ливень"),
    95: ("Thunderstorm", "гроза"),
    96: ("Thunderstorm", "гроза с градом"),
    99: ("Thunderstorm", "сильная гроза с градом"),
}

_PRECIPITATION_MAINS = {"Rain", "Snow", "Drizzle", "Thunderstorm"}


def map_weather_code(code: int | None) -> tuple[str, str]:
    """Сопоставить код WMO с (короткий код, описание на русском)."""
    if code is None or code not in _WMO:
        return ("Clouds", "облачно")
    return _WMO[code]


def map_geocode_entry(entry: dict, fallback_city: str) -> dict | None:
    """Преобразовать элемент ответа geocoding API в наш формат."""
    if not entry:
        return None
    lat = entry.get("latitude")
    lon = entry.get("longitude")
    if lat is None or lon is None:
        return None
    # Локальное название на русском, если есть
    local_names = entry.get("local_names") or {}
    display_name = local_names.get("ru") or entry.get("name") or fallback_city
    return {"lat": float(lat), "lon": float(lon), "city": display_name}


def map_forecast(data: dict) -> dict | None:
    """Преобразовать ответ forecast API в наш словарь погоды."""
    current = data.get("current") or {}
    if not current:
        return None
    code = current.get("weather_code")
    main, description = map_weather_code(code)
    return {
        "temp": float(current.get("temperature_2m", 0)),
        "feels_like": float(current.get("apparent_temperature", 0)),
        "description": description,
        "main": main,
        "wind_speed": float(current.get("wind_speed_10m", 0)),
        "has_precipitation": main in _PRECIPITATION_MAINS,
    }


async def geocode_city(city: str) -> dict | None:
    """Найти координаты по названию города."""
    if not city or not city.strip():
        return None

    params = {"name": city.strip(), "count": 1, "language": "ru", "format": "json"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(GEOCODING_API, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception:
        logger.exception("Ошибка геокодирования для %r", city)
        return None

    if not data or not data.get("results"):
        return None
    return map_geocode_entry(data["results"][0], city)


async def get_weather(lat: float, lon: float) -> dict | None:
    """Получить текущую погоду по координатам."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": (
            "temperature_2m,apparent_temperature,weather_code,wind_speed_10m"
        ),
        "wind_speed_unit": "ms",
        "timezone": "auto",
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(FORECAST_API, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception:
        logger.exception("Ошибка получения погоды lat=%s lon=%s", lat, lon)
        return None

    return map_forecast(data)
