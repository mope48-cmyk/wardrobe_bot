"""Получение погоды и геокодирование.

Основной источник погоды — OpenWeatherMap (OWM_API_KEY из .env, ~1000
запросов/день бесплатно). Фолбэк — Open-Meteo (без ключа). Если недоступны
оба — используется FALLBACK_WEATHER (см. get_weather_or_fallback).

Контракт функций сохранён:

    geocode_city(city) -> {"lat", "lon", "city"} | None
    get_weather(lat, lon) -> {"temp", "feels_like", "description", "main",
                               "wind_speed", "has_precipitation"} | None
"""

import asyncio
import logging
import os

import httpx

logger = logging.getLogger(__name__)

GEOCODING_API = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"
OWM_API = "https://api.openweathermap.org/data/2.5/weather"

# Таймаут: соединение — 5с, чтение — 10с (в сумме не даём зависнуть).
_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
# Попыток с ретраем на транспортные ошибки (итого = _MAX_ATTEMPTS раз).
_MAX_ATTEMPTS = 3

# Нейтральная погода для деградации, когда API недоступен: образ собирается
# по сезону/умолчанию, а пользователь не остаётся без ответа.
FALLBACK_WEATHER = {
    "temp": 10.0,
    "feels_like": 10.0,
    "description": "неизвестна",
    "main": "Clouds",
    "wind_speed": 0.0,
    "has_precipitation": False,
}

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


async def _request_json(url: str, params: dict) -> dict | None:
    """GET с ретраем на транспортные ошибки; None при итоговой неудаче."""
    last_exc = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                r = await client.get(url, params=params)
                r.raise_for_status()
                return r.json()
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt < _MAX_ATTEMPTS - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
        except Exception:
            # Не-транспортная ошибка (например, битый JSON) — ретраить бессмысленно
            logger.exception("Ошибка запроса %s", url)
            return None

    logger.warning(
        "API %s недоступен после %d попыток: %r",
        url, _MAX_ATTEMPTS, last_exc,
    )
    return None


async def geocode_city(city: str) -> dict | None:
    """Найти координаты по названию города."""
    if not city or not city.strip():
        return None

    params = {"name": city.strip(), "count": 1, "language": "ru", "format": "json"}
    data = await _request_json(GEOCODING_API, params)
    if not data or not data.get("results"):
        return None
    return map_geocode_entry(data["results"][0], city)


def map_owm(data: dict) -> dict | None:
    """Преобразовать ответ OpenWeatherMap в наш словарь погоды."""
    main = data.get("main") or {}
    weather_arr = data.get("weather") or []
    if not main or not weather_arr:
        return None

    w = weather_arr[0]
    code_main = w.get("main") or "Clouds"
    has_precip = (
        code_main in _PRECIPITATION_MAINS
        or bool(data.get("rain"))
        or bool(data.get("snow"))
    )
    return {
        "temp": float(main.get("temp", 0)),
        "feels_like": float(main.get("feels_like", 0)),
        "description": w.get("description") or "",
        "main": code_main,
        "wind_speed": float((data.get("wind") or {}).get("speed", 0)),
        "has_precipitation": has_precip,
    }


async def _get_weather_owm(lat: float, lon: float) -> dict | None:
    """OpenWeatherMap (основной источник)."""
    api_key = os.getenv("OWM_API_KEY", "").strip()
    if not api_key:
        logger.warning("OWM_API_KEY не задан — пропускаем OpenWeatherMap")
        return None

    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric",
        "lang": "ru",
    }
    data = await _request_json(OWM_API, params)
    if not data:
        logger.warning("OpenWeatherMap недоступен (пустой ответ)")
        return None
    if str(data.get("cod")) != "200":
        logger.warning("OpenWeatherMap вернул ошибку: %s", data.get("message"))
        return None
    return map_owm(data)


async def get_weather(lat: float, lon: float) -> dict | None:
    """Получить текущую погоду: OWM, затем Open-Meteo как фолбэк."""
    owm = await _get_weather_owm(lat, lon)
    if owm is not None:
        return owm

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": (
            "temperature_2m,apparent_temperature,weather_code,wind_speed_10m"
        ),
        "wind_speed_unit": "ms",
        "timezone": "auto",
    }
    data = await _request_json(FORECAST_API, params)
    if not data:
        return None
    return map_forecast(data)


async def get_weather_or_fallback(lat: float, lon: float) -> tuple[dict, bool]:
    """Погода или нейтральный фолбэк.

    Возвращает (weather, was_fallback): при сбое API подставляется
    FALLBACK_WEATHER, чтобы вызовы не обрывались и пользователь оставался
    с ответом.
    """
    weather = await get_weather(lat, lon)
    if weather is None:
        return FALLBACK_WEATHER, True
    return weather, False
