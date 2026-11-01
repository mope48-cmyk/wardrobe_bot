"""Тесты погодного модуля на Open-Meteo (app/services/weather.py)."""

import httpx
import pytest

from app.services.weather import (
    FALLBACK_WEATHER,
    get_weather,
    get_weather_or_fallback,
    map_forecast,
    map_geocode_entry,
    map_owm,
    map_weather_code,
)


def test_map_weather_code_known():
    assert map_weather_code(0) == ("Clear", "ясно")
    assert map_weather_code(63) == ("Rain", "дождь")
    assert map_weather_code(71) == ("Snow", "небольшой снег")
    assert map_weather_code(95) == ("Thunderstorm", "гроза")


def test_map_weather_code_unknown_defaults_to_clouds():
    assert map_weather_code(None) == ("Clouds", "облачно")
    assert map_weather_code(999) == ("Clouds", "облачно")


def test_map_geocode_entry_ru_name():
    entry = {
        "latitude": 55.7558,
        "longitude": 37.6173,
        "name": "Moscow",
        "local_names": {"ru": "Москва"},
    }
    assert map_geocode_entry(entry, "Moscow") == {
        "lat": 55.7558,
        "lon": 37.6173,
        "city": "Москва",
    }


def test_map_geocode_entry_empty():
    assert map_geocode_entry({}, "x") is None
    assert map_geocode_entry(None, "x") is None
    assert map_geocode_entry({"latitude": None, "longitude": None}, "x") is None


def test_map_forecast_rain():
    data = {
        "current": {
            "temperature_2m": 12.3,
            "apparent_temperature": 10.1,
            "weather_code": 63,
            "wind_speed_10m": 5.2,
        }
    }
    weather = map_forecast(data)
    assert weather["temp"] == 12.3
    assert weather["feels_like"] == 10.1
    assert weather["main"] == "Rain"
    assert weather["has_precipitation"] is True
    assert weather["wind_speed"] == 5.2


def test_map_forecast_clear_no_precipitation():
    data = {"current": {"weather_code": 0, "temperature_2m": 20}}
    weather = map_forecast(data)
    assert weather["main"] == "Clear"
    assert weather["has_precipitation"] is False


def test_map_forecast_empty():
    assert map_forecast({}) is None
    assert map_forecast({"current": None}) is None


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _FlakyClient:
    """Первый _N вызовов (глобально) падают с ConnectTimeout, дальше — успех."""

    fails = 0
    total_calls = 0

    def __init__(self, timeout=None):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None):
        _FlakyClient.total_calls += 1
        if _FlakyClient.total_calls <= _FlakyClient.fails:
            raise httpx.ConnectTimeout("boom")
        return _FakeResponse(
            {"current": {"temperature_2m": 5.0, "weather_code": 0}}
        )


@pytest.mark.asyncio
async def test_get_weather_retries_then_succeeds(monkeypatch):
    _FlakyClient.fails = 2
    _FlakyClient.total_calls = 0
    monkeypatch.setattr("app.services.weather.httpx.AsyncClient", _FlakyClient)

    weather = await get_weather(48.85, 2.35)

    assert weather is not None
    assert weather["temp"] == 5.0
    assert weather["main"] == "Clear"


@pytest.mark.asyncio
async def test_get_weather_gives_up_after_max_attempts(monkeypatch, caplog):
    _FlakyClient.fails = 999
    _FlakyClient.total_calls = 0
    monkeypatch.setattr("app.services.weather.httpx.AsyncClient", _FlakyClient)

    weather = await get_weather(48.85, 2.35)

    assert weather is None
    assert "недоступен после 3 попыток" in caplog.text


@pytest.mark.asyncio
async def test_get_weather_or_fallback_returns_fallback(monkeypatch):
    async def _none(*a, **k):
        return None

    monkeypatch.setattr("app.services.weather.get_weather", _none)

    weather, was_fallback = await get_weather_or_fallback(48.85, 2.35)

    assert was_fallback is True
    assert weather is FALLBACK_WEATHER


@pytest.mark.asyncio
async def test_get_weather_or_fallback_passthrough(monkeypatch):
    async def _some(*a, **k):
        return {"temp": 5.0}

    monkeypatch.setattr("app.services.weather.get_weather", _some)

    weather, was_fallback = await get_weather_or_fallback(48.85, 2.35)

    assert was_fallback is False
    assert weather["temp"] == 5.0


def test_map_owm_rain():
    data = {
        "cod": 200,
        "main": {"temp": 15.2, "feels_like": 13.8},
        "weather": [{"main": "Rain", "description": "небольшой дождь"}],
        "wind": {"speed": 4.5},
        "rain": {"1h": 0.2},
    }
    weather = map_owm(data)
    assert weather["temp"] == 15.2
    assert weather["feels_like"] == 13.8
    assert weather["main"] == "Rain"
    assert weather["description"] == "небольшой дождь"
    assert weather["wind_speed"] == 4.5
    assert weather["has_precipitation"] is True


def test_map_owm_empty():
    assert map_owm({}) is None
    assert map_owm({"main": {}, "weather": []}) is None


@pytest.mark.asyncio
async def test_get_weather_prefers_owm(monkeypatch):
    owm_result = {"temp": 12.0, "main": "Clear", "has_precipitation": False}

    async def _fake_owm(lat, lon):
        return owm_result

    async def _fail(*a, **k):
        raise AssertionError("Open-Meteo не должен вызываться при успешном OWM")

    monkeypatch.setattr("app.services.weather._get_weather_owm", _fake_owm)
    monkeypatch.setattr("app.services.weather._request_json", _fail)

    assert await get_weather(48.85, 2.35) == owm_result


@pytest.mark.asyncio
async def test_get_weather_falls_back_to_meteo(monkeypatch):
    async def _none(lat, lon):
        return None

    meteo_data = {"current": {"temperature_2m": 7.5, "weather_code": 61}}

    async def _meteo(url, params):
        return meteo_data

    monkeypatch.setattr("app.services.weather._get_weather_owm", _none)
    monkeypatch.setattr("app.services.weather._request_json", _meteo)

    weather = await get_weather(48.85, 2.35)
    assert weather is not None
    assert weather["main"] == "Rain"
    assert weather["temp"] == 7.5
