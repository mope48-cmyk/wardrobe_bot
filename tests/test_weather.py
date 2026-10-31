"""Тесты погодного модуля на Open-Meteo (app/services/weather.py)."""

from app.services.weather import map_forecast, map_geocode_entry, map_weather_code


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
