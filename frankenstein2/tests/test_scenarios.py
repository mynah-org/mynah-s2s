"""Demo scenarios (pure Python): the catalogue, the offline tools, and the live tools on canned API answers."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frankenstein2 import scenarios as SC  # noqa: E402


def test_every_scenario_is_complete():
    for s in SC.SCENARIOS.values():
        p = s.public()
        assert {"en", "it", "fr"} <= set(p["title"]) and {"en", "it", "fr"} <= set(p["examples"])
        for t in s.tools:
            assert t.parameters["type"] == "object" and set(t.parameters["required"]) <= set(t.parameters["properties"])
        if s.tools:
            assert {"en", "it", "fr"} <= set(s.demo) and s.system_prompt
    assert SC.get("nope").id == "store" and SC.get(None).id == "store"


def test_calculator_is_safe_and_exact():
    assert json.loads(SC.calculate({"expression": "2450*0.18"}))["result"] == 441.0
    assert json.loads(SC.calculate({"expression": "18% * 2450"}))["result"] == 441.0
    assert json.loads(SC.calculate({"expression": "340 × 12"}))["result"] == 4080
    for bad in ("__import__('os').system('true')", "2**10000", "1/0", "open('x')"):
        assert "error" in json.loads(SC.calculate({"expression": bad}))


def test_restaurant_books_and_cancels():
    b = SC.Bookings()
    av = json.loads(b.availability({"date": "tomorrow", "people": 4}))
    assert av["people"] == 4 and av["free_times"]
    assert json.loads(b.availability({"date": "today", "people": 12}))["available"] is False
    t = av["free_times"][0]
    r = json.loads(b.book({"name": "Rossi", "date": "tomorrow", "time": t, "people": 4}))
    assert r["booked"] and r["code"].startswith("MH-")
    again = json.loads(b.book({"name": "Rossi", "date": "tomorrow", "time": t, "people": 4}))
    assert again["code"] == r["code"] and again["already_booked"] and len(b.by_code) == 1   # one table, not two
    assert "past" in json.loads(b.book({"name": "Rossi", "date": "2023-10-25", "time": t, "people": 4}))["error"]
    assert "past" in json.loads(b.availability({"date": "2023-10-25", "people": 4}))["error"]
    assert json.loads(b.cancel({"code": r["code"].lower()}))["cancelled"]
    assert "error" in json.loads(b.cancel({"code": r["code"]}))


def test_weather_and_time_on_canned_api_answers(monkeypatch):
    def fake(url, params=None):
        if "geocoding" in url:
            return {"results": [{"name": "Milano", "country": "Italia", "latitude": 45.46, "longitude": 9.19,
                                 "timezone": "Europe/Rome"}]}
        return {"current": {"temperature_2m": 11.4, "apparent_temperature": 9.8, "weather_code": 61,
                            "wind_speed_10m": 7.2, "relative_humidity_2m": 80},
                "daily": {"time": ["2026-11-19", "2026-11-20", "2026-11-21"], "weather_code": [61, 3, 0],
                          "temperature_2m_max": [12.1, 13.0, 14.2], "temperature_2m_min": [6.0, 5.5, 4.9],
                          "precipitation_probability_max": [80, 20, 0]}}
    monkeypatch.setattr(SC, "_get", fake)
    w = json.loads(SC.get_weather({"city": "Milano"}))
    assert w["place"] == "Milano, Italia" and w["now"]["conditions"] == "light rain" and w["tomorrow"]["max_c"] == 13
    t = json.loads(SC.get_local_time({"city": "Milano"}))
    assert t["timezone"] == "Europe/Rome" and len(t["time"]) == 5


def test_network_errors_become_tool_errors(monkeypatch):
    def boom(url, params=None):
        raise OSError("no network")
    monkeypatch.setattr(SC, "_get", boom)
    out = json.loads(SC.timed(SC.get_weather)({"city": "Paris"}))
    assert "error" in out and "no network" in out["error"]
