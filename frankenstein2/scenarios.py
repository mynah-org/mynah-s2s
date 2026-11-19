"""Demo scenarios: an agent persona (system prompt), its tools, and example lines per language.

A session picks one with ``"scenario": "<id>"`` in session.start (default ``store``: the furniture-store tenant of
server/build.py). The web UI lists them (``public()``, sent with the server's health probe): a title and a blurb,
"try saying" examples and the demo dialogue's caller lines in Italian, English and French.

Tools are plain functions ``handler(args: dict) -> str`` (a JSON string for the model). The live ones call free
public APIs that need no key (Open-Meteo, Frankfurter / ECB, Nager.Date, Wikipedia) with a short timeout; nothing
about the caller is sent besides the tool's arguments. Adding a scenario = one more entry in SCENARIOS.
"""

from __future__ import annotations

import ast
import json
import operator
import random
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta

UA = "mynah-s2s-demo/0.1 (+https://github.com/mynah-org/mynah-s2s)"
HTTP_TIMEOUT_S = 6.0


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: object                     # callable(args: dict) -> str


@dataclass
class Scenario:
    id: str
    title: dict                         # lang -> title
    blurb: dict                         # lang -> one line
    system_prompt: str                  # English; the engine adds the session language and the call policy
    tools: list = field(default_factory=list)     # [] = the server's tenant (the store)
    examples: dict = field(default_factory=dict)  # lang -> [what the caller could say]
    demo: dict = field(default_factory=dict)      # lang -> [demo dialogue caller lines]

    def public(self) -> dict:
        return {"id": self.id, "title": self.title, "blurb": self.blurb, "system_prompt": self.system_prompt,
                "tools": [t.name for t in self.tools], "examples": self.examples, "demo": self.demo,
                "languages": sorted(set(self.examples) | set(self.demo))}


def _get(url: str, params: dict | None = None) -> dict:
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as r:
        return json.loads(r.read().decode("utf-8"))


def _out(d) -> str:
    return json.dumps(d, ensure_ascii=False)


# ------------------------------------------------------------------ weather and time (Open-Meteo, no key)
WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
       51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
       61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain", 71: "light snow",
       73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers", 82: "violent showers",
       85: "snow showers", 86: "heavy snow showers", 95: "thunderstorm", 96: "thunderstorm with hail",
       99: "thunderstorm with heavy hail"}


def geocode(place: str, lang: str = "en") -> dict | None:
    d = _get("https://geocoding-api.open-meteo.com/v1/search", {"name": place, "count": 1, "language": lang,
                                                                 "format": "json"})
    r = (d.get("results") or [None])[0]
    if not r:
        return None
    return {"name": r.get("name"), "country": r.get("country"), "lat": r["latitude"], "lon": r["longitude"],
            "timezone": r.get("timezone") or "UTC"}


def get_weather(args: dict) -> str:
    place = str(args.get("city") or args.get("place") or "").strip()
    if not place:
        return _out({"error": "which city?"})
    g = geocode(place)
    if g is None:
        return _out({"error": f"no place called {place!r}"})
    d = _get("https://api.open-meteo.com/v1/forecast", {
        "latitude": g["lat"], "longitude": g["lon"], "timezone": "auto", "forecast_days": 3,
        "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,relative_humidity_2m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"})
    cur, day = d.get("current", {}), d.get("daily", {})

    def daily(i):
        return {"date": day["time"][i], "conditions": WMO.get(day["weather_code"][i], "unknown"),
                "min_c": round(day["temperature_2m_min"][i]), "max_c": round(day["temperature_2m_max"][i]),
                "rain_chance_pct": day["precipitation_probability_max"][i]}
    return _out({"place": f"{g['name']}, {g['country']}",
                 "now": {"temp_c": round(cur.get("temperature_2m", 0)),
                         "feels_like_c": round(cur.get("apparent_temperature", 0)),
                         "conditions": WMO.get(cur.get("weather_code"), "unknown"),
                         "wind_kmh": round(cur.get("wind_speed_10m", 0)),
                         "humidity_pct": cur.get("relative_humidity_2m")},
                 "today": daily(0), "tomorrow": daily(1), "day_after": daily(2)})


def get_local_time(args: dict) -> str:
    from zoneinfo import ZoneInfo
    place = str(args.get("city") or args.get("place") or "").strip()
    g = geocode(place) if place else None
    if g is None:
        return _out({"error": f"no place called {place!r}"})
    now = datetime.now(ZoneInfo(g["timezone"]))
    return _out({"place": f"{g['name']}, {g['country']}", "time": now.strftime("%H:%M"),
                 "weekday": now.strftime("%A"), "date": now.strftime("%Y-%m-%d"), "timezone": g["timezone"],
                 "utc_offset": now.strftime("%z")})


# ------------------------------------------------------------------ travel (Frankfurter / ECB, Nager.Date)
def convert_currency(args: dict) -> str:
    try:
        amount = float(str(args.get("amount", 1)).replace(",", "."))
    except ValueError:
        return _out({"error": "amount must be a number"})
    f, t = str(args.get("from", "EUR")).upper()[:3], str(args.get("to", "USD")).upper()[:3]
    d = _get("https://api.frankfurter.app/latest", {"amount": amount, "from": f, "to": t})
    if t not in (d.get("rates") or {}):
        return _out({"error": f"no rate for {f} to {t}"})
    return _out({"amount": amount, "from": f, "to": t, "result": round(d["rates"][t], 2), "rate_date": d.get("date"),
                 "source": "European Central Bank"})


def next_public_holiday(args: dict) -> str:
    cc = str(args.get("country_code", "")).upper()[:2]
    if len(cc) != 2:
        return _out({"error": "a two-letter country code, e.g. IT, FR, US"})
    d = _get(f"https://date.nager.at/api/v3/NextPublicHolidays/{cc}")
    if not d:
        return _out({"error": f"no holidays known for {cc}"})
    return _out({"country": cc, "next": [{"date": h["date"], "name": h["localName"], "english_name": h["name"]}
                                         for h in d[:3]]})


# ------------------------------------------------------------------ knowledge (Wikipedia, calculator)
def wikipedia_summary(args: dict) -> str:
    topic = str(args.get("topic", "")).strip()
    lang = str(args.get("language", "en"))[:2] or "en"
    if not topic:
        return _out({"error": "which topic?"})
    s = _get(f"https://{lang}.wikipedia.org/w/api.php", {"action": "opensearch", "search": topic, "limit": 1,
                                                         "namespace": 0, "format": "json"})
    if not s or len(s) < 2 or not s[1]:
        return _out({"error": f"nothing on {lang}.wikipedia for {topic!r}"})
    title = s[1][0]
    d = _get(f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title.replace(" ", "_")))
    text = (d.get("extract") or "")[:700]
    return _out({"title": d.get("title", title), "summary": text, "source": f"{lang}.wikipedia.org"})


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos, ast.FloorDiv: operator.floordiv}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(b) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / ** ( )")


def calculate(args: dict) -> str:
    expr = str(args.get("expression", "")).replace("×", "*").replace("÷", "/").replace(",", ".")
    expr = expr.replace("%", "/100")              # "18%*2450" as said; the modulo operator is not offered
    try:
        v = _eval(ast.parse(expr, mode="eval"))
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as e:
        return _out({"error": f"cannot compute {expr!r}: {e}"})
    return _out({"expression": expr, "result": round(v, 6) if isinstance(v, float) else v})


# ------------------------------------------------------------------ restaurant booking (state in memory)
class Bookings:
    """A fictional restaurant: 20:00 is full on Fridays and Saturdays; tables for up to 8."""

    def __init__(self):
        self.lock = threading.Lock()
        self.by_code: dict = {}

    @staticmethod
    def _date(s: str):
        s = str(s or "").strip().lower()
        today = datetime.now().date()
        rel = {"today": 0, "oggi": 0, "aujourd'hui": 0, "tomorrow": 1, "domani": 1, "demain": 1}
        if s in rel:
            return today + timedelta(days=rel[s])
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m"):
            try:
                d = datetime.strptime(s, fmt).date()
                return d.replace(year=today.year) if fmt == "%d/%m" else d
            except ValueError:
                pass
        return None

    def availability(self, args: dict) -> str:
        d = self._date(args.get("date"))
        if d is None:
            return _out({"error": "date as YYYY-MM-DD, or today / tomorrow"})
        people = int(args.get("people") or 2)
        if people > 8:
            return _out({"available": False, "reason": "tables for up to 8 people; larger groups call the manager"})
        full = {"20:00"} if d.weekday() in (4, 5) else set()
        slots = [t for t in ("19:00", "19:30", "20:00", "20:30", "21:00") if t not in full]
        return _out({"date": d.isoformat(), "weekday": d.strftime("%A"), "people": people, "free_times": slots})

    def book(self, args: dict) -> str:
        d = self._date(args.get("date"))
        name, t = str(args.get("name", "")).strip(), str(args.get("time", "")).strip()
        if d is None or not name or not t:
            return _out({"error": "name, date and time are needed"})
        if t == "20:00" and d.weekday() in (4, 5):
            return _out({"error": "20:00 is full that day", "free_times": ["19:00", "19:30", "20:30", "21:00"]})
        code = f"MH-{random.randint(1000, 9999)}"
        with self.lock:
            self.by_code[code] = {"name": name, "date": d.isoformat(), "time": t, "people": int(args.get("people") or 2)}
        return _out({"booked": True, "code": code, **self.by_code[code]})

    def cancel(self, args: dict) -> str:
        code = str(args.get("code", "")).upper().replace(" ", "")
        with self.lock:
            b = self.by_code.pop(code, None)
        return _out({"cancelled": True, "code": code, **b} if b else {"error": f"no booking {code}"})


_BOOK = Bookings()

S = {"type": "string"}
SCENARIOS = {
    "store": Scenario(
        "store", {"en": "Furniture store", "it": "Negozio di arredamento", "fr": "Magasin de meubles"},
        {"en": "Catalogue, orders and returns of a store; it confirms before changing anything.",
         "it": "Catalogo, ordini e resi di un negozio; chiede conferma prima di cambiare qualcosa.",
         "fr": "Catalogue, commandes et retours d'un magasin ; il confirme avant de modifier quoi que ce soit."},
        "", [],
        {"it": ["Avete un divano a tre posti? Quanto costa?", "Vorrei restituire un ordine."],
         "en": ["Do you have a three-seat sofa? How much is it?", "I'd like to return an order."],
         "fr": ["Avez-vous un canapé trois places ? Combien coûte-t-il ?", "Je voudrais retourner une commande."]},
        {}),
    "weather": Scenario(
        "weather", {"en": "Weather & time", "it": "Meteo e orari", "fr": "Météo et heure"},
        {"en": "Live weather and local time anywhere (Open-Meteo); compare two cities.",
         "it": "Meteo e ora locale in tempo reale ovunque (Open-Meteo); confronta due città.",
         "fr": "Météo et heure locale en direct partout (Open-Meteo) ; compare deux villes."},
        ("You are a friendly voice assistant for weather and time. For any question about the weather or the time "
         "in a place, call the tools: get_weather for forecasts, get_local_time for the time. To compare places, "
         "call the tool once per place. Answer with the essentials: temperature, sky, rain chance. Round numbers."),
        [Tool("get_weather", "Current weather and the forecast for today, tomorrow and the day after in a city.",
              {"type": "object", "properties": {"city": S}, "required": ["city"]}, get_weather),
         Tool("get_local_time", "The current local time and date in a city.",
              {"type": "object", "properties": {"city": S}, "required": ["city"]}, get_local_time)],
        {"it": ["Che tempo fa domani a Milano?", "Fa più caldo a Roma o a Londra adesso?", "Che ore sono a Tokyo?"],
         "en": ["What's the weather in New York tomorrow?", "Is it warmer in London or in Rome right now?",
                "What time is it in Tokyo?"],
         "fr": ["Quel temps fera-t-il demain à Paris ?", "Il fait plus chaud à Lyon ou à Montréal ?",
                "Quelle heure est-il à Tokyo ?"]},
        {"it": ["Buongiorno, che tempo fa domani a Milano?", "E a New York, invece?", "Grazie mille, arrivederci!"],
         "en": ["Hi, what's the weather like in New York tomorrow?", "And how about London?", "Thanks, goodbye!"],
         "fr": ["Bonjour, quel temps fera-t-il demain à Paris ?", "Et à Montréal ?", "Merci beaucoup, au revoir !"]}),
    "travel": Scenario(
        "travel", {"en": "Travel helper", "it": "Assistente di viaggio", "fr": "Assistant de voyage"},
        {"en": "Currency rates (ECB), the next public holidays of a country, local time and weather.",
         "it": "Cambi valuta (BCE), prossimi giorni festivi di un paese, ora locale e meteo.",
         "fr": "Taux de change (BCE), prochains jours fériés d'un pays, heure locale et météo."},
        ("You are a travel helper on the phone. Use the tools for every number or date: convert_currency for money, "
         "next_public_holiday for holidays (two-letter country code), get_local_time and get_weather for places. "
         "Chain them when useful, e.g. weather and holidays for a trip. Be brief and concrete."),
        [Tool("convert_currency", "Convert an amount between currencies at today's European Central Bank rate.",
              {"type": "object", "properties": {"amount": {"type": "number"}, "from": S, "to": S},
               "required": ["amount", "from", "to"]}, convert_currency),
         Tool("next_public_holiday", "The next public holidays of a country (ISO two-letter code, e.g. IT, FR, US).",
              {"type": "object", "properties": {"country_code": S}, "required": ["country_code"]},
              next_public_holiday),
         Tool("get_local_time", "The current local time and date in a city.",
              {"type": "object", "properties": {"city": S}, "required": ["city"]}, get_local_time),
         Tool("get_weather", "Current weather and the forecast for the next days in a city.",
              {"type": "object", "properties": {"city": S}, "required": ["city"]}, get_weather)],
        {"it": ["Quanti dollari sono duecento euro?", "Quando è il prossimo giorno festivo in Francia?",
                "Vado a Londra domani: che tempo fa e che ore sono lì?"],
         "en": ["How much is two hundred dollars in euros?", "When is the next public holiday in Japan?",
                "I'm flying to Rome tomorrow: what's the weather and the local time?"],
         "fr": ["Combien font deux cents euros en dollars ?", "Quel est le prochain jour férié en Italie ?",
                "Je pars à New York demain : quel temps et quelle heure là-bas ?"]},
        {"it": ["Ciao, sto organizzando un viaggio a New York, quanti dollari sono cinquecento euro?",
                "E che tempo farà là domani?", "Perfetto, grazie!"],
         "en": ["Hi, I'm planning a trip to Rome, how much is five hundred dollars in euros?",
                "And what will the weather be like there tomorrow?", "Great, thank you!"],
         "fr": ["Bonjour, je prépare un voyage à Rome, combien font cinq cents dollars en euros ?",
                "Et quel temps fera-t-il là-bas demain ?", "Parfait, merci !"]}),
    "knowledge": Scenario(
        "knowledge", {"en": "Knowledge & numbers", "it": "Conoscenza e calcoli", "fr": "Savoir et calculs"},
        {"en": "Short answers from Wikipedia in your language, and exact calculations.",
         "it": "Risposte brevi da Wikipedia nella tua lingua e calcoli esatti.",
         "fr": "Réponses courtes tirées de Wikipédia dans ta langue, et calculs exacts."},
        ("You are a curious, precise voice assistant. For facts about people, places or things, call "
         "wikipedia_summary with the topic and the caller's language code (it, en, fr, ...) and answer in two "
         "sentences from its summary. For any arithmetic, call calculate with a plain expression (e.g. 2450*0.18); "
         "never compute in your head."),
        [Tool("wikipedia_summary", "The opening summary of the Wikipedia article on a topic, in a given language.",
              {"type": "object", "properties": {"topic": S, "language": S}, "required": ["topic", "language"]},
              wikipedia_summary),
         Tool("calculate", "Evaluate an arithmetic expression with + - * / ** ( ) and decimals.",
              {"type": "object", "properties": {"expression": S}, "required": ["expression"]}, calculate)],
        {"it": ["Chi era Leonardo da Vinci?", "Quanto fa il diciotto per cento di duemilaquattrocentocinquanta?",
                "Cos'è la Torre di Pisa?"],
         "en": ["Who was Ada Lovelace?", "What is fifteen percent of three hundred and twenty?",
                "What is the Golden Gate Bridge?"],
         "fr": ["Qui était Marie Curie ?", "Combien font dix-huit pour cent de deux mille quatre cent cinquante ?",
                "Qu'est-ce que le Mont-Saint-Michel ?"]},
        {"it": ["Ciao, chi era Leonardo da Vinci?", "Quanto fa trecentoquaranta per dodici?", "Grazie, ciao!"],
         "en": ["Hi, who was Ada Lovelace?", "What's three hundred forty times twelve?", "Thanks, bye!"],
         "fr": ["Bonjour, qui était Marie Curie ?", "Combien font trois cent quarante fois douze ?",
                "Merci, au revoir !"]}),
    "restaurant": Scenario(
        "restaurant", {"en": "Restaurant booking", "it": "Prenotazione ristorante", "fr": "Réservation restaurant"},
        {"en": "Books a table in several turns, confirms before booking, gives a booking code; can cancel.",
         "it": "Prenota un tavolo in più turni, chiede conferma prima di prenotare, dà un codice; può annullare.",
         "fr": "Réserve une table en plusieurs tours, confirme avant de réserver, donne un code ; peut annuler."},
        ("You are the voice assistant of Trattoria Mynah, a restaurant. To book, you need the date, the time, the "
         "number of people and a name: ask for what is missing, one thing at a time. Check availability first; "
         "before booking, repeat the details and ask for confirmation; then book and give the booking code. To "
         "cancel, ask for the code. Dates for the tools: YYYY-MM-DD, or today / tomorrow."),
        [Tool("check_availability", "Free times for a table on a date for a number of people.",
              {"type": "object", "properties": {"date": S, "people": {"type": "integer"}},
               "required": ["date", "people"]}, _BOOK.availability),
         Tool("book_table", "Book a table (only after the caller confirmed the details).",
              {"type": "object", "properties": {"name": S, "date": S, "time": S, "people": {"type": "integer"}},
               "required": ["name", "date", "time", "people"]}, _BOOK.book),
         Tool("cancel_booking", "Cancel a booking by its code.",
              {"type": "object", "properties": {"code": S}, "required": ["code"]}, _BOOK.cancel)],
        {"it": ["Vorrei prenotare un tavolo per quattro domani sera.", "Avete posto sabato alle otto?",
                "Devo annullare una prenotazione."],
         "en": ["I'd like a table for four tomorrow evening.", "Do you have room on Saturday at eight?",
                "I need to cancel a booking."],
         "fr": ["Je voudrais réserver une table pour quatre demain soir.", "Vous avez de la place samedi à vingt heures ?",
                "Je dois annuler une réservation."]},
        {"it": ["Buonasera, vorrei prenotare un tavolo per quattro persone domani alle otto e mezza.",
                "A nome Rossi.", "Sì, confermo, grazie!"],
         "en": ["Good evening, I'd like a table for four people tomorrow at eight thirty.", "The name is Smith.",
                "Yes, please confirm, thank you!"],
         "fr": ["Bonsoir, je voudrais une table pour quatre personnes demain à vingt heures trente.",
                "Au nom de Martin.", "Oui, je confirme, merci !"]}),
}

TOOL_RULE = {
    "it": ("Quando la risposta richiede dati che non hai (meteo, orari, cambi, fatti, disponibilità), chiama prima lo "
           "strumento adatto. Non dire mai che stai cercando o controllando senza chiamare uno strumento."),
    "fr": ("Quand la réponse demande des données que tu n'as pas (météo, heure, taux, faits, disponibilités), appelle "
           "d'abord l'outil adapté. Ne dis jamais que tu cherches ou vérifies sans appeler un outil."),
    "en": ("When the answer needs data you do not have (weather, time, rates, facts, availability), call the right "
           "tool first. Never say you are searching or checking without calling a tool."),
}


def get(sid: str | None) -> Scenario:
    return SCENARIOS.get(sid or "store", SCENARIOS["store"])


def public() -> list:
    return [s.public() for s in SCENARIOS.values()]


def timed(fn):
    """A handler that reports its own wall time and turns network errors into a tool error for the model."""
    def h(args):
        t0 = time.monotonic()
        try:
            return fn(args)
        except Exception as e:  # noqa: BLE001 - a failing API is data for the model ("I could not reach ...")
            return _out({"error": f"{type(e).__name__}: {str(e)[:120]}", "ms": round((time.monotonic() - t0) * 1e3)})
    return h
