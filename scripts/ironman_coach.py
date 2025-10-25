"""Interactive Ironman training coach.

This module provides an interactive command line assistant that helps
athletes plan their preparation for an Ironman competition. The
implementation follows the requirements from the project brief: the user
can look up an Ironman race, describe their current fitness and goal,
receive a suggested preparation time, and generate a week-by-week
training plan similar to the sample schedule that was provided.

The script intentionally has no external dependencies besides the Python
standard library so it can run in constrained environments.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class RaceOption:
    """Representation of a race returned by the Ironman API."""

    title: str
    location: Optional[str]
    url: Optional[str]
    start_date: Optional[str]


@dataclass
class AthleteProfile:
    """Information about the athlete that is relevant for planning."""

    swim_minutes_3800: float
    bike_minutes_180: float
    run_minutes_marathon: float
    weekly_training_hours: float
    strength_training: bool
    experience_level: str  # "beginner", "intermediate", "advanced"


@dataclass
class TargetExpectations:
    """Expected finishing targets for the athlete."""

    category: str
    swim_minutes_3800: float
    bike_minutes_180: float
    run_minutes_marathon: float
    total_minutes: float


IRONMAN_ENDPOINT_CANDIDATES = (
    "https://www.ironman.com/api/races?search={query}",
    "https://www.ironman.com/api/races-search?eventLocation={query}",
    "https://www.ironman.com/api/races-search?query={query}",
    "https://www.ironman.com/api/search?type=race&query={query}",
)


STATIC_EVENT_DATA: Sequence[RaceOption] = (
    RaceOption(
        title="IRONMAN World Championship Nice",
        location="Nice, Frankreich",
        url="https://www.ironman.com/im-world-championship",
        start_date="2024-09-22",
    ),
    RaceOption(
        title="IRONMAN Kalmar",
        location="Kalmar, Schweden",
        url="https://www.ironman.com/im-kalmar",
        start_date="2024-08-17",
    ),
    RaceOption(
        title="IRONMAN Hamburg",
        location="Hamburg, Deutschland",
        url="https://www.ironman.com/im-hamburg",
        start_date="2024-08-18",
    ),
    RaceOption(
        title="IRONMAN Barcelona",
        location="Calella, Spanien",
        url="https://www.ironman.com/im-barcelona",
        start_date="2024-10-06",
    ),
    RaceOption(
        title="IRONMAN Florida",
        location="Panama City Beach, USA",
        url="https://www.ironman.com/im-florida",
        start_date="2024-11-02",
    ),
)


WeekPlan = List[Dict[str, object]]


def _make_week(entries: Sequence[Dict[str, object]]) -> WeekPlan:
    return [copy.deepcopy(item) for item in entries]


WEEK_TEMPLATES: Dict[Tuple[str, str], WeekPlan] = {}


def _register_template(phase: str, load: str, entries: Sequence[Dict[str, object]]):
    WEEK_TEMPLATES[(phase, load)] = _make_week(entries)

# ---------------------------------------------------------------------------
# Training plan templates derived from the provided specification
# ---------------------------------------------------------------------------
_register_template(
    "Grundlagenphase 1",
    "W1",
    [
        {"Tag": "Montag", "Training_1": "P"},
        {
            "Tag": "Dienstag",
            "Training_1": "L",
            "Training_2_Alternative": "K",
            "Details": {
                "L": "Locker 10 min Einlaufen, Koordinationsprogramm (4 Übungen je 2×), Kern: 4×(9 min locker + 30 s Tempo + 30 s schnell, nicht Sprint), 10 min Auslaufen",
                "K": "Muskelaufbau",
            },
        },
        {
            "Tag": "Mittwoch",
            "Training_1": "L",
            "Details": {
                "L": "Locker 60 min Dauerlauf, dabei 4–8×8 Froschsprünge einbauen",
            },
        },
        {
            "Tag": "Donnerstag",
            "Training_1": "S",
            "Details": {
                "S": "2× 35 min GA1, alle 200 m für 20 m Sprint, mal mit Abstoßen, mal ohne; kurze Dehnpause zwischen den Intervallen",
            },
        },
        {
            "Tag": "Freitag",
            "Training_1": "S",
            "Details": {
                "S": "Locker Ein- und Ausschwimmen je mind. 10 min; Kern: 10–16× 100 m (je 50 m langsam steigernd bis schnell, 15 m Sprint, Rest sehr locker); Pausen bis zur Erholung (ca. 45 s–1 min)",
            },
        },
        {
            "Tag": "Samstag",
            "Training_1": "R",
            "Details": {
                "R": "2–3 h GA1 Dauerbelastung bei 80–100 U/min, gelegentlich aufstehen, kleiner Gang",
            },
        },
        {
            "Tag": "Sonntag",
            "Training_1": "L",
            "Details": {
                "L": "Langer Lauf GA1 mind. 120–150 min",
            },
        },
    ],
)

_register_template(
    "Grundlagenphase 1",
    "W2",
    [
        {"Tag": "Montag", "Training_1": "P", "Kommentar": "Ruhetag / Pause"},
        {
            "Tag": "Dienstag",
            "Training_1": "L",
            "Training_2_Alternative": "K",
            "Details": {
                "L": "Locker 10 min Einlaufen, Koordinationsprogramm (4 Übungen je 2× sauber ausführen), Kern: 4×(9 min locker + 30 s Tempo aufnehmen + 30 s Sprint) – alles am Stück, 10 min locker Auslaufen",
                "K": "Muskelaufbau",
            },
        },
        {
            "Tag": "Mittwoch",
            "Training_1": "L",
            "Details": {
                "L": "Lockerer Dauerlauf 60 min, dabei 4–8× 8 Froschsprünge einbauen",
            },
        },
        {
            "Tag": "Donnerstag",
            "Training_1": "S",
            "Details": {
                "S": "2× 35 min GA1, trotzdem auf gute Körperspannung achten; alle 200 m für 20 m Sprint, mal mit Abstoßen, mal ohne; kurze Dehnpause zwischen den Intervallen",
            },
        },
        {
            "Tag": "Freitag",
            "Training_1": "S",
            "Details": {
                "S": "Locker Ein- und Ausschwimmen je mind. 10 min; Kern: 10–16× 100 m (je 50 m langsam steigernd bis schnell, 15 m Sprint, Rest sehr locker); Pausen bis zur Erholung (ca. 45 s–1 min)",
            },
        },
        {
            "Tag": "Samstag",
            "Training_1": "R",
            "Training_2_Alternative": "S",
            "Details": {
                "R": "2–3 h GA1 Dauerbelastung bei 80–100 U/min, alle 10 min für 20 s Sprint",
                "S": "Locker je 10 min Ein-/Ausschwimmen; Kern: 4× 400 m GA1/2 (je 200 m Kraul, 100 m Rücken, 100 m Brust)",
            },
        },
        {
            "Tag": "Sonntag",
            "Training_1": "L",
            "Details": {
                "L": "Langer Lauf GA1 mind. 90–120 min, alle 10 min für 20 s leicht steigern",
            },
        },
    ],
)

_register_template(
    "Grundlagenphase 1",
    "W3",
    [
        {"Tag": "Montag", "Training_1": "P"},
        {
            "Tag": "Dienstag",
            "Training_1": "L",
            "Training_2_Alternative": "K",
            "Details": {
                "L": "Locker 10 min Einlaufen, Koordinationsprogramm (4 Übungen je 2×), Kern: 4×(9 min locker + 30 s Tempo + 30 s Sprint), alles am Stück, 10 min Auslaufen",
                "K": "Muskelaufbau",
            },
        },
        {
            "Tag": "Mittwoch",
            "Training_1": "L",
            "Details": {
                "L": "Locker 60 min Dauerlauf, 4–8× 8 Froschsprünge",
            },
        },
        {
            "Tag": "Donnerstag",
            "Training_1": "S",
            "Details": {
                "S": "2× 35 min GA1 mit Körperspannung; alle 200 m 20 m Sprint",
            },
        },
        {
            "Tag": "Freitag",
            "Training_1": "S",
            "Details": {
                "S": "Ein- und Ausschwimmen je 10 min; Kern: 10–16×100 m (50 m steigernd, 15 m Sprint, locker), Pause 45 s–1 min",
            },
        },
        {
            "Tag": "Samstag",
            "Training_1": "R",
            "Training_2_Alternative": "S",
            "Details": {
                "R": "2–3 h GA1 bei 80–100 rpm, alle 8 min 20 s Sprint",
                "S": "Ein- und Ausschwimmen je 10 min; Kern: 4×600 m GA1/2 (200 m Kraul, 100 m Rücken, 100 m Brust)",
            },
        },
        {
            "Tag": "Sonntag",
            "Training_1": "L",
            "Details": {
                "L": "Langer Lauf GA1 90–120 min, alle 1 min auf kurzen Bodenkontakt achten",
            },
        },
    ],
)

_register_template(
    "Grundlagenphase 1",
    "Wreg",
    [
        {
            "Tag": "Montag",
            "Training_1": "R",
            "Details": {"R": "Locker rollen 60–75 min"},
        },
        {
            "Tag": "Dienstag",
            "Training_1": "P",
            "Training_2_Alternative": "K",
            "Details": {"K": "Leichtes Krafttraining, Schwerpunkt Dehnen/Wellness"},
        },
        {
            "Tag": "Mittwoch",
            "Training_1": "S",
            "Details": {"S": "Locker 45 min, Lagenwechsel, Technikprogramm"},
        },
        {"Tag": "Donnerstag", "Training_1": "P"},
        {
            "Tag": "Freitag",
            "Training_1": "S",
            "Details": {
                "S": "Ein- und Ausschwimmen je 10 min; Kern: 10–16×100 m (50 m steigernd, 25 m halten, locker), Pause 45 s–1 min",
            },
        },
        {
            "Tag": "Samstag",
            "Training_1": "R",
            "Details": {"R": "1–2 h GA1 bei 80–100 rpm"},
        },
        {
            "Tag": "Sonntag",
            "Training_1": "L",
            "Details": {"L": "GA1-Lauf 60 min regenerativ"},
        },
    ],
)

_register_template(
    "Grundlagenphase 2",
    "Wreg",
    [
        {"Tag": "Montag", "Training_1": "P"},
        {"Tag": "Dienstag", "Training_1": "P"},
        {
            "Tag": "Mittwoch",
            "Training_1": "K",
            "Details": {"K": "Maximalkraft + Rumpfstabi"},
        },
        {
            "Tag": "Donnerstag",
            "Training_1": "R",
            "Details": {"R": "75 min GA1 rollen, 4×30 s Steigerung im Stehen"},
        },
        {
            "Tag": "Freitag",
            "Training_1": "L",
            "Details": {"L": "Technischer Dauerlauf mit Koordinationsübungen GA1/2"},
        },
        {
            "Tag": "Samstag",
            "Training_1": "S",
            "Details": {
                "S": "Ein- und Ausschwimmen je 1000 m mit Frequenzwechseln, 4–8×100 m mit Paddles GA2",
            },
        },
        {"Tag": "Sonntag", "Training_1": "P"},
    ],
)

_register_template(
    "Grundlagenphase 2",
    "W1",
    [
        {
            "Tag": "Montag",
            "Training_1": "L",
            "Details": {
                "L": "20 min Einlaufen, Koordinationsprogramm (je Übung 2×), 50 min GA2, Koordination (1× schnell), 10 min Auslaufen",
            },
        },
        {"Tag": "Dienstag", "Training_1": "P"},
        {
            "Tag": "Mittwoch",
            "Training_1": "R",
            "Training_2_Alternative": "K",
            "Details": {
                "R": "20 min Einrollen, 3×8 min GA2 dicker Gang, 3 min Pause, locker Ausrollen",
                "K": "Maximalkraft + Rumpfstabi",
            },
        },
        {
            "Tag": "Donnerstag",
            "Training_1": "L",
            "Details": {
                "L": "10 min Einlaufen, 4×30 s Steigerung, 9×45 s Bergsprints, 5 min GA2, Auslaufen; Alternative: 9×(1 min Abfahrtshocke + 20 Ausfallschritte + 20–30 Springschritte)",
            },
        },
        {
            "Tag": "Freitag",
            "Training_1": "S",
            "Details": {
                "S": "200 m Einlaufen, Koordination, 6×50 m steigernd, 2000 m Wechseltempo GA1/2, 4–6×100 m langer Zug GA2, Pause 45 s, Auslaufen",
            },
        },
        {
            "Tag": "Samstag",
            "Training_1": "L",
            "Training_2_Alternative": "S",
            "Details": {
                "L": "2–2.5 h GA1, alle 15 min 10–16 Ausfallschritte",
                "S": "Ein- und Ausschwimmen je 200–400 m, 12×100 m Paddles GA2",
            },
        },
        {
            "Tag": "Sonntag",
            "Training_1": "R",
            "Details": {
                "R": "3 h (45 min GA1 + 15 min GA2), alle 5 min 20 s Aufstehen",
            },
        },
    ],
)

_register_template(
    "Grundlagenphase 2",
    "W2",
    [
        {
            "Tag": "Montag",
            "Training_1": "L",
            "Details": {
                "L": "20 min Einlaufen, Koordination (2×), 50 min GA2, Koordination (1× schnell), 10 min Auslaufen",
            },
        },
        {"Tag": "Dienstag", "Training_1": "P"},
        {
            "Tag": "Mittwoch",
            "Training_1": "R",
            "Training_2_Alternative": "K",
            "Details": {
                "R": "20 min Einrollen, 4×8 min GA2 dicker Gang, 3 min Pause, Ausrollen",
                "K": "Kraftausdauer + Rumpfstabi",
            },
        },
        {
            "Tag": "Donnerstag",
            "Training_1": "L",
            "Details": {
                "L": "10 min Einlaufen, 4×30 s Steigerung, 9×45 s Bergsprints, 5 min GA2, Auslaufen; Alternative: 9×(1 min Abfahrtshocke + 20 Ausfallschritte + 20–30 Springschritte)",
            },
        },
        {
            "Tag": "Freitag",
            "Training_1": "S",
            "Details": {
                "S": "200 m Einlaufen, Koordination, 6×50 m steigernd, 2000 m Wechseltempo GA1/2, 4–6×100 m langer Zug GA2, Pause 45 s, Auslaufen",
            },
        },
        {
            "Tag": "Samstag",
            "Training_1": "L",
            "Training_2_Alternative": "S",
            "Details": {
                "L": "2–2.5 h GA1, alle 15 min 10–16 Ausfallschritte",
                "S": "Ein- und Ausschwimmen je 200–400 m, 16×100 m Paddles im Wechsel, GA2",
            },
        },
        {
            "Tag": "Sonntag",
            "Training_1": "R",
            "Details": {
                "R": "3 h (45 min GA1 + 15 min GA2), alle 5 min 20 s Aufstehen",
            },
        },
    ],
)

_register_template(
    "Grundlagenphase 2",
    "W3",
    [
        {
            "Tag": "Montag",
            "Training_1": "L",
            "Details": {
                "L": "20 min Einlaufen, Koordinationsprogramm (je Übung 2×), Kern: 50 min GA2-Dauerlauf; danach Koordinationsprogramm (1×, schnell), 10 min Auslaufen",
            },
        },
        {"Tag": "Dienstag", "Training_1": "P"},
        {
            "Tag": "Mittwoch",
            "Training_1": "R",
            "Training_2_Alternative": "K",
            "Details": {
                "R": "20 min Einrollen, 5×8 min GA2 dicker Gang (letzte Minute im Stehen), je 3 min Rollen; locker Ausrollen",
                "K": "Kraftausdauer + Rumpfstabi (Rad- und Krafteinheit trennen oder ausreichend pausieren)",
            },
        },
        {
            "Tag": "Donnerstag",
            "Training_1": "L",
            "Details": {
                "L": "10 min Einlaufen, 4×30 s Steigerung; 9× ~45 s steil bergauf (intensiv abdrücken), Pausen bis Erholung; anschließend 5 min GA2 mit kurzem Bodenkontakt; Auslaufen. Alternative: 9×(1 min Abfahrtshocke + 20 weite Ausfallschritte + 20–30 Springschritte)",
            },
        },
        {
            "Tag": "Freitag",
            "Training_1": "S",
            "Details": {
                "S": "200 m Einlaufen, Koordinationsprogramm, 6×50 m steigernd, Kern: 2000 m Wechseltempo GA1/2 alle 50 m; 4–6×100 m langer Zug GA2 (Züge zählen und reduzieren), je 45 s Pause; Auslaufen",
            },
        },
        {
            "Tag": "Samstag",
            "Training_1": "L",
            "Training_2_Alternative": "S",
            "Details": {
                "L": "Langer Dauerlauf 2–2.5 h GA1/2; alle 15 min 10–16 Ausfallschritte",
                "S": "Ein- und Ausschwimmen je 200–400 m; 8–10×200 m Paddles/ohne im Wechsel im oberen GA2; Züge zählen, gleichmäßige Zeiten",
            },
        },
        {
            "Tag": "Sonntag",
            "Training_1": "R",
            "Details": {
                "R": "3–4 h Dauerbelastung: je 45 min GA1 (kleiner Gang) + 15 min GA2 mit Kraft; alle 5 min 20 s Aufstehen",
            },
        },
    ],
)

_register_template(
    "Aufbauphase 1",
    "Wreg",
    _make_week(WEEK_TEMPLATES[("Grundlagenphase 2", "Wreg")]),
)

_register_template(
    "Aufbauphase 1",
    "W1",
    [
        {
            "Tag": "Dienstag",
            "Training_1": "L",
            "Details": {
                "L": "20 min Einlaufen; Kern: 10×8 Froschsprünge explosiv (je 2 min Trabpause); anschließend 10 min zügig GA2 und 10 min locker Auslaufen",
            },
        },
        {
            "Tag": "Donnerstag",
            "Training_1": "K",
            "Details": {"K": "Schnellkraft am Gerät + Rumpfstabi"},
        },
        {
            "Tag": "Freitag",
            "Training_1": "S",
            "Details": {
                "S": "1500 m Einlaufen (Wechseltempo alle 50 m GA1/GA2); Kern: 20×25 m Sprint (+25 m locker), jeder 3./4. mit Paddles, je 25 s Pause; 1000 m Auslaufen",
            },
        },
        {
            "Tag": "Samstag",
            "Training_1": "L",
            "Training_2_Alternative": "R",
            "Details": {
                "L": "20–30 km GA1; alle 10 min 1 min bis EB steigern",
                "R": "20 min Einrollen; Kern: 16×10 s gegen hohen Widerstand sprinten; 2 min Pause (Rollen); 10 min Ausrollen",
            },
        },
        {
            "Tag": "Sonntag",
            "Training_1": "KE",
            "Details": {
                "KE": "4–6 h Koppeleinheit R+L in GA1 bis GA2 (mehrheitlich GA1); Wechsel <10 min; mind. 20 km Lauf; alle 10–15 min 20 s antreten.",
            },
        },
    ],
)

_register_template(
    "Aufbauphase 1",
    "W2",
    [
        {"Tag": "Montag", "Training_1": "P"},
        {"Tag": "Dienstag", "Training_1": "P"},
        {
            "Tag": "Mittwoch",
            "Training_1": "K",
            "Details": {"K": "Explosivkraft am Gerät + Rumpfstabi (gut aufwärmen)"},
        },
        {
            "Tag": "Donnerstag",
            "Training_1": "L",
            "Details": {
                "L": "Tempoarbeit: 20 min Einlaufen; Kern: 5×2 km (1. Hälfte knapp unter IANS, 2. Hälfte knapp über IANS), 4 min Trabpausen; locker Auslaufen",
            },
        },
        {
            "Tag": "Freitag",
            "Training_1": "S",
            "Details": {
                "S": "800 m locker Einlaufen; Kern: 5×400 m (200 m GA1 + 100 m GA2 + 100 m Tempo); 600 m locker Auslaufen; 4×50 m Steigerungen",
            },
        },
        {
            "Tag": "Samstag",
            "Training_1": "R",
            "Training_2_Alternative": "S",
            "Details": {
                "R": "Locker Einrollen; Kern: 4×15 min Aeroposition im Bereich IANS; je 5 min Pause fast passiv; 20 min locker Ausrollen",
                "S": "2000 m regenerativ-technisch",
            },
        },
        {
            "Tag": "Sonntag",
            "Training_1": "KE",
            "Details": {
                "KE": "4–6 h Koppeleinheit R+L in GA1 bis GA2 zu etwa gleichen Anteilen; Wechsel <10 min; mind. 20 km Lauf; beim L/R alle 10–15 min 20 s antreten.",
            },
        },
    ],
)

_register_template(
    "Aufbauphase 1",
    "W3",
    [
        {
            "Tag": "Dienstag",
            "Training_1": "L",
            "Details": {"L": "5×2 km: 1. km zügig GA2, 2. km IANS; dazwischen 3 min Pause passiv"},
        },
        {
            "Tag": "Donnerstag",
            "Training_1": "S",
            "Training_2_Alternative": "K",
            "Details": {
                "S": "(optional) 2000 m regenerativ-technisch",
                "K": "Schnellkraft am Gerät + Rumpfstabi",
            },
        },
        {
            "Tag": "Samstag",
            "Training_1": "R",
            "Training_2_Alternative": "L",
            "Details": {
                "R": "Locker Einrollen; Kern: 4×15 min Aeroposition im Bereich IANS; je 5 min Pause fast passiv; 20 min locker Ausrollen",
                "L": "Regenerativer Lauf um 10 km",
            },
        },
        {
            "Tag": "Sonntag",
            "Training_1": "KE",
            "Details": {
                "KE": "4–6 h Koppeleinheit R+L in GA1 bis EB zu etwa gleichen Anteilen; Wechsel <10 min; mind. 20 km Lauf; beim L/R alle 10–15 min 20 s antreten.",
            },
        },
    ],
)

_register_template(
    "Aufbauphase 2",
    "Wreg",
    [
        {"Tag": "Montag", "Training_1": "P", "Kommentar": "Woche frei; nur Training bei Spaß und Motivation (z. B. Sauna, Wandern, technisches Schwimmen)"},
        {"Tag": "Samstag", "Training_1": "K", "Details": {"K": "Rumpfstabi"}},
    ],
)

_register_template(
    "Aufbauphase 2",
    "W1",
    [
        {
            "Tag": "Dienstag",
            "Training_1": "K",
            "Details": {"K": "Rumpfstabi"},
        },
        {
            "Tag": "Mittwoch",
            "Training_1": "L",
            "Details": {"L": "16–20 km GA1-Dauerlauf"},
        },
        {"Tag": "Donnerstag", "Training_1": "P"},
        {
            "Tag": "Freitag",
            "Training_1": "S",
            "Details": {
                "S": "1000 m Einlaufen + Technik; 2×4×200 m Paddles GA2 (30 s Pause), zwischen Blöcken 100 m locker, 400 m gleiten, Auslaufen",
            },
        },
        {
            "Tag": "Samstag",
            "Training_1": "R",
            "Training_2_Alternative": "K",
            "Details": {
                "R": "2×15 min Aeroposition IANS + 5 min Pause passiv + 20 min Ausrollen",
                "K": "Rumpfstabi",
            },
        },
        {
            "Tag": "Sonntag",
            "Training_1": "KE",
            "Details": {
                "KE": "4–6 h Koppeleinheit R + L GA1–GA2 (gleiche Anteile); Wechsel <10 min; mind. 20 km Lauf; alle 10–15 min 20 s antreten",
            },
        },
    ],
)

_register_template(
    "Aufbauphase 2",
    "W2",
    [
        {
            "Tag": "Montag",
            "Training_1": "S",
            "Details": {"S": "3800 m Wechseltempo (GA2, jeder 4. 100 m EB bei erhöhter Frequenz)"},
        },
        {
            "Tag": "Dienstag",
            "Training_1": "K",
            "Details": {"K": "Rumpfstabi"},
        },
        {
            "Tag": "Mittwoch",
            "Training_1": "L",
            "Details": {"L": "Tempoarbeit 5×3 km (2 km unter IANS + 1 km über IANS), 4 min Trabpause"},
        },
        {"Tag": "Donnerstag", "Training_1": "P"},
        {
            "Tag": "Freitag",
            "Training_1": "S",
            "Details": {"S": "Optional 2000 m regenerativ technisch"},
        },
        {
            "Tag": "Samstag",
            "Training_1": "R",
            "Training_2_Alternative": "K",
            "Details": {
                "R": "3×25 min Aeroposition IANS + 5 min Pause passiv + 20 min Ausrollen",
                "K": "Regenerativer Lauf 5–8 km + Rumpfstabi",
            },
        },
        {
            "Tag": "Sonntag",
            "Training_1": "KE",
            "Details": {
                "KE": "6–8 h Koppeleinheit R + L GA1–GA2 (50/50); Wechsel <10 min; mind. 30 km Lauf; Ernährung optimieren",
            },
        },
    ],
)

_register_template(
    "Aufbauphase 2",
    "W3",
    [
        {
            "Tag": "Montag",
            "Training_1": "S",
            "Details": {"S": "1000 m Einlaufen; 12×100 m (75 m schnell + 25 m Sprint, 45 s Pause), Auslaufen"},
        },
        {
            "Tag": "Dienstag",
            "Training_1": "K",
            "Details": {"K": "Rumpfstabi"},
        },
        {
            "Tag": "Mittwoch",
            "Training_1": "L",
            "Details": {"L": "4×4 km (2 km unter IANS, 1 km an, 1 km über IANS), 4 min Trabpause"},
        },
        {"Tag": "Donnerstag", "Training_1": "P"},
        {
            "Tag": "Freitag",
            "Training_1": "L",
            "Details": {"L": "20 km Fahrtspiel zwischen GA1 und IANS nach Tagesverfassung"},
        },
        {
            "Tag": "Samstag",
            "Training_1": "R",
            "Training_2_Alternative": "K",
            "Details": {
                "R": "4×20 min Aeroposition IANS + 5 min Pause passiv + 20 min Ausrollen",
                "K": "Regenerativer Lauf 5–8 km + Rumpfstabi",
            },
        },
        {
            "Tag": "Sonntag",
            "Training_1": "KE",
            "Details": {
                "KE": "6–8 h Koppeleinheit R + L GA1–GA2 (50/50); Wechsel <10 min; mind. 30 km Lauf; Ernährung optimieren",
            },
        },
    ],
)

_register_template(
    "Spezifische Phase",
    "W3_race",
    [
        {
            "Tag": "Montag",
            "Training_1": "S",
            "Kommentar": "Mitteldistanz (Wettkampf in Woche 9 bis 7)",
        },
        {
            "Tag": "Dienstag",
            "Training_1": "K",
            "Training_2_Alternative": "S",
            "Details": {
                "K": "Rumpfstabi",
                "S": "1.5 km regenerativ",
            },
        },
        {
            "Tag": "Mittwoch",
            "Training_1": "L",
            "Details": {"L": "4×4 km (2 km unter IANS, 1 km an, 1 km über IANS), 4 min Trabpause"},
        },
        {"Tag": "Donnerstag", "Training_1": "P"},
        {
            "Tag": "Freitag",
            "Training_1": "L",
            "Training_2_Alternative": "S",
            "Details": {
                "L": "20 km Fahrtspiel zwischen GA1 und IANS",
                "S": "2 km regenerativ",
            },
        },
        {
            "Tag": "Samstag",
            "Training_1": "R",
            "Details": {
                "R": "200 km auf Zug (GA2/IANS); Ernährung optimieren",
            },
        },
        {
            "Tag": "Sonntag",
            "Training_1": "KE",
            "Details": {
                "KE": "80 min R regenerativ + 40 min L locker",
            },
        },
    ],
)

_register_template(
    "Spezifische Phase",
    "W3",
    [
        {
            "Tag": "Montag",
            "Training_1": "S",
            "Details": {"S": "1000 m Einlaufen + Technik, 10×50 m Sprint max. (1 min Pause), 200 m Test, Auslaufen"},
        },
        {
            "Tag": "Dienstag",
            "Training_1": "K",
            "Details": {"K": "Rumpfstabi"},
        },
        {
            "Tag": "Mittwoch",
            "Training_1": "L",
            "Details": {"L": "4×5 km (2 km unter IANS, 2 km an, 1 km über IANS), 4 min Trabpause"},
        },
        {"Tag": "Donnerstag", "Training_1": "P"},
        {
            "Tag": "Freitag",
            "Training_1": "S",
            "Details": {"S": "Warm-up außerhalb; 100 m Ein; 3×20 min auf Zug (GA2–EB), 5 min Pause, Auslaufen"},
        },
        {
            "Tag": "Samstag",
            "Training_1": "L",
            "Training_2_Alternative": "K",
            "Details": {
                "L": "30 km GA1/2 + 6 km Endbeschleunigung bis Zieltempo",
                "K": "Kurze Rumpfstabi-Einheit",
            },
        },
        {
            "Tag": "Sonntag",
            "Training_1": "KE",
            "Details": {
                "KE": "1 h L GA2 + 1 h R IANS (Aeroposition) + 1 h L GA2 + 20 min R IANS + 40 min R Ausrollen; Ernährung optimieren",
            },
        },
    ],
)

_register_template(
    "Spezifische Phase",
    "W3_peak",
    [
        {
            "Tag": "Montag",
            "Training_1": "S",
            "Details": {
                "S": "1500 m locker bis zügig Ein, 6×100 m Paddles GA2 gleiten, 200 m locker, 6×100 m Paddles hohe Frequenz im EB, je 30 s Pause, 10 min Aus",
            },
        },
        {
            "Tag": "Dienstag",
            "Training_1": "R",
            "Training_2_Alternative": "K",
            "Details": {
                "R": "15 min Einrollen, 4×(4×(20 s Vollsprint + 40 s erholen) mit 5 min lockerer Pause), 15 min Aus",
                "K": "Identisch wie Radtraining oder ergänzend Rumpfstabi",
            },
        },
        {
            "Tag": "Mittwoch",
            "Training_1": "L",
            "Details": {"L": "12 km GA2 + 4 km IANS"},
        },
        {"Tag": "Donnerstag", "Training_1": "P"},
        {
            "Tag": "Freitag",
            "Training_1": "S",
            "Details": {
                "S": "Warm-up außerhalb; 300 m Ein, 8×(25 m Tauchen + 25 m Sprint), 12× 100 m Wettkampftempo (15 s Pause), Orientierungsschwimmen im See (Punkt fixieren), Auslaufen",
            },
        },
        {
            "Tag": "Samstag",
            "Training_1": "L",
            "Training_2_Alternative": "K",
            "Details": {
                "L": "30 km GA1/2 + 6 km Endbeschleunigung bis Zieltempo",
                "K": "Kurze Rumpfstabi-Einheit",
            },
        },
        {
            "Tag": "Sonntag",
            "Training_1": "R",
            "Details": {
                "R": "180 km auf Zug (GA2/IANS) + 15 min Lauf Zieltempo, locker Auslaufen",
            },
        },
    ],
)

_register_template(
    "Spezifische Phase",
    "W3_runfocus",
    [
        {
            "Tag": "Montag",
            "Training_1": "S",
            "Details": {"S": "1500 m regenerativ"},
        },
        {
            "Tag": "Dienstag",
            "Training_1": "R",
            "Details": {
                "R": "15 min Einrollen, 4×(4×(20 s Vollsprint + 40 s erholen) mit 5 min Pause), 15 min Aus",
            },
        },
        {
            "Tag": "Mittwoch",
            "Training_1": "L",
            "Details": {"L": "12 km GA2 + 10 km IANS"},
        },
        {"Tag": "Donnerstag", "Training_1": "P"},
        {
            "Tag": "Freitag",
            "Training_1": "S",
            "Details": {
                "S": "300 m Ein, 8×(25 m Tauchen + 25 m Sprint), 12–16× 100 m Wettkampftempo (15 s Pause); Orientierungsschwimmen im See; Auslaufen",
            },
        },
        {
            "Tag": "Samstag",
            "Training_1": "L",
            "Training_2_Alternative": "K",
            "Details": {
                "L": "20 km Tempo-Dauerlauf im Zieltempo",
                "K": "Rumpfstabi",
            },
        },
        {
            "Tag": "Sonntag",
            "Training_1": "KE",
            "Details": {
                "KE": "3.5 h Rad im Zieltempo + 1 h Laufen Zieltempo",
            },
        },
    ],
)

_register_template(
    "Spezifische Phase",
    "Wreg",
    [
        {"Tag": "Montag", "Training_1": "P", "Kommentar": "Woche frei; nur Training bei Spaß und Motivation (z. B. erste Runden im See, Wandern, Erholung)"},
        {"Tag": "Dienstag", "Training_1": "P"},
        {"Tag": "Mittwoch", "Training_1": "P"},
        {"Tag": "Donnerstag", "Training_1": "P"},
        {"Tag": "Freitag", "Training_1": "P"},
        {"Tag": "Samstag", "Training_1": "P"},
        {"Tag": "Sonntag", "Training_1": "P"},
    ],
)

_register_template(
    "Tapering",
    "Wreg",
    [
        {"Tag": "Montag", "Training_1": "P"},
        {
            "Tag": "Dienstag",
            "Training_1": "S",
            "Details": {"S": "1500 m regenerativ; ab sofort möglichst im See"},
        },
        {
            "Tag": "Mittwoch",
            "Training_1": "L",
            "Details": {"L": "60 min locker, Dehnpausen einbauen"},
        },
        {"Tag": "Donnerstag", "Training_1": "P"},
        {
            "Tag": "Freitag",
            "Training_1": "R",
            "Details": {"R": "90 min regenerativ"},
        },
        {
            "Tag": "Samstag",
            "Training_1": "L",
            "Details": {"L": "20 min zügig, locker Auslaufen"},
        },
        {
            "Tag": "Sonntag",
            "Training_1": "WK",
            "Details": {"WK": "Kurztriathlon zum Testen"},
        },
    ],
)

_register_template(
    "Tapering",
    "W2",
    [
        {
            "Tag": "Montag",
            "Training_1": "S",
            "Details": {"S": "2 km regenerativ"},
        },
        {
            "Tag": "Dienstag",
            "Training_1": "L",
            "Training_2_Alternative": "K",
            "Details": {
                "L": "20 min Einlaufen, 4×5 min im angestrebten Marathontempo, Auslaufen",
                "K": "Rumpfstabi",
            },
        },
        {
            "Tag": "Mittwoch",
            "Training_1": "R",
            "Details": {
                "R": "2–3 h locker rollen mit Frequenzwechseln und 6×15 s Steigerungen",
            },
        },
        {"Tag": "Donnerstag", "Training_1": "P"},
        {
            "Tag": "Freitag",
            "Training_1": "R",
            "Training_2_Alternative": "L",
            "Details": {
                "R": "2 h Rad im Wettkampftempo + 10 min Lauf im Wettkampftempo, kurz Auslaufen",
                "L": "10 min locker Auslaufen",
            },
        },
        {
            "Tag": "Samstag",
            "Training_1": "L",
            "Training_2_Alternative": "K",
            "Details": {
                "L": "1 h im angestrebten Wettkampftempo",
                "K": "Rumpfstabi",
            },
        },
        {
            "Tag": "Sonntag",
            "Training_1": "KE",
            "Details": {
                "KE": "3 h locker Rad + 30 min Lauf Wettkampftempo + 30 min locker Rad Ausrollen",
            },
        },
    ],
)

_register_template(
    "Tapering",
    "W1",
    [
        {"Tag": "Montag", "Training_1": "P"},
        {"Tag": "Dienstag", "Training_1": "P"},
        {
            "Tag": "Mittwoch",
            "Training_1": "R",
            "Details": {"R": "90 min im Wettkampftempo"},
        },
        {"Tag": "Donnerstag", "Training_1": "P"},
        {
            "Tag": "Freitag",
            "Training_1": "L",
            "Details": {"L": "20 min Wettkampftempo + 20 min locker + 10 min Wettkampftempo + 10 min Auslaufen – Ironman-Race-Pace!"},
        },
        {
            "Tag": "Samstag",
            "Training_1": "S",
            "Training_2_Alternative": "R",
            "Details": {
                "S": "WK-Simulation zur Wettkampfzeit: Warm-up außerhalb, 1500 m Wettkampfstart, 1500 m Ausgleiten",
                "R": "20 min Einrollen; 4×15 min flüssig bis Wettkampftempo steigern (5 min Rollpausen), nicht überziehen, Ausrollen + Dehnen",
            },
        },
        {
            "Tag": "Sonntag",
            "Training_1": "L",
            "Details": {"L": "45 min regenerativ, 6×100 m Steigerungen, danach intensiv dehnen"},
        },
    ],
)

_register_template(
    "Tapering",
    "WKW",
    [
        {"Tag": "Montag", "Training_1": "P"},
        {"Tag": "Dienstag", "Training_1": "P"},
        {
            "Tag": "Mittwoch",
            "Training_1": "KE",
            "Details": {
                "KE": "60 min Rad GA2; alle 10 min 1 min im Stehen gegen Widerstand beschleunigen; Wechsel zu Lauf: 10 min IANS hohe Frequenz + 10 min GA2 mit kurzem Bodenkontakt + 10 min locker Auslaufen",
            },
        },
        {"Tag": "Donnerstag", "Training_1": "P"},
        {
            "Tag": "Freitag",
            "Training_1": "S",
            "Details": {
                "S": "300 m locker Ein, 3×100 m locker steigernd, kurz Aus; 300 m locker Ein, 3×100 m locker steigernd, kurz Aus",
            },
        },
        {
            "Tag": "Samstag",
            "Training_1": "KE",
            "Details": {
                "KE": "30 min locker Rad (4×20 s leicht steigern) + 20 min locker Lauf (4×20 s locker steigern)",
            },
        },
        {
            "Tag": "Sonntag",
            "Training_1": "WK",
            "Details": {"WK": "Wettkampf – viel Erfolg!"},
        },
    ],
)

LEGEND = {
    "R": "Radtraining",
    "S": "Schwimmen",
    "L": "Laufen",
    "K": "Krafttraining",
    "KE": "Koppeleinheit",
    "WK": "Wettkampf",
    "P": "Pause",
    "LD": "Leistungsdiagnostik",
    "GA1": "Grundlagenausdauer 1 (niedrige Intensität, Fettstoffwechselbereich)",
    "GA2": "Grundlagenausdauer 2 (mittlere Intensität)",
    "EB": "Entwicklungsbereich (höhere Intensität)",
    "Kraftarten": {
        "Max.Kraft": "Maximalkraft",
        "Hypertr.": "Hypertrophie",
        "KA": "Kraftausdauer",
        "Explosiv": "Explosivkraft",
        "Schnell.": "Schnelligkeit",
        "Koord.": "Koordination",
    },
}

INFO_BLOCK = {
    "Vorgaben": {
        "Leistungsdiagnostiken": "Vor Leistungsdiagnostiken immer 2 Ruhetage setzen. 3 bis 4 Leistungsdiagnostiken über den Trainingszeitraum verteilen, ideal an Phasenübergängen.",
        "Wettkämpfe": "Trainingswettkämpfe mit mindestens 3 Tagen verminderter Intensität vor- und nachbereiten.",
        "Pausen": "Pausen bei Bedarf einlegen, um Überlastung zu vermeiden. Drei Tage Pause zur rechten Zeit sind besser als eine Verletzung oder Übertraining.",
        "Trainingslager": "Vor allem in der Aufbauphase 2 wichtig. Schwerpunkte beibehalten, Umfänge leicht erhöhen. GA-Anteile dürfen stärker erhöht werden. Nach dem Trainingslager eine Wreg-Woche einplanen.",
    },
    "Belastungszyklen": {
        "W1": "Einstiegswoche mit niedriger Belastung",
        "W2": "Mittlere Belastung",
        "W3": "Hohe Belastung",
        "Wreg": "Regenerationswoche zur aktiven Erholung",
        "WKW": "Wettkampf-Woche",
    },
    "Definitionen_Ausdauertraining": {
        "GA1": "Locker, aerob, Fettstoffwechseltraining. Beispiel: 30+ Minuten lockerer Dauerlauf.",
        "GA2": "Mittlere Intensität (Mischstoffwechsel). Beispiel: Dauerbelastung im GA2-Bereich.",
        "EB": "Entwicklungsbereich, an der anaeroben Schwelle (IANS). Beispiel: 6×5 min im EB.",
        "Schnelligkeit": "Kurze anaerobe Belastungen, z. B. 6×20 s Sprint.",
        "Schnellkraft": "Explosive, kurze Belastungen. Beispiel: 4×12 Froschsprünge.",
        "Kraftausdauer": "Kraftintensive, ausdauerorientierte Belastungen. Beispiel: 5×5 min Rad dicker Gang im GA2.",
    },
    "Definitionen_Gerätetraining": {
        "Maximalkraft": "5 Sätze à 5–7 Wiederholungen mit maximaler Anstrengung.",
        "Hypertrophie": "Muskelaufbau, 3 Sätze à 8–12 Wiederholungen.",
        "Kraftausdauer": "3 Sätze à 15–25 Wiederholungen zur Ermüdungsresistenz.",
        "Explosivkraft": "Variante der Schnellkraft, 3 Sätze à 10 Wiederholungen, Tempo 4-1.",
        "Schnellkraft": "4 Sätze à 5 Wiederholungen, sehr schnell (Tempo 1-1).",
        "Rumpfstabilisation": "Rumpf-Athletik (z. B. Unterarmstütz längs/seitlich).",
        "Koordination": "Verbesserung des intra-/intermuskulären Zusammenspiels, z. B. Lauf-ABC, Einarmschwimmen.",
    },
    "Rahmenplan": {
        "Hinweis": "Phasenfolge: Grundlagenphase 1–2, Aufbauphase 1–2, Spezifische Phase, Tapering.",
        "Highlights": [
            "2–3× 10 km-Wettkämpfe zwischen Woche 24–18",
            "Halbmarathon in Woche 14–12",
            "Mitteldistanz in Woche 9–7",
            "Langdistanz (Wettkampf) in Woche 1",
        ],
    },
}

# ---------------------------------------------------------------------------
# Event lookup helpers
# ---------------------------------------------------------------------------

def _parse_event_payload(payload: object) -> List[RaceOption]:
    events: List[RaceOption] = []
    if isinstance(payload, dict):
        items = payload.get("items") or payload.get("data") or payload.get("results")
        if isinstance(items, list):
            for item in items:
                events.extend(_parse_event_payload(item))
            return events
        # Sometimes the payload itself is the event
        title = item_title = payload.get("title") or payload.get("name")
        if item_title:
            events.append(
                RaceOption(
                    title=str(item_title),
                    location=payload.get("eventLocation") or payload.get("location") or payload.get("city"),
                    url=payload.get("url") or payload.get("pageUrl") or payload.get("link"),
                    start_date=payload.get("eventStartDate") or payload.get("date") or payload.get("startDate"),
                )
            )
        return events
    if isinstance(payload, list):
        for item in payload:
            events.extend(_parse_event_payload(item))
    return events


def fetch_ironman_events(query: str, limit: int = 10) -> List[RaceOption]:
    query_encoded = urllib.parse.quote_plus(query.strip())
    for endpoint in IRONMAN_ENDPOINT_CANDIDATES:
        url = endpoint.format(query=query_encoded)
        try:
            with urllib.request.urlopen(url, timeout=8) as response:
                if response.status >= 400:
                    continue
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError):
            continue
        events = _parse_event_payload(data)
        if events:
            deduped: Dict[str, RaceOption] = {}
            for event in events:
                if not event.title:
                    continue
                key = event.title.lower()
                if key not in deduped:
                    deduped[key] = event
            return list(deduped.values())[:limit]
    # Fallback to static data filtered by query
    query_lower = query.lower()
    filtered = [event for event in STATIC_EVENT_DATA if query_lower in event.title.lower() or query_lower in (event.location or "").lower()]
    return filtered[:limit] if filtered else list(STATIC_EVENT_DATA)[:limit]

# ---------------------------------------------------------------------------
# Athlete profiling helpers
# ---------------------------------------------------------------------------

def parse_time_input(value: str) -> float:
    value = value.strip()
    if not value:
        raise ValueError("Leerer Zeitstring")
    if value.endswith("h"):
        return float(value[:-1]) * 60.0
    if ":" in value:
        parts = [float(part) for part in value.split(":")]
        if len(parts) == 2:
            hours, minutes = parts
            seconds = 0.0
        elif len(parts) == 3:
            hours, minutes, seconds = parts
        else:
            raise ValueError("Zeitformat muss mm:ss oder hh:mm:ss sein")
        return hours * 60 + minutes + seconds / 60.0
    return float(value)


def yes_no_input(prompt: str) -> bool:
    while True:
        answer = input(prompt + " [j/n]: ").strip().lower()
        if answer in {"j", "ja", "y", "yes"}:
            return True
        if answer in {"n", "nein", "no"}:
            return False
        print("Bitte mit 'j' oder 'n' antworten.")


def select_from_list(prompt: str, options: Sequence[str]) -> int:
    for idx, option in enumerate(options, start=1):
        print(f"  {idx}. {option}")
    while True:
        answer = input(prompt + " (Nummer eingeben): ").strip()
        try:
            value = int(answer)
        except ValueError:
            print("Bitte eine gültige Zahl eingeben.")
            continue
        if 1 <= value <= len(options):
            return value - 1
        print("Auswahl außerhalb des gültigen Bereichs.")


TARGET_CATEGORY_PRESETS: Dict[str, TargetExpectations] = {
    "einfach": TargetExpectations(
        category="Einfach überleben (unter Cut-off)",
        swim_minutes_3800=120,
        bike_minutes_180=480,
        run_minutes_marathon=300,
        total_minutes=16 * 60,
    ),
    "sub14": TargetExpectations(
        category="Finish unter 14 Stunden",
        swim_minutes_3800=100,
        bike_minutes_180=420,
        run_minutes_marathon=250,
        total_minutes=840,
    ),
    "sub12": TargetExpectations(
        category="Finish unter 12 Stunden",
        swim_minutes_3800=90,
        bike_minutes_180=360,
        run_minutes_marathon=220,
        total_minutes=720,
    ),
    "sub10": TargetExpectations(
        category="Finish unter 10 Stunden",
        swim_minutes_3800=75,
        bike_minutes_180=320,
        run_minutes_marathon=200,
        total_minutes=600,
    ),
}

def gather_athlete_profile() -> AthleteProfile:
    print("\nBitte gib deinen aktuellen Leistungsstand an (Zeiten in Minuten oder hh:mm).")
    while True:
        try:
            swim = parse_time_input(input("Aktuelle Schwimmzeit für 3,8 km: "))
            break
        except ValueError as exc:
            print(f"Ungültige Eingabe: {exc}")
    while True:
        try:
            bike = parse_time_input(input("Aktuelle Radzeit für 180 km: "))
            break
        except ValueError as exc:
            print(f"Ungültige Eingabe: {exc}")
    while True:
        try:
            run = parse_time_input(input("Aktuelle Laufzeit Marathon: "))
            break
        except ValueError as exc:
            print(f"Ungültige Eingabe: {exc}")
    while True:
        try:
            weekly_hours = float(input("Trainingsstunden pro Woche (Durchschnitt): ").strip())
            if weekly_hours < 0:
                raise ValueError
            break
        except ValueError:
            print("Bitte eine positive Zahl eingeben.")
    strength = yes_no_input("Machst du regelmäßig Kraft- oder Stabitraining?")
    experience_idx = select_from_list(
        "Wie schätzt du deine Triathlon-Erfahrung ein?",
        ["Einsteiger/in", "Fortgeschritten", "Sehr erfahren"],
    )
    experience = ["beginner", "intermediate", "advanced"][experience_idx]
    return AthleteProfile(
        swim_minutes_3800=swim,
        bike_minutes_180=bike,
        run_minutes_marathon=run,
        weekly_training_hours=weekly_hours,
        strength_training=strength,
        experience_level=experience,
    )


def gather_target_expectations() -> TargetExpectations:
    print("\nWelche Zielsetzung hast du für deinen Ironman?")
    options = [
        "Einfach finishen (unter Cut-off)",
        "Finish unter 14 Stunden",
        "Finish unter 12 Stunden",
        "Finish unter 10 Stunden",
        "Eigene Zielzeiten eingeben",
    ]
    choice = select_from_list("Bitte eine Option wählen", options)
    if choice == 4:
        while True:
            try:
                swim = parse_time_input(input("Zielzeit Schwimmen (3,8 km): "))
                bike = parse_time_input(input("Zielzeit Rad (180 km): "))
                run = parse_time_input(input("Zielzeit Marathon: "))
                break
            except ValueError as exc:
                print(f"Ungültige Eingabe: {exc}")
        total = swim + bike + run + 45  # 45 Minuten Reserve für Wechsel/Verpflegung
        return TargetExpectations(
            category="Individuelle Zielzeiten",
            swim_minutes_3800=swim,
            bike_minutes_180=bike,
            run_minutes_marathon=run,
            total_minutes=total,
        )
    keys = ["einfach", "sub14", "sub12", "sub10"]
    preset = TARGET_CATEGORY_PRESETS[keys[choice]]
    return TargetExpectations(
        category=preset.category,
        swim_minutes_3800=preset.swim_minutes_3800,
        bike_minutes_180=preset.bike_minutes_180,
        run_minutes_marathon=preset.run_minutes_marathon,
        total_minutes=preset.total_minutes,
    )

def estimate_preparation_weeks(profile: AthleteProfile, target: TargetExpectations) -> int:
    swim_gap = max(0.0, (profile.swim_minutes_3800 - target.swim_minutes_3800) / target.swim_minutes_3800)
    bike_gap = max(0.0, (profile.bike_minutes_180 - target.bike_minutes_180) / target.bike_minutes_180)
    run_gap = max(0.0, (profile.run_minutes_marathon - target.run_minutes_marathon) / target.run_minutes_marathon)
    endurance_gap = (swim_gap + bike_gap + run_gap) / 3.0

    base_weeks = 20 + endurance_gap * 14

    if profile.experience_level == "beginner":
        base_weeks += 10
    elif profile.experience_level == "intermediate":
        base_weeks += 4
    else:
        base_weeks -= 2

    if profile.weekly_training_hours < 6:
        base_weeks += 8
    elif profile.weekly_training_hours < 10:
        base_weeks += 4
    elif profile.weekly_training_hours > 15:
        base_weeks -= 2

    if not profile.strength_training:
        base_weeks += 1.5

    if target.total_minutes <= 600:  # sub 10h Anspruch
        base_weeks += 6
    elif target.total_minutes <= 720:
        base_weeks += 2
    elif target.total_minutes >= 900:
        base_weeks -= 3

    return int(min(42, max(14, round(base_weeks))))

BASE_PHASE_SEQUENCE: List[Tuple[str, str]] = [
    ("Grundlagenphase 1", "W2"),
    ("Grundlagenphase 1", "W3"),
    ("Grundlagenphase 1", "Wreg"),
    ("Grundlagenphase 1", "W1"),
    ("Grundlagenphase 1", "W2"),
    ("Grundlagenphase 1", "W3"),
    ("Grundlagenphase 2", "Wreg"),
    ("Grundlagenphase 2", "W1"),
    ("Grundlagenphase 2", "W2"),
    ("Grundlagenphase 2", "W3"),
    ("Grundlagenphase 2", "Wreg"),
    ("Grundlagenphase 2", "W1"),
    ("Aufbauphase 1", "W2"),
    ("Aufbauphase 1", "W3"),
    ("Aufbauphase 1", "Wreg"),
    ("Aufbauphase 1", "W1"),
    ("Aufbauphase 1", "W2"),
    ("Aufbauphase 2", "W3"),
    ("Aufbauphase 2", "Wreg"),
    ("Aufbauphase 2", "W1"),
    ("Aufbauphase 2", "W2"),
    ("Aufbauphase 2", "W3"),
    ("Spezifische Phase", "W3_race"),
    ("Spezifische Phase", "W3"),
    ("Spezifische Phase", "Wreg"),
    ("Spezifische Phase", "W3_peak"),
    ("Spezifische Phase", "W3_runfocus"),
    ("Tapering", "Wreg"),
    ("Tapering", "W2"),
    ("Tapering", "W1"),
    ("Tapering", "WKW"),
]


EXTENSION_CYCLE: List[Tuple[str, str]] = [
    ("Grundlagenphase 1", "W1"),
    ("Grundlagenphase 1", "W2"),
    ("Grundlagenphase 1", "W3"),
    ("Grundlagenphase 1", "Wreg"),
    ("Grundlagenphase 2", "W1"),
    ("Grundlagenphase 2", "W2"),
    ("Grundlagenphase 2", "W3"),
    ("Grundlagenphase 2", "Wreg"),
]


def build_phase_sequence(desired_weeks: int) -> List[Tuple[str, str]]:
    sequence = list(BASE_PHASE_SEQUENCE)
    if desired_weeks > len(sequence):
        extra = desired_weeks - len(sequence)
        idx = 0
        while extra > 0:
            phase, load = EXTENSION_CYCLE[idx % len(EXTENSION_CYCLE)]
            sequence.insert(0, (phase, load))
            idx += 1
            extra -= 1
    elif desired_weeks < len(sequence):
        sequence = sequence[-desired_weeks:]
    return sequence


def _load_display(load: str) -> Tuple[str, Optional[str]]:
    if "_" in load:
        base, variant = load.split("_", 1)
        return base, variant
    return load, None


def build_training_plan(desired_weeks: int) -> Dict[str, object]:
    sequence = build_phase_sequence(desired_weeks)
    plan: List[Dict[str, object]] = []
    total_weeks = len(sequence)
    for idx, (phase, load_key) in enumerate(sequence):
        template = WEEK_TEMPLATES.get((phase, load_key))
        if template is None:
            raise KeyError(f"Kein Template für {phase} / {load_key} vorhanden")
        display_load, variation = _load_display(load_key)
        week_number = total_weeks - idx
        entry: Dict[str, object] = {
            "Woche": week_number,
            "Phase": phase,
            "Belastung": display_load,
            "Trainingsplan": _make_week(template),
        }
        if variation:
            entry["Variation"] = variation
        plan.append(entry)
    return {
        "Trainingsplan": plan,
        "Legende": copy.deepcopy(LEGEND),
        "Info": copy.deepcopy(INFO_BLOCK),
    }

# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------

def interactive_session(args: argparse.Namespace) -> Dict[str, object]:
    print("Willkommen beim KI-Ironman-Trainingscoach!\n")
    query = args.search or input("Nach welchem Ironman suchst du? ")
    events = fetch_ironman_events(query, limit=8)
    if not events:
        raise RuntimeError("Keine passenden Ironman-Rennen gefunden.")
    print("Gefundene Rennen:")
    options = [
        f"{event.title} – {event.location or 'Ort unbekannt'} ({event.start_date or 'Datum tbd'})"
        for event in events
    ]
    event_idx = select_from_list("Welches Rennen möchtest du wählen?", options)
    event = events[event_idx]
    print(f"\nAusgewähltes Rennen: {event.title}")
    if event.location:
        print(f"Ort: {event.location}")
    if event.start_date:
        print(f"Datum: {event.start_date}")
    if event.url:
        print(f"Weitere Infos: {event.url}")

    profile = gather_athlete_profile()
    target = gather_target_expectations()

    suggested_weeks = estimate_preparation_weeks(profile, target)
    print(
        f"\nAuf Basis deiner Angaben empfehlen wir eine Vorbereitung von etwa {suggested_weeks} Wochen."
    )

    chosen_weeks: Optional[int] = None
    if args.weeks:
        chosen_weeks = max(14, min(42, int(args.weeks)))
        print(f"Vorbereitungszeit durch Argument überschrieben: {chosen_weeks} Wochen")
    while chosen_weeks is None:
        answer = input(
            "Wie viele Wochen möchtest du planen? (Enter für Vorschlag oder Zahl zwischen 14 und 42): "
        ).strip()
        if not answer:
            chosen_weeks = suggested_weeks
            break
        try:
            weeks = int(answer)
        except ValueError:
            print("Bitte eine Zahl eingeben.")
            continue
        if 14 <= weeks <= 42:
            chosen_weeks = weeks
        else:
            print("Die Vorbereitungszeit muss zwischen 14 und 42 Wochen liegen.")
    plan = build_training_plan(chosen_weeks)
    plan["Ausgewähltes_Rennen"] = {
        "Titel": event.title,
        "Ort": event.location,
        "Datum": event.start_date,
        "URL": event.url,
    }
    plan["Athlet"] = {
        "Schwimmen_aktuell_min": profile.swim_minutes_3800,
        "Rad_aktuell_min": profile.bike_minutes_180,
        "Lauf_aktuell_min": profile.run_minutes_marathon,
        "Wochenstunden": profile.weekly_training_hours,
        "Krafttraining": profile.strength_training,
        "Erfahrung": profile.experience_level,
    }
    plan["Ziel"] = {
        "Kategorie": target.category,
        "Schwimmen_min": target.swim_minutes_3800,
        "Rad_min": target.bike_minutes_180,
        "Lauf_min": target.run_minutes_marathon,
        "Gesamt_min": target.total_minutes,
    }
    return plan


def run_demo_plan() -> Dict[str, object]:
    event = STATIC_EVENT_DATA[0]
    profile = AthleteProfile(
        swim_minutes_3800=115,
        bike_minutes_180=470,
        run_minutes_marathon=270,
        weekly_training_hours=10,
        strength_training=True,
        experience_level="intermediate",
    )
    target = TARGET_CATEGORY_PRESETS["sub12"]
    weeks = estimate_preparation_weeks(profile, target)
    plan = build_training_plan(weeks)
    plan["Ausgewähltes_Rennen"] = {
        "Titel": event.title,
        "Ort": event.location,
        "Datum": event.start_date,
        "URL": event.url,
    }
    plan["Athlet"] = profile.__dict__
    plan["Ziel"] = {
        "Kategorie": target.category,
        "Schwimmen_min": target.swim_minutes_3800,
        "Rad_min": target.bike_minutes_180,
        "Lauf_min": target.run_minutes_marathon,
        "Gesamt_min": target.total_minutes,
    }
    return plan


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="KI Ironman Coach")
    parser.add_argument("--search", help="Vorauswahl für die Rennsuche")
    parser.add_argument("--weeks", type=int, help="Vorbereitungsdauer in Wochen überschreiben")
    parser.add_argument("--output", help="Datei, in die der Plan geschrieben werden soll")
    parser.add_argument("--demo", action="store_true", help="Beispielplan ohne Interaktion ausgeben")
    args = parser.parse_args(argv)

    try:
        if args.demo:
            plan = run_demo_plan()
        else:
            plan = interactive_session(args)
    except KeyboardInterrupt:
        print("\nAbgebrochen.")
        return 1
    except Exception as exc:  # pragma: no cover - Fehlermeldung für CLI
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1

    output = json.dumps(plan, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output)
        print(f"Trainingsplan gespeichert in {args.output}")
    else:
        print("\n--- Trainingsplan ---")
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
