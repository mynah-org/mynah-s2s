"""CPU unit tests for normalize.py (plain asserts; run: python3 test_normalize.py, or pytest)."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from frankenstein2.text.normalize import normalize, number_words, ordinal_words, split_safe  # noqa: E402

CASES = [
    # the TTS comparison's failures: times and prices
    ("it", "Alle 10:30.", "Alle dieci e trenta."),
    ("it", "Costa 49,90 €.", "Costa quarantanove euro e novanta."),
    ("fr", "À 10:30.", "À dix heures trente."),
    ("fr", "Ça coûte 49,90 €.", "Ça coûte quarante-neuf euros quatre-vingt-dix."),
    ("en", "At 10:30.", "At ten thirty."),
    ("en", "It costs €49.90.", "It costs forty-nine euros and ninety cents."),
    ("en", "It costs $1.", "It costs one dollar."),
    # integers / decimals with the locale's separators
    ("it", "Ho 21 anni e 3,5 kg.", "Ho ventuno anni e tre virgola cinque kg."),
    ("it", "1.234 pezzi", "milleduecentotrentaquattro pezzi"),
    ("fr", "71, 80, 91 et 200 000", "soixante et onze, quatre-vingts, quatre-vingt-onze et deux cent mille"),
    ("en", "1,234 units and 3.5 kg", "one thousand two hundred thirty-four units and three point five kg"),
    # times
    ("it", "alle 13:05", "alle tredici e cinque"),
    ("it", "alle 9:00", "alle nove"),
    ("fr", "à 10h30", "à dix heures trente"),
    ("en", "at 9:05 pm", "at nine oh five p m"),
    ("en", "at 7:00", "at seven o'clock"),
    # dates
    ("it", "il 14/11", "il quattordici novembre"),
    ("it", "il 14 novembre 2026", "il quattordici novembre duemilaventisei"),
    ("it", "dal 1° maggio", "dal primo maggio"),
    ("fr", "le 1er mai 2026", "le premier mai deux mille vingt-six"),
    ("fr", "le 14/11/2026", "le quatorze novembre deux mille vingt-six"),
    ("en", "on 11/14", "on November fourteenth"),
    ("en", "on November 14th, 2026", "on November fourteenth, twenty twenty-six"),
    ("en", "the 14th of November", "the fourteenth of November"),
    # ordinals
    ("it", "il 3° piano e la 2ª porta", "il terzo piano e la seconda porta"),
    ("it", "il 23° posto", "il ventitreesimo posto"),
    ("fr", "le 2e étage", "le deuxième étage"),
    ("en", "the 21st item", "the twenty-first item"),
    # percentages
    ("it", "sconto del 15%", "sconto del quindici per cento"),
    ("fr", "2,5 %", "deux virgule cinq pour cent"),
    ("en", "15% off", "fifteen percent off"),
    # codes, postcodes, phones
    ("it", "ordine XYZ123", "ordine X Y Z uno due tre"),
    ("en", "order A1B2", "order A one B two"),
    ("it", "CAP 10121", "CAP uno zero uno due uno"),
    ("fr", "code 75011", "code sept cinq zéro un un"),
    ("it", "tel +39 333 123 4567", "tel più tre nove, tre tre tre, uno due tre, quattro cinque sei sette"),
    ("en", "call 555-123-4567", "call five five five, one two three, four five six seven"),
    # foreign separators: a group that is not 3 digits is a decimal; a first number > 12 is a day
    ("en", "It costs 49,90 euros.", "It costs forty-nine euros and ninety cents."),
    ("it", "Pesa 3.5 kg.", "Pesa tre virgola cinque kg."),
    ("en", "arrives on 14/11", "arrives on November fourteenth"),
    # untouched
    ("it", "Buongiorno, come posso aiutarla?", "Buongiorno, come posso aiutarla?"),
]


def test_cases():
    bad = []
    for lang, src, want in CASES:
        got = normalize(src, lang)
        if got != want:
            bad.append((lang, src, got, want))
    assert not bad, "\n".join(f"{l}: {s!r} -> {g!r} (want {w!r})" for l, s, g, w in bad)


def test_numbers():
    assert number_words(1, "it") == "uno" and number_words(23, "it") == "ventitré"
    assert number_words(180, "it") == "centottanta" and number_words(2026, "it") == "duemilaventisei"
    assert number_words(80, "fr") == "quatre-vingts" and number_words(81, "fr") == "quatre-vingt-un"
    assert number_words(71, "fr") == "soixante et onze" and number_words(200, "fr") == "deux cents"
    assert number_words(1_000_000, "en") == "one million"
    assert ordinal_words(1, "fr", fem=True) == "première" and ordinal_words(5, "fr") == "cinquième"
    assert ordinal_words(12, "en") == "twelfth" and ordinal_words(20, "en") == "twentieth"


def test_no_digits_left():
    for lang, src, _ in CASES:
        assert not any(c.isdigit() for c in normalize(src, lang)), (lang, src)


def test_split_safe():
    assert split_safe("Costa 49,") == len("Costa ")               # the price may continue
    assert split_safe("Alle 10:") == len("Alle ")
    assert split_safe("Il numero è +39 333 ") == len("Il numero è ")
    assert split_safe("Buongiorno, come ") == len("Buongiorno, come ")
    assert split_safe("Costa 49,90 € e poi") == len("Costa 49,90 € e ")


if __name__ == "__main__":
    for f in (test_cases, test_numbers, test_no_digits_left, test_split_safe):
        f()
    print("ok", len(CASES), "cases")
