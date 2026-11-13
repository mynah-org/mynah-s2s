"""Unit tests for gold_v21 (plain asserts; run: python3 test_gold_v21.py). Cases come from the V2.1 replies that the
blinded manual check labelled, plus normalisation edge cases."""
from gold_v21 import canon_tokens, echo, hit, parse_number, spec

CASES = [
    # (cat, gold_text, aliases, reply, expected)
    # numbers: spelled, embedded in larger numbers, units
    ("mkqa:number", "4.0", ["4"], "The height of the Eiffel Tower is three hundred and twenty four meters.", False),
    ("mkqa:number", "6.0", ["6"], "The Book has forty six chapters.", False),
    ("mkqa:number", "6.0", ["6"], "The Book of Ephesians has six chapters.", True),
    ("mkqa:number", "25.0", ["25"], "The winner receives twenty five points.", True),
    ("mkqa:number", "5.0", ["5"], "The forty nine ers have won five Super Bowl rings.", True),
    ("mkqa:number", "384400", [], "About 384,400 kilometers.", True),
    ("mkqa:number", "384400", [], "three hundred eighty four thousand four hundred kilometers", True),
    ("mkqa:number", "35.0", ["35"], "West Virginia entered the Union as the 35th state.", True),
    ("mkqa:number", "35.0", ["35"], "It was the thirty fifth state.", True),
    ("mkqa:number_with_unit", "11.0 years", ["11 years"], "at the age of eleven months.", False),
    ("mkqa:number_with_unit", "11.0 years", ["11 years"], "harry potter is eleven years old in the first book", True),
    ("mkqa:number_with_unit", "18.0 years", ["18 years"], "The legal drinking age in Albania is eighteen, though", True),
    ("mkqa:number_with_unit", "8.0 seasons", ["8 seasons"], "There are eight episodes in the first season", False),
    ("mkqa:number_with_unit", "108.0 °F", ["108 °F"], "the record is one hundred and eight degrees Fahrenheit", True),
    # arithmetic: last number is the answer
    ("svamp:arithmetic", "2", [], "The sphere has two types of matter: the north and the south. If the radius is seven", False),
    ("svamp:arithmetic", "41", [], "<s>So the answer is two.</s><s>If Sarah had forty three apples and now has two, she must have used forty one apples.", False),
    ("svamp:arithmetic", "21", [], "<s>Since forty two divided by two equals twenty one, Frank put twenty one pieces in each bag.</s><s>each bag has two pieces", True),
    ("svamp:arithmetic", "21", [], "<s>To find out how many cookies he ate, we need to</s><s>So, Paco ate twenty one cookies.", True),
    ("svamp:arithmetic", "1", [], "He passed four more hours on the Chinese exam than the Spanish one.", False),  # pronoun 'one'
    ("svamp:arithmetic", "1", [], "He spends one more hour on Chinese than on Spanish.", True),
    ("svamp:arithmetic", "21", [], "twenty eight - seven equal sign twenty one So, Paco ate twenty one cookies.", True),
    ("svamp:arithmetic", "2", [], "Each World Cup has thirty two teams, which are split into eight groups of four teams each.", False),
    # dates
    ("mkqa:date", "1977", ["1977"], "Star Wars premiered on may twenty fifth, nineteen seventy seven.", True),
    ("mkqa:date", "1707", ["1707"], "The Act of Union in seventeen oh seven created the Kingdom.", True),
    ("mkqa:date", "2003", ["2003"], "released in two thousand three.", True),
    ("mkqa:date", "1980-04-15", [], "He was born on September third, nineteen eighty.", False),
    ("mkqa:date", "1980-04-15", [], "He was born on April fifteenth, nineteen eighty.", True),
    ("mkqa:date", "1977-05-25", [], "released on May 25th, 1977", True),
    ("mkqa:date", "2017-10", [], "in 2017", False),
    # entities
    ("mkqa:entity", "Buzz Aldrin", [], "The second person was Buzz Aldrin, who joined Neil Armstrong.", True),
    ("mkqa:entity", "Pelé", [], "Pele is the record holder", True),
    ("mkqa:entity", "United States of America", ["the United States", "U.S.A."], "The United States provided support", True),
    ("mkqa:short_phrase", "stop-motion", [], '"Coraline" is a stop-motion animated film.', True),
    ("mkqa:entity", "Olaf", [], "The snowman's name is Frosty.", False),
]


def test_parse():
    assert parse_number(["three", "hundred", "and", "twenty", "four"]) == 324
    assert parse_number(["nineteen", "seventy", "seven"]) == 1977
    assert parse_number(["seventeen", "oh", "seven"]) == 1707
    assert parse_number(["twenty", "fifth"]) == 25
    assert parse_number(["two", "thousand", "three"]) == 2003
    assert canon_tokens("forty one thousand") == ["41000"]
    assert canon_tokens("a hundred and sixty four hours") == ["164", "hours"]


def test_cases():
    bad = []
    for cat, g, al, reply, exp in CASES:
        got = hit(reply, spec(cat, g, al))
        if got != exp:
            bad.append((cat, g, reply, exp, got))
    assert not bad, "\n".join(map(str, bad))


def test_echo():
    assert echo("Where was the first mcdonald's built in france?", spec("mkqa:entity", "France", []))
    assert echo("Brenda prepared 18 cookies for 10 guests", spec("svamp:arithmetic", "18", []))
    assert not echo("When was star wars a new hope made?", spec("mkqa:date", "1977", []))


if __name__ == "__main__":
    test_parse()
    test_cases()
    test_echo()
    print(f"OK: parse, {len(CASES)} cases, echo")
