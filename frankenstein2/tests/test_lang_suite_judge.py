"""The language suite's judges (pure Python): is a reply in the session's language?"""
from frankenstein2.lab.lang_suite import char_ratio, in_language


def test_short_replies_without_stop_words_count_by_their_letters():
    assert in_language("Nemáme žádné trojmístné pohovky. Můžete mi říct, jaký typ pohovky hledáte?", "cs")
    assert in_language("Üç kişilik kanep bulamadım. Hangi ürünü kastettiğinizi söyleyebilir misiniz?", "tr")


def test_english_drift_is_caught():
    assert not in_language("Our three-seater sofa, the Fjordby, costs 499 euros. Would you like to know more?", "ru")
    assert not in_language("Yes, we have the sofa for you and it costs 499 euros.", "it")
    assert in_language("Минутку, сейчас проверю. У нас есть диван.", "ru")


def test_char_ratio_ignores_case_and_punctuation():
    assert char_ratio("Hallo, Welt!", "hallo welt") == 1.0
