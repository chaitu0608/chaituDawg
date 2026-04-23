"""Deterministic fuzz-style tests for extraction helpers."""

from __future__ import annotations

import random
import string

from src.tools import extract_email, extract_name, extract_platform, normalize_platform


def _random_text(rng: random.Random, length: int) -> str:
    alphabet = string.ascii_letters + string.digits + " -_.,;:!?@#/"
    return "".join(rng.choice(alphabet) for _ in range(length))


def test_extractors_do_not_crash_on_random_noise() -> None:
    rng = random.Random(1337)
    for _ in range(500):
        sample = _random_text(rng, rng.randint(0, 120))
        _ = extract_name(sample)
        _ = extract_email(sample)
        _ = extract_platform(sample)


def test_extract_email_finds_embedded_email_in_noise() -> None:
    noisy = "xx!! contact me >>> ava@example.com <<< thanks"
    assert extract_email(noisy) == "ava@example.com"


def test_platform_normalization_is_stable_for_aliases() -> None:
    assert normalize_platform("yt") == "YouTube"
    assert normalize_platform("tik tok") == "TikTok"
    assert normalize_platform(" insta ") == "Instagram"
