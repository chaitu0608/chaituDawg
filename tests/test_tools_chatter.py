"""Regression: high-intent boilerplate must not be mistaken for a contact name."""

from __future__ import annotations

from src.tools import extract_name


def test_extract_name_rejects_sign_me_up() -> None:
    assert extract_name("Sign me up") is None


def test_extract_name_rejects_get_started() -> None:
    assert extract_name("Get started") is None


def test_extract_name_still_parses_explicit_introduction() -> None:
    assert extract_name("My name is Zoe") == "Zoe"
