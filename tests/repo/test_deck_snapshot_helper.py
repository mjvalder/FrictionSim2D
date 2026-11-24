"""The deck-snapshot harness must not make scratch paths collide across processes."""
import uuid

from tests.helpers.deck_snapshot import _deterministic


def test_deterministic_uuids_differ_in_their_leading_hex():
    with _deterministic():
        first, second = uuid.uuid4(), uuid.uuid4()
    assert first != second
    assert first.hex[:8] != second.hex[:8]
    assert first.hex[:8] != "00000000"
