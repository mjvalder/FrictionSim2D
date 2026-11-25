"""Unit tests for the settings/config staging choke point (spec follow-up A,
2026-09-22 review round 2: broaden candidate detection, fail closed on an
unparsable candidate).

`frictionsim2d.aiida.data.redact` needs no AiiDA profile -- it is plain YAML/JSON/INI
handling plus a `repository`-shaped stub -- but lives under `tests/aiida/`
next to the store tests it protects, and skips itself the same way they do
so it never runs in an environment without aiida-core installed.
"""
from __future__ import annotations

import json
from typing import Dict

import pytest
import yaml

pytest.importorskip('aiida')
pytestmark = pytest.mark.aiida

from frictionsim2d.aiida.data.redact import (  # noqa: E402
    UnparsableDocumentError, is_settings_or_config_document,
    sanitize_document_bytes, stage_file,
)

PASSWORD = 'hunter2-central-db-password'


class _FakeRepository:
    """Enough of AiiDA's repository interface for `stage_file`: records
    every write by name, distinguishing the two `put_object_from_*` calls."""

    def __init__(self) -> None:
        self.objects: Dict[str, bytes] = {}

    def put_object_from_file(self, path, name):
        with open(path, 'rb') as handle:
            self.objects[name] = handle.read()

    def put_object_from_bytes(self, data, name):
        self.objects[name] = data


# --- is_settings_or_config_document: item 2's basename/location matrix ----

@pytest.mark.parametrize('name', [
    'settings.yaml', 'settings.yml', 'config.json',
    'Settings.YAML', 'config.JSON',
    'settings.yaml.bak', 'config.json.bak',
    'config.ini', 'Config.INI',
    'provenance/settings.yaml', 'provenance/config.json',
    'provenance/manifest.json',
    'afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/provenance/settings.yaml',
])
def test_is_candidate_matches_every_probed_shape(name):
    assert is_settings_or_config_document(name)


@pytest.mark.parametrize('name', [
    'lammps/system.in',
    'results/friction_f30_a0_s2.txt',
    'data/hold_f30.txt',
    'provenance.json',
    'log_slide.lammps',
    'provenance/cif/h-MoS2.cif',
    'provenance/potentials/MoS.sw',
    'afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/provenance/cif/h-MoS2.cif',
])
def test_is_candidate_excludes_ordinary_files(name):
    assert not is_settings_or_config_document(name)


# --- sanitize_document_bytes: format handling ------------------------------

def test_sanitizes_json_regardless_of_a_bak_suffix():
    data = json.dumps({'database': {'password': PASSWORD, 'host': 'x'}}).encode()

    out = sanitize_document_bytes(data, 'config.json.bak')

    parsed = json.loads(out)
    assert parsed['database']['password'] == ''
    assert parsed['database']['host'] == 'x'


def test_sanitizes_yaml_under_an_uppercase_name():
    data = yaml.safe_dump({'database': {'password': PASSWORD}}).encode()

    out = sanitize_document_bytes(data, 'Settings.YAML')

    assert yaml.safe_load(out)['database']['password'] == ''


def test_sanitizes_ini_and_blanks_secret_named_options():
    data = (
        "[DEFAULT]\n"
        "api_key = default-section-key\n"
        "[database]\n"
        f"password = {PASSWORD}\n"
        "host = db.example.org\n"
    ).encode()

    out = sanitize_document_bytes(data, 'config.ini').decode()

    assert PASSWORD not in out
    assert 'default-section-key' not in out      # [DEFAULT] options are secrets too
    assert 'db.example.org' in out


# --- fail closed (item 3) --------------------------------------------------

def test_unparsable_ini_fails_closed():
    with pytest.raises(UnparsableDocumentError, match='config.ini'):
        sanitize_document_bytes(b'[unterminated section\nkey = value\n', 'config.ini')


def test_multi_document_yaml_fails_closed():
    bad = b"---\na: 1\n---\nb: 2\n"
    with pytest.raises(UnparsableDocumentError,
                       match='could not be parsed as JSON or YAML'):
        sanitize_document_bytes(bad, 'provenance/settings.yaml')


def test_non_utf8_content_fails_closed():
    with pytest.raises(UnparsableDocumentError, match='not valid UTF-8'):
        sanitize_document_bytes(b'\xff\xfe\x00\x01', 'provenance/settings.yaml')


def test_stage_file_raises_for_an_unparsable_candidate_and_stores_nothing(tmp_path):
    repo = _FakeRepository()
    bad_file = tmp_path / 'settings.yaml'
    bad_file.write_bytes(b'\xff\xfe\x00\x01')

    with pytest.raises(UnparsableDocumentError):
        stage_file(repo, bad_file, 'settings.yaml')
    assert repo.objects == {}


def test_stage_file_leaves_non_candidates_byte_identical(tmp_path):
    repo = _FakeRepository()
    script = tmp_path / 'system.in'
    script.write_bytes(b'# lammps input\n')

    stage_file(repo, script, 'lammps/system.in')

    assert repo.objects['lammps/system.in'] == b'# lammps input\n'
