"""The manual describes what the code does (Phase 3).

Three help strings said "window is the only tier postprocessing reads". All
three tiers are read now, and a stale help string is worse than none: it
tells a user not to enable the data they need.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'documentation' / 'postprocessing.md'


def test_the_manual_page_exists_and_is_indexed():
    assert DOC.is_file()
    index = (ROOT / 'documentation' / 'README.md').read_text(encoding='utf-8')
    assert 'postprocessing.md' in index


def test_it_documents_the_export_schema_and_the_csv():
    text = DOC.read_text(encoding='utf-8')

    assert 'output_full_<size>.json' in text
    assert '"schema": 2' in text
    assert 'summary_<size>.csv' in text
    for column in ('rel_path', 'load_kind', 'mean_cof', 'gap_min'):
        assert column in text


def test_it_documents_every_plot_type_and_the_preset():
    from src.postprocessing.plot_config import PLOT_TYPES
    from src.postprocessing.presets import PRESETS

    text = DOC.read_text(encoding='utf-8')

    for plot_type in PLOT_TYPES:
        assert f'`{plot_type}`' in text
    for preset in PRESETS:
        assert preset in text


def test_it_documents_the_new_cli_options():
    text = DOC.read_text(encoding='utf-8')

    for option in ('--export', '--validate', '--preset', '--schema'):
        assert option in text


def test_it_carries_the_migration_warning():
    """Task 19 changed what mean_cof means. A reader who regenerates a figure
    and gets a different number has to be able to find out why."""
    text = DOC.read_text(encoding='utf-8')

    assert '## Migration' in text
    assert 'ratio_of_means' in text or 'mean(LF)/mean(NF)' in text


def test_no_help_string_still_claims_window_is_the_only_tier():
    """The claim lived in three places at once; they are corrected together
    or not at all."""
    from src.core.config import OutputSettings
    from src.gui.schema.overrides import SETTINGS_MODEL, field_override

    description = OutputSettings.model_fields['channels'].description or ''
    # The channels override lives in the SETTINGS table, not the per-config
    # one -- `channels` is a settings.yaml field, not an .ini field.
    detail = field_override(SETTINGS_MODEL, 'output', 'channels').detail or ''
    commands = (ROOT / 'documentation' / 'commands.md').read_text(
        encoding='utf-8')

    for text in (description, detail, commands, DOC.read_text(encoding='utf-8')):
        assert 'only tier' not in text
    assert 'all three tiers' in description.lower()
    assert 'all three' in detail.lower()
