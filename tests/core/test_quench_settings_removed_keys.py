"""`QuenchSettings` still loads the keys removed from it, with a warning."""
import warnings

from frictionsim2d.core.config import QuenchSettings


def test_removed_quench_keys_load_with_a_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        settings = QuenchSettings.model_validate(
            {'run_local': True, 'n_procs': 4, 'allow_generate': False})
    assert settings.melt_steps == 20000 and settings.equilibrate_steps == 10000
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert not hasattr(settings, 'run_local')


def test_removed_quench_keys_are_logged_where_the_cli_shows_them(caplog):
    with caplog.at_level('WARNING', logger='frictionsim2d.core.config'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            QuenchSettings.model_validate({'run_local': True})
    assert any('run_local' in r.getMessage() for r in caplog.records)
