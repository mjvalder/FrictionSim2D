"""Retention tests (spec §5 F6, Task 9)."""
import hashlib

from src.postprocessing.retention import register_artefacts, verify_retention


def test_slide_template_writes_final_state(tmp_path):
    from src.builders.afm import AFMSimulation
    from tests.test_afm_builder import (_make_afm_config,
                                        _prepare_builder_for_slide_write)
    config = _make_afm_config(tmp_path, layers=[1])
    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    slide_path = _prepare_builder_for_slide_write(builder, tmp_path)
    text = slide_path.read_text()
    # write_data after the production run, before the loop tails
    run_pos = text.index('run             500000')
    wd_pos = text.index('final_f$(v_find)N_a$(v_a).data')
    assert wd_pos > run_pos


def test_verify_retention_flags_missing(tmp_path):
    prov = {'retained_artefacts': [
        {'path': 'data/load_30N.data', 'sha256': '00' * 32}]}
    assert verify_retention(tmp_path, prov) == ['data/load_30N.data']


def test_verify_retention_flags_mismatch_and_passes_match(tmp_path):
    f = tmp_path / 'data' / 'load_30N.data'
    f.parent.mkdir()
    f.write_text('payload')
    good = hashlib.sha256(b'payload').hexdigest()
    assert verify_retention(tmp_path, {'retained_artefacts': [
        {'path': 'data/load_30N.data', 'sha256': good}]}) == []
    bad = verify_retention(tmp_path, {'retained_artefacts': [
        {'path': 'data/load_30N.data', 'sha256': '00' * 32}]})
    assert bad == ['data/load_30N.data (checksum mismatch)']


def test_register_artefacts_roundtrip(tmp_path):
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data' / 'final_f30N_a0.data').write_text('x')
    entries = register_artefacts(tmp_path, ['data/*.data'])
    assert len(entries) == 1
    assert verify_retention(tmp_path, {'retained_artefacts': entries}) == []
