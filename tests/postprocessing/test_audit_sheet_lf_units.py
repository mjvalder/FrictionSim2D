"""The unit-correction census must be reproducible, not a number in a doc."""
from frictionsim2d.postprocessing.audit_lf_units import classify_file

HDR = "# Time-averaged data\n# TimeStep v_xfrict v_yfrict v_sx v_sy v_fz\n"


def test_classify_lj_branch(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text(HDR + "1000 2.0 1.0 0.5 0.25 -10.0\n")
    assert classify_file(p) == 'affected'


def test_classify_many_body_branch(tmp_path):
    # xfrict IS v_sx on that branch, so the columns are bit-identical
    p = tmp_path / "b.txt"
    p.write_text(HDR + "1000 0.5 0.25 0.5 0.25 -10.0\n")
    assert classify_file(p) == 'unaffected'


def test_classify_schema2_is_unaffected(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("# schema=2 lf_units=nN\n"
                 "# TimeStep v_xfrict v_yfrict v_sx v_sy v_fz\n"
                 "1000 2.0 1.0 0.5 0.25 -10.0\n")
    assert classify_file(p) == 'unaffected'


def test_classify_headerless(tmp_path):
    p = tmp_path / "d.txt"
    p.write_text("1000 2.0 1.0\n")
    assert classify_file(p) == 'unparseable'


def test_classify_missing_discriminator_column(tmp_path):
    # without v_sx the branch cannot be identified
    p = tmp_path / "e.txt"
    p.write_text("# Time-averaged data\n# TimeStep v_xfrict v_fz\n"
                 "1000 2.0 -10.0\n")
    assert classify_file(p) == 'unparseable'
