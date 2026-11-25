"""The unit-correction census must be reproducible, not a number in a doc."""
import pytest

from frictionsim2d.postprocessing.audit_lf_units import classify_file

HDR = "# Time-averaged data\n# TimeStep v_xfrict v_yfrict v_sx v_sy v_fz\n"


@pytest.mark.parametrize('content, expected', [
    pytest.param(HDR + "1000 2.0 1.0 0.5 0.25 -10.0\n", 'affected', id='lj'),
    # xfrict IS v_sx on the many-body branch, so the columns are bit-identical
    pytest.param(HDR + "1000 0.5 0.25 0.5 0.25 -10.0\n", 'unaffected', id='many_body'),
    pytest.param("# schema=2 lf_units=nN\n"
                 "# TimeStep v_xfrict v_yfrict v_sx v_sy v_fz\n"
                 "1000 2.0 1.0 0.5 0.25 -10.0\n", 'unaffected', id='schema2'),
    pytest.param("1000 2.0 1.0\n", 'unparseable', id='headerless'),
    # without v_sx the branch cannot be identified
    pytest.param("# Time-averaged data\n# TimeStep v_xfrict v_fz\n1000 2.0 -10.0\n",
                 'unparseable', id='no_discriminator'),
])
def test_classify_file(tmp_path, content, expected):
    p = tmp_path / "a.txt"
    p.write_text(content)
    assert classify_file(p) == expected
