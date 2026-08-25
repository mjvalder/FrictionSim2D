"""The lammps marker must cover every lmp_serial-guarded test.

Without this, `-m 'not lammps'` silently lets a multi-minute LAMMPS test into
the fast loop the moment someone adds a guard without the marker.
"""


def test_every_lmp_guarded_test_carries_the_marker(request):
    session = request.session
    unmarked = []
    for item in session.items:
        guarded = any(
            "lmp_serial" in str(m.kwargs.get("reason", ""))
            for m in item.iter_markers(name="skipif")
        )
        if guarded and not any(item.iter_markers(name="lammps")):
            unmarked.append(item.nodeid)
    assert not unmarked, f"lmp_serial-guarded but unmarked: {unmarked[:5]}"
