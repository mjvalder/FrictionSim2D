"""A material sweep previews exactly one, genuine, deterministic member.

Before this, ``PreviewController.start`` handed the *whole* config
(``materials_list`` and all) to the worker, which called
``generate_package`` -- and so built every material in the sweep -- only to
preview ``result.created[0]``. Which material that was ended up arbitrary
from the user's point of view (whichever the sweep happened to build first
internally), and it made previewing an N-material sweep take N builds.

``PreviewRequest.single_member`` fixes the selection: it reduces the config
to its first member via the same ``expand_config_sweeps`` a real run uses,
before anything reaches the worker.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.gui.controllers.preview import PreviewRequest, cache_key


def _sweep_config(materials_txt: Path) -> dict:
    """A minimal [2D] sweep config that needs no real cif/pot files.

    ``preflight_materials`` only checks files for keys actually present in
    the section, so leaving cif_path/pot_path out makes every listed
    material trivially "usable" -- what's under test is member *selection*,
    not file resolution.
    """
    return {
        '2D': {
            'materials_list': str(materials_txt),
            'mat': '{mat}',
            'x': 60,
            'y': 60,
        },
        'general': {
            'temp': 300,
        },
    }


@pytest.fixture(name='materials_txt')
def _materials_txt(tmp_path) -> Path:
    path = tmp_path / 'materials.txt'
    path.write_text('h_MoS2\nh_WS2\nh_graphene\n', encoding='utf-8')
    return path


class TestPreviewRequestReducesToFirstMember:
    """A sweep config must resolve to one, deterministic, genuine member."""

    def test_single_member_drops_materials_list_and_substitutes_mat(self, materials_txt):
        request = PreviewRequest(config=_sweep_config(materials_txt), model='indent')

        single = request.single_member()

        assert 'materials_list' not in single['2D']
        assert single['2D']['mat'] == 'h_MoS2', \
            'must pick the first material, not an arbitrary one'

    def test_first_member_is_deterministic_across_calls(self, materials_txt):
        request = PreviewRequest(config=_sweep_config(materials_txt), model='indent')

        assert request.single_member()['2D']['mat'] == 'h_MoS2'
        assert request.single_member()['2D']['mat'] == 'h_MoS2'

    def test_preview_label_names_the_shown_material(self, materials_txt):
        request = PreviewRequest(config=_sweep_config(materials_txt), model='indent')

        label = request.preview_label()

        assert label is not None
        assert 'h_MoS2' in label
        assert '3' in label, 'should say how many materials the sweep has'

    def test_non_sweep_config_has_no_preview_label(self):
        config = {'2D': {'mat': 'h_MoS2', 'x': 60, 'y': 60}}
        request = PreviewRequest(config=config, model='indent')

        assert request.preview_label() is None
        assert request.single_member() == config

    def test_non_sweep_cache_key_is_unchanged(self):
        """A non-sweep config's key must not shift just because the
        reduction machinery now runs -- existing cached previews stay valid."""
        config = {'2D': {'mat': 'h_MoS2', 'x': 60, 'y': 60}}
        request = PreviewRequest(config=config, model='indent', layer=None,
                                 referenced_files=[])

        assert request.key() == cache_key(config, 'indent', None, [])

    def test_two_different_sweeps_do_not_collide_in_the_cache(self, tmp_path):
        """Same base config, different materials_list -- the caches must not
        alias just because both are 'a sweep of the same shape'."""
        list_a = tmp_path / 'a.txt'
        list_a.write_text('h_MoS2\nh_WS2\n', encoding='utf-8')
        list_b = tmp_path / 'b.txt'
        list_b.write_text('h_WSe2\nh_MoSe2\n', encoding='utf-8')

        request_a = PreviewRequest(config=_sweep_config(list_a), model='indent')
        request_b = PreviewRequest(config=_sweep_config(list_b), model='indent')

        assert request_a.key() != request_b.key()


class TestOneMaterialSweepStillSubstitutes:
    """Ticking exactly one material is an ordinary sweep, not "no sweep".

    ``_first_sweep_member`` used to key off ``len(members) <= 1``, which
    conflated "not a sweep at all" with "a sweep of exactly one material" --
    both produce a single member. The latter is still a genuine,
    already-substituted member (``expand_config_sweeps`` strips
    ``materials_list`` and fills in ``{mat}`` regardless of how many
    materials were selected), and discarding it in favour of the original,
    unsubstituted config left ``cif_path``/``pot_path`` carrying a literal
    ``{mat}`` the builder cannot resolve.
    """

    @pytest.fixture(name='one_material_txt')
    def _one_material_txt(self, tmp_path) -> Path:
        path = tmp_path / 'one_material.txt'
        path.write_text('h_MoS2\n', encoding='utf-8')
        return path

    def _templated_config(self, materials_txt: Path) -> dict:
        # cif_path/pot_path deliberately omitted, same as `_sweep_config`
        # above: `preflight_materials` checks files only for keys actually
        # present in the section, so a real cif/pot directory would be
        # needed to keep "h_MoS2" usable if they were here. What's under
        # test is member *selection*, not file resolution -- that end-to-end
        # path (real files, real substitution into cif_path/pot_path) is
        # covered by `tests/test_gui_controllers_staged_paths.py
        # ::TestPreviewControllerOneMaterialSweepStagesSubstitutedPaths`.
        return {
            '2D': {
                'materials_list': str(materials_txt),
                'mat': '{mat}',
                'x': 60,
                'y': 60,
            },
            'general': {'temp': 300},
        }

    def test_single_material_still_drops_materials_list_and_substitutes_mat(
            self, one_material_txt):
        request = PreviewRequest(
            config=self._templated_config(one_material_txt), model='indent')

        single = request.single_member()

        assert 'materials_list' not in single['2D']
        assert single['2D']['mat'] == 'h_MoS2'

    def test_single_material_still_gets_a_preview_label(self, one_material_txt):
        """Not "no sweep" -- a one-material sweep is still worth naming, just
        not as "1 of 1", which reads oddly for a sweep the user deliberately
        narrowed to one material."""
        request = PreviewRequest(
            config=self._templated_config(one_material_txt), model='indent')

        label = request.preview_label()

        assert label is not None
        assert 'h_MoS2' in label
        assert '1 of 1' not in label

    def test_no_materials_list_at_all_is_still_not_a_sweep(self):
        """The other half of the discrimination: a plain, single-member,
        never-swept config must still fall back to itself unchanged --
        exactly the case the old `len(members) <= 1` rule was trying (and
        failing, for the one-material-sweep case above) to catch."""
        config = {
            '2D': {'mat': 'h_MoS2', 'cif_path': 'cif/h_MoS2.cif', 'x': 60, 'y': 60},
        }
        request = PreviewRequest(config=config, model='indent')

        assert request.single_member() == config
        assert request.preview_label() is None
