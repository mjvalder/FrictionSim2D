"""The config being edited.

Holds the same ``{section: {key: value}}`` mapping that ``parse_config``
returns, so what the GUI edits is exactly what the CLI reads -- there is no
intermediate representation to drift.

Reading and writing go through the shared registry (``parse_config`` /
``write_config``), and validation through ``validate_config``, so no rule about
what a config means is restated here.

Qt-free.
"""

from __future__ import annotations

import copy
import os
import re
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

from ...core.config import (
    AFMSimulationConfig,
    SheetConfig,
    SheetOnSheetSimulationConfig,
    load_settings,
    parse_config,
)
from ...core.config import write_config
from ..config_validation import ConfigIssue, ConfigValidation, validate_config
from ...run.placeholders import has_template_marker
from ...run.expand import (
    ExpansionSummary, count_expansion, expand_config_sweeps,
    missing_pressure, outer_loop_conflict, read_material_entries, read_material_list,
)
from ...run.naming import loopability, output_dir_for
from ...core.utils.paths import _find_existing  # noqa: PLC2701
from ..schema.introspect import SectionSpec, describe_model, describe_section
from ..schema.overrides import override_for_section, overrides_for
from ..schema.simulation_types import MODES, sections_for
from .materials import STRUCTURE_EXTENSION

__all__ = ['ConfigDocument', 'DocumentError', 'ExpansionDirs',
           'hetero_support_error']

logger = logging.getLogger(__name__)


class ExpansionDirs(NamedTuple):
    """The Generate preview: where this config would build, or why it cannot.

    Exactly one side is populated. `error` carries the expansion's own
    message verbatim -- the pair guards in `src.run.expand.expand_config_sweeps`
    say precisely which section is wrong and what to put there, and that
    sentence is the whole value of showing it.
    """

    #: One directory per expanded config, duplicates included.
    dirs: List[Path]
    #: Why nothing could be expanded, or None when it could.
    error: Optional[str] = None


#: A numbered sheet section. Two of them make a heterostructure: `[2D-1]` is
#: the first material, `[2D-2]` the second. Matched exactly, so nothing that
#: merely looks like one (`2D-`, `2Dx`) is folded (mirrors
#: `src.run.naming.loopability_section`).
_NUMBERED_SHEET = re.compile(r'^2D-(\d+)$')

#: The written spelling of the first sheet when there is a second one. The
#: document itself always calls it `2D` -- see `normalised_values`.
FIRST_SHEET = '2D-1'

TOO_MANY_SHEETS = (
    "This config defines {count} sheet sections ({sections}). A "
    "heterostructure is exactly two materials, [2D-1] and [2D-2]; more than "
    "that has no section in the form to hold it -- open it in the CLI instead."
)

NON_CANONICAL_SHEETS = (
    "This config numbers its sheet sections {sections}. The form holds the "
    "first material as [2D-1] and the second as [2D-2]; loading this would "
    "have to renumber the config silently."
)

MIXED_SHEET_SECTIONS = (
    "This config mixes a plain [2D] section with numbered ones ({sections}). "
    "Only the numbered sections build the stack -- [2D] is overwritten by the "
    "first of them -- so the config does not mean what it reads as. Give the "
    "first material as [2D-1] instead."
)


class DocumentError(Exception):
    """The document cannot represent this config."""


@dataclass(frozen=True)
class PairListCollapse:
    """What dropping the second sheet would do to an ordered-pair list.

    Handed to the caller *before* anything happens, because the caller is the
    only one that can ask the user: the material list is the one part of a
    config edit that reaches disk immediately, so collapsing it is not
    undoable by closing without saving the way every other edit is.
    """

    #: The pair list as configured today, resolved on disk.
    source: Optional[Path]
    #: Where the single-material list will be written -- the config's own
    #: list, never `source`'s file unless they are already the same. None
    #: when there is nowhere legitimate to write (an unsaved config with no
    #: workspace), which `remove_second_material` refuses on.
    target: Optional[Path]
    #: How many pair lines the list holds.
    pairs: int
    #: The second members, which the remaining sheet has no use for. What the
    #: user is actually being asked about.
    dropped: List[str]
    #: The first members, deduped in file order: the collapsed list itself.
    firsts: List[str]


def _write_atomically(target: Path, text: str) -> None:
    """Write `text` to `target`, or leave `target` exactly as it was.

    A plain ``write_text`` truncates first, so a write that fails part-way --
    a full disk, a share that drops -- leaves a half-written material list
    behind while the caller's own rollback restores a document that no longer
    matches it. Writing a sibling and renaming makes the swap atomic: the old
    list survives every failure mode except the rename itself.

    A symlinked target is resolved first, so this keeps following the link (a
    list the user deliberately pointed the config at through one) instead of
    quietly replacing the link with a regular file.
    """
    resolved = Path(os.path.realpath(target))
    scratch = resolved.with_name(f".{resolved.name}.tmp")
    try:
        scratch.write_text(text, encoding='utf-8')
        scratch.replace(resolved)
    finally:
        # A failed write leaves the sibling behind; a successful one has
        # already renamed it away.
        scratch.unlink(missing_ok=True)


def hetero_support_error(model: str) -> Optional[str]:
    """Why ``model`` cannot hold a second material, or None when it can.

    Every mode the GUI offers routes a two-material config to a
    heterostructure builder (see ``SimulationMode.supports_hetero``), so this
    returns None for every real model today and the refusal below is currently
    unreachable. It is asked rather than assumed so that a mode added without
    a hetero build path refuses the second material instead of dropping it on
    the way to a builder that ignores ``sheets``.
    """
    entry = MODES.get(model)
    if entry is None:
        return f"Unknown simulation model: {model}"
    if entry[1].supports_hetero:
        return None
    return (
        f"'{entry[1].label}' has no heterostructure build, so it cannot keep "
        "the second material. Remove the second material first."
    )


def root_model_for(model: str):
    """The pydantic root model a simulation model validates against."""
    from ...run.naming import AFM_LIKE_MODELS  # noqa: PLC0415

    return AFMSimulationConfig if model in AFM_LIKE_MODELS else SheetOnSheetSimulationConfig


@dataclass
class ConfigDocument:
    """A run config under edit."""

    model: str = 'afm'
    values: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    source_path: Optional[Path] = None
    #: Fallback directory for files an unsaved document needs to write (e.g.
    #: a material list) when there is no `source_path` to write beside yet.
    #: `MainWindow` sets this from its `Workspace` -- never a substitute for
    #: `source_path`'s directory, only what is used in its absence. Left None
    #: outside the GUI (e.g. in CLI/test use of this class), which is exactly
    #: when there is no workspace to fall back to.
    workspace_dir: Optional[Path] = None
    #: What an unsaved document's relative paths resolve against (see
    #: `base_dir`). Without it they resolved against the process's cwd --
    #: wherever the GUI happened to be launched -- so no relative path ever
    #: validated and a sweep over relative folders listed nothing.
    workspace_root: Optional[Path] = None
    dirty: bool = False
    #: `unexpanded_template` issues held back by the most recent validate().
    #: Not an error when a material sweep is configured -- expansion
    #: substitutes them -- but worth reporting as pending work.
    pending_template_issues: List[Any] = field(default_factory=list)

    # ------------------------------------------------------------------ load

    @classmethod
    def from_file(cls, path: Union[str, Path], model: str) -> 'ConfigDocument':
        """Load an existing config.

        A heterostructure config -- ``[2D-1]`` plus ``[2D-2]`` -- is held as
        the first sheet under the document's usual ``2D`` key and the second
        under ``2D-2``, so every existing single-material code path keeps
        working unchanged and only the second material is new.

        Raises:
            DocumentError: If the config's sheet sections are not something
                the form can hold: more than two, numbered non-canonically, or
                a plain ``[2D]`` mixed with numbered ones. Refusing beats
                silently renumbering someone's config.
        """
        path = Path(path)
        values = cls._absorb_numbered_sheets(parse_config(path), model)
        return cls(model=model, values=values, source_path=path, dirty=False)

    @staticmethod
    def _absorb_numbered_sheets(values: Dict[str, Dict[str, Any]],
                                model: str) -> Dict[str, Dict[str, Any]]:
        """Rename ``[2D-1]`` to the document's own ``2D`` key, in place order.

        ``[2D-2]`` keeps its name: it is the document's key for the second
        material. A lone ``[2D-1]`` is a single-material config written in the
        numbered spelling, and loads as one -- it will be written back as a
        plain ``[2D]``, which is the same config.
        """
        numbered = [k for k in values if _NUMBERED_SHEET.match(k)]
        if not numbered:
            return values

        numbered.sort(key=lambda k: int(_NUMBERED_SHEET.match(k).group(1)))
        listed = ', '.join(numbered)

        if '2D' in values:
            raise DocumentError(MIXED_SHEET_SECTIONS.format(sections=listed))
        if len(numbered) > 2:
            raise DocumentError(
                TOO_MANY_SHEETS.format(count=len(numbered), sections=listed))
        if numbered not in ([FIRST_SHEET], [FIRST_SHEET, ConfigDocument.SECOND_SHEET]):
            raise DocumentError(NON_CANONICAL_SHEETS.format(sections=listed))

        reason = (hetero_support_error(model)
                  if len(numbered) == 2 else None)
        if reason:
            raise DocumentError(reason)

        return {('2D' if key == FIRST_SHEET else key): section
                for key, section in values.items()}

    @classmethod
    def blank(cls, model: str) -> 'ConfigDocument':
        """A new config with every required section present and defaulted."""
        document = cls(model=model)
        for section in document.sections():
            # A section the model marks Optional can still be mandatory in
            # practice -- [sub] is Optional[SubstrateConfig], but every model
            # except the PES tip scan aborts without it.
            if not section.optional or document.is_section_locked(section.key):
                document.values[section.key] = document._defaults_for(section)
        return document

    # -------------------------------------------------------------- schema

    def sections(self) -> List[SectionSpec]:
        """Sections relevant to this model, in display order."""
        specs = {s.key: s for s in describe_model(root_model_for(self.model))}
        relevant = sections_for(self.model)
        overrides = overrides_for(self.model)

        ordered = [specs[key] for key in relevant if key in specs]
        ordered.sort(key=lambda s: overrides.get(s.key).order
                     if overrides.get(s.key) else 100)
        return ordered

    def section(self, key: str) -> Optional[SectionSpec]:
        return next((s for s in self.sections() if s.key == key), None)

    def _defaults_for(self, section: SectionSpec) -> Dict[str, Any]:
        # Gated fields (e.g. the tip position quartet) are excluded even
        # though tip_x_offset/tip_y_offset default to 0.0 and would otherwise
        # pre-fill: every blank config would then arrive with those keys
        # already set, ticking the "set tip position" gate on sight and
        # defeating the whole feature.
        override = overrides_for(self.model).get(section.key)
        gated = {
            key for key, fo in (override.fields if override else {}).items()
            if fo.gate
        }
        return {
            f.key: f.default
            for f in section.fields
            if f.key not in gated and (f.default is not None or f.required)
        }

    # --------------------------------------------------------------- values

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.values.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        """Set a value, marking the document dirty when it actually changed."""
        current = self.values.setdefault(section, {})
        if current.get(key) == value:
            return
        current[key] = value
        self.dirty = True

    def has_section(self, key: str) -> bool:
        return key in self.values

    def enable_section(self, key: str, enabled: bool) -> None:
        """Add or remove an optional section.

        An absent section is how the models express "no substrate", "no flake":
        the field is ``Optional[...]`` and defaults to None.
        """
        spec = self.section(key)
        if spec is None:
            raise DocumentError(f"Unknown section: {key}")

        if not enabled and self.is_section_locked(key):
            # Checked before the presence test: refusing only when the section
            # happens to be present would let the same request succeed or fail
            # depending on state.
            raise DocumentError(self.section_lock_reason(key))

        if enabled and key not in self.values:
            self.values[key] = self._defaults_for(spec)
            self.dirty = True
        elif not enabled and key in self.values:
            del self.values[key]
            self.dirty = True

    def is_section_locked(self, key: str) -> bool:
        """Whether a nominally optional section cannot actually be removed."""
        return self.section_lock_reason(key) is not None

    # ----------------------------------------------------- second material

    #: The document's key for the second sheet of a heterostructure. The
    #: first keeps the plain `2D` key every single-material config uses, so
    #: nothing that reads `[2D]` has to learn about pairs; only writing out
    #: renames it (`normalised_values`).
    SECOND_SHEET = '2D-2'

    def has_second_material(self) -> bool:
        """Whether this config builds a two-material heterostructure."""
        return self.SECOND_SHEET in self.values

    def second_material_spec(self) -> SectionSpec:
        """The schema of the second sheet's section.

        The same ``SheetConfig`` the first sheet uses, minus
        ``materials_list``: only the first sheet carries the sweep list, and
        each of its lines names *both* materials, so a second list would be a
        second, conflicting sweep.
        """
        spec = describe_section(
            SheetConfig, name='second_material', key=self.SECOND_SHEET,
            optional=True,
        )
        spec.fields = [f for f in spec.fields if f.key != 'materials_list']
        return spec

    def add_second_material(self) -> None:
        """Add the second sheet, defaulted. A no-op when there is one already
        -- adding twice must not wipe what is in it.

        On a config that is *already* templated for a material sweep the new
        sheet is templated to match, and the first is re-spelled from `{mat}`
        to `{mat1}`. Which button the user pressed first must not decide
        whether the sweep is correct: a `[2D-2]` seeded with plain defaults
        under a pair list carries no placeholder, so expansion leaves it
        pinned and every run builds the same second material. The GUI's commit
        path re-templates on the next commit with the sweep box ticked, but a
        config *saved* in the half-templated state and run from the CLI never
        gets that chance.
        """
        if self.has_second_material():
            return
        self.values[self.SECOND_SHEET] = self._defaults_for(
            self.second_material_spec())
        self.dirty = True

        structures, potentials, extension = self.material_directories()
        if structures is not None:
            self._write_material_templates(structures, potentials, extension)

    def pair_list_collapse_plan(self) -> Optional[PairListCollapse]:
        """What removing the second sheet would do to the material list, or
        None when it would do nothing to it.

        None covers every case worth not asking about: no second sheet, no
        list, an unreadable one, or one that names single materials already.
        A prompt raised on all of those is a prompt nobody reads.
        """
        if not self.has_second_material():
            return None
        entries = self.material_entries()
        pairs = [entry for entry in entries if len(entry) == 2]
        if not pairs:
            return None
        return PairListCollapse(
            source=self._existing_material_list_target(),
            target=self._default_material_list_target(),
            pairs=len(pairs),
            dropped=list(dict.fromkeys(entry[1] for entry in pairs)),
            firsts=list(dict.fromkeys(entry[0] for entry in entries if entry)),
        )

    def remove_second_material(self) -> None:
        """Drop the second sheet, and everything that only it gives meaning.

        ``hetero_stacking`` orders the layers of a two-material stack; left
        behind on a single-material config it configures nothing, and reads as
        though a heterostructure were still being built.

        A *pair* material list is the dangerous leftover, and the reason this
        does more than delete a section. Its lines name two materials each
        (``'a + b'``), which `read_material_list` refuses outright --
        `material_names` swallows that and reports ``[]``, so the sweep stops
        sweeping and the user gets one run instead of N with nothing reported
        anywhere. The collapsed list -- each pair's *first* member, which is
        the sheet that is left -- is written to *this config's own* list file,
        never over the file the picker was pointed at unless they are already
        the same. A curated list can live anywhere and be shared by several
        configs, and every other edit in a document is discardable by closing
        without saving; a rewritten `.txt` is not.

        Ask `pair_list_collapse_plan()` first if the user should be consulted
        -- this method does not ask, and does not know how to.

        Raises:
            DocumentError: If the collapsed list has nowhere to go, or could
                not be written. Nothing is changed in either case: dropping
                the sheet regardless would leave the config pointed at a list
                of pairs it can no longer use, which is the silent
                stop-sweeping this whole path exists to prevent.
        """
        if not self.has_second_material():
            # The stacking pop below is still worth doing on its own -- a
            # hand-written config can carry it without a [2D-2] -- but the
            # list and template rewrites are meaningless without a pair.
            if (self.values.get('general') or {}).pop(
                    'hetero_stacking', None) is not None:
                self.dirty = True
            return

        plan = self.pair_list_collapse_plan()
        if plan is not None and plan.target is None:
            raise DocumentError(
                "This config's material list names ordered pairs, and there "
                "is nowhere to write the single-material list that replaces "
                "them. Save the config first, then remove the second material."
            )

        # Everything below either all happens or none of it does: the write is
        # the one step that can fail, and half a removal -- section gone, pair
        # list still configured -- is worse than either outcome on its own.
        snapshot = copy.deepcopy(self.values)
        was_dirty = self.dirty
        try:
            del self.values[self.SECOND_SHEET]
            self.dirty = True
            if (self.values.get('general') or {}).pop(
                    'hetero_stacking', None) is not None:
                self.dirty = True

            if plan is not None:
                self.write_material_names(plan.firsts, target=plan.target)

            # Re-spell the surviving sheet from `{mat1}` back to `{mat}`. Both
            # substitute identically, so this is legibility rather than
            # correctness -- but a one-sheet config spelled for a pair reads
            # as though the pair were still there. Mirrors
            # `add_second_material`, which templates the new sheet the same
            # way round.
            structures, potentials, extension = self.material_directories()
            if structures is not None:
                self._write_material_templates(structures, potentials, extension)
        except OSError as exc:
            self.values = snapshot
            self.dirty = was_dirty
            raise DocumentError(
                f"The single-material list could not be written to "
                f"'{plan.target if plan else ''}' ({exc}), so the second "
                "material was left in place."
            ) from exc

    def conflicts(self) -> List[str]:
        """Settings that do nothing as configured, in plain words.

        Not validation errors: each of these builds and runs, it just does not
        do what it looks like it does. A material sweep together with a second
        material is deliberately absent -- that combination is the ordered-pair
        sweep, not a conflict.

        A value equal to the model's own default is never reported. `blank()`
        seeds every defaulted key, so `hetero_stacking` is present as
        ``'grouped'`` on every single-material config the GUI creates;
        reporting that would put a "this does nothing" line under every new
        config, for a setting nobody chose -- which is how a panel of real
        warnings becomes one nobody reads.
        """
        out: List[str] = []
        stacking = self.get('general', 'hetero_stacking')
        if (stacking and stacking != self._field_default('general', 'hetero_stacking')
                and not self.has_second_material()):
            out.append(
                "[general] hetero_stacking orders the layers of a two-material "
                "stack; with a single material it has no effect."
            )
        return out

    def _field_default(self, section: str, key: str) -> Any:
        """The model's own default for a field, or None when it has none.

        Read off the introspected spec rather than restated here, so a default
        changed in `core.config` cannot leave a copy of the old one behind.
        """
        spec = self.section(section)
        field_spec = spec.field_map().get(key) if spec else None
        return field_spec.default if field_spec else None

    def section_lock_reason(self, key: str) -> Optional[str]:
        override = overrides_for(self.model).get(key)
        return override.required_reason if override else None

    # ----------------------------------------------------------- simulation

    def set_model(self, model: str, drop_second: bool = False) -> None:
        """Switch simulation model, keeping the sections the new one shares.

        Sections the new model has no use for are dropped, and its required
        ones are added with defaults, so the document always matches its model.

        The second material is *not* one of those sections: it survives the
        switch, because every mode builds a heterostructure. Pass
        ``drop_second`` to discard it deliberately.

        ``drop_second`` has no caller in the GUI, and cannot get one while
        every mode supports a heterostructure -- the mode combo has no reason
        to throw a material away. It is kept because it is the far side of
        the refusal below: a mode that could not build one would make
        `set_model` raise rather than lose a material silently, and this is
        the argument that says "yes, drop it" instead. Deleting it would
        leave that refusal with no documented way past it.

        Raises:
            DocumentError: If the target model is unknown, or (see
                `hetero_support_error` -- currently unreachable) has no
                heterostructure build while a second material is present.
                Losing a whole material to a dropdown change would be silent
                and unrecoverable.
        """
        if model not in MODES:
            raise DocumentError(f"Unknown simulation model: {model}")

        if drop_second:
            self.remove_second_material()
        elif self.has_second_material():
            reason = hetero_support_error(model)
            if reason:
                raise DocumentError(reason)

        if model == self.model:
            return

        self.model = model
        relevant = set(sections_for(model)) | {self.SECOND_SHEET}

        for key in [k for k in self.values if k not in relevant]:
            del self.values[key]

        for spec in self.sections():
            # Locked sections count as required: [sub] is Optional on the model
            # but every model except the PES tip scan aborts without it. Leaving
            # it out here is how switching pes_tip -> afm used to produce a
            # config that could not build, with a form that silently discarded
            # everything typed into the substrate.
            required = not spec.optional or self.is_section_locked(spec.key)
            if required and spec.key not in self.values:
                self.values[spec.key] = self._defaults_for(spec)

        self.dirty = True

    # ---------------------------------------------------------- validation

    @property
    def base_dir(self) -> Optional[Path]:
        """Directory that relative paths in this config resolve against: the
        config's own directory once saved, else the workspace root."""
        if self.source_path:
            return self.source_path.parent
        return Path(self.workspace_root) if self.workspace_root else None

    def validate(self, settings=None) -> ConfigValidation:
        """Validate against the real pydantic model.

        No rule is re-implemented here: cross-field physics constraints live in
        the models and their errors are mapped back to the field that caused
        them.

        One issue kind is reclassified rather than reported. A config driving a
        material sweep is *meant* to carry `{mat}` placeholders -- expansion
        substitutes them, and `preflight_paths` documents the kind as "not user
        error". Treating it as fatal made a correctly configured sweep
        impossible to render or generate. With no material list, though,
        nothing will ever substitute the placeholder, so it stays an error.

        When a material sweep is configured, `unexpanded_template` issues are
        moved to `pending_template_issues` and reported via that attribute
        instead. This means `ok` can be True while `config` is None: the
        preflight finds unresolved templates so the model was never
        constructed. Callers must not assume `.config` is usable when `ok` is
        True; check `pending_template_issues` first.

        Two checks cannot come from the model, because both depend on which
        builder the config routes to, which a pydantic model cannot see:
        whether `outer_loop` conflicts with `scan_angle_force`
        (`src.run.expand.outer_loop_conflict`), and whether a loaded slide is
        missing its `[general] pressure` (`src.run.expand.missing_pressure`).
        They are asked here rather than reimplemented, so the panel refuses
        exactly what `run_simulations` refuses.
        """
        payload = self.normalised_values()
        payload['settings'] = (settings or load_settings()).model_dump()
        result = validate_config(
            payload, root_model_for(self.model), base_dir=self.base_dir,
        )
        result.issues = self._issues_for_form(result.issues)

        conflict = outer_loop_conflict(self.values, self.model)
        if conflict is not None:
            result.issues.append(ConfigIssue(
                message=conflict, section='general', key='outer_loop',
            ))
        no_load = missing_pressure(self.values, self.model)
        if no_load is not None:
            result.issues.append(ConfigIssue(
                message=no_load, section='general', key='pressure',
            ))

        self.pending_template_issues = []
        if not self.has_material_list():
            return result

        pending, hard = [], []
        for issue in result.issues:
            (pending if issue.kind == 'unexpanded_template' else hard).append(issue)
        self.pending_template_issues = pending
        return ConfigValidation(issues=hard, config=result.config)

    def _issues_for_form(self, issues: List[ConfigIssue]) -> List[ConfigIssue]:
        """Re-address validated-config issues to the sections the form shows.

        Validation sees the written shape (`normalised_values`) and the model's
        own derived `sheets` list; the form shows `2D` and `2D-2`. Three
        spellings therefore have to come home:

        * ``[2D-1]``, from the path preflight, is the first sheet -- ``2D``;
        * ``sheets[0]`` is the same sheet again (the model mirrors the first
          numbered section onto its `2D` field as well as into the list), so
          it maps to ``2D`` and then dedupes against the report that already
          arrived there;
        * ``sheets[1]`` is the second material -- ``2D-2``.

        The *key* is left alone throughout: it is the field name, which is
        what the form decorates by. An issue mapped to the right section but
        keyed by a list index decorates nothing at all.
        """
        mapped: List[ConfigIssue] = []
        seen = set()
        for issue in issues:
            if issue.section == FIRST_SHEET:
                issue = replace(issue, section='2D')
            elif issue.section == 'sheets':
                issue = replace(
                    issue, index=None,
                    section=(self.SECOND_SHEET if (issue.index or 0) >= 1 else '2D'),
                )
            marker = (issue.section, issue.key, issue.kind, issue.message)
            if marker in seen:
                continue
            seen.add(marker)
            mapped.append(issue)
        return mapped

    def has_material_list(self) -> bool:
        """Whether a material sweep is configured."""
        return bool(self.get('2D', 'materials_list'))

    #: [2D] keys that must carry {mat} for a sweep to substitute per material.
    TEMPLATED_KEYS = ('mat', 'cif_path', 'pot_path')

    def material_names(self) -> List[str]:
        """Materials named by the configured list, or [] if there is none.

        Unreadable is not an error: the field is mid-edit more often than not.
        """
        path = self.get('2D', 'materials_list')
        if not isinstance(path, str) or not path:
            return []
        try:
            return read_material_list(path, self.base_dir)
        except (FileNotFoundError, OSError, ValueError):
            return []

    def material_entries(self) -> List[Tuple[str, ...]]:
        """The configured list read as entries: singles, or ordered pairs.

        The pair-aware sibling of `material_names`, which cannot represent a
        ``'a + b'`` line at all (`read_material_list` refuses one rather than
        silently keeping half of it). Unreadable is not an error, for the same
        reason as there: the field is mid-edit more often than not.
        """
        path = self.get('2D', 'materials_list')
        if not isinstance(path, str) or not path:
            return []
        try:
            return read_material_entries(path, self.base_dir)
        except (FileNotFoundError, OSError, ValueError):
            return []

    @staticmethod
    def _material_list_lines(entries: Sequence[Any]) -> List[str]:
        """One list line per entry: a name, or ``'mat1 + mat2'`` for a pair.

        ``'+'`` and not ``','``: a comma line already means several separate
        single materials, and has meant that since before pairs existed.
        """
        lines: List[str] = []
        for entry in entries:
            if isinstance(entry, (tuple, list)):
                members = [str(m).strip() for m in entry if str(m).strip()]
                if not members:
                    continue
                if len(members) > 2:
                    raise DocumentError(
                        f"'{' + '.join(members)}' names {len(members)} materials; "
                        "an ordered pair is exactly 2, one per sheet section."
                    )
                lines.append(' + '.join(members))
            elif entry and str(entry).strip():
                lines.append(str(entry).strip())
        return lines

    def write_material_names(self, names: Sequence[Any],
                             target: Optional[Path] = None) -> Optional[Path]:
        """Write the material list and point the config at it.

        `target` overrides where it goes. Only one caller states it --
        `remove_second_material`, which must write the collapsed list to this
        config's own file rather than over a curated pair list several configs
        may share -- and the resolution below is the default for everyone else.

        ``names`` holds plain material names, or ``(mat1, mat2)`` ordered
        pairs for a two-sheet sweep -- written as ``'mat1 + mat2'`` lines,
        which is what `read_material_entries` reads back.

        Selected names have to reach disk as the .txt the CLI reads; there is no
        GUI-only representation of a sweep. Written beside the config when it has
        been saved, so the pair travels together.

        A currently configured ``materials_list`` that resolves to a real file
        is written back to that same file -- not retargeted to the
        config-stem default. Without this, pointing the GUI's `.txt` picker
        at a curated file and then ticking/unticking a material silently
        abandoned the curated file (left with stale contents) in favour of a
        freshly created ``{stem}_materials.txt``, defeating the whole point
        of browsing to an existing list. The stem-based default is only ever
        used the first time a sweep is configured, when there is nothing to
        resolve yet.

        Returns:
            Where the list was written, or None when nothing was written --
            either because `names` was empty (the sweep was cleared, not an
            error) or because there is nowhere safe to put it (see below).

        An unsaved document (`source_path` is None) with no `workspace_dir`
        either has nowhere legitimate to write: falling back to the process's
        current directory used to drop a `materials.txt` wherever the app
        happened to be launched from, silently shadowing any other config's
        relatively-referenced list that happened to resolve there first (see
        `src.core.utils.search_paths`, which tries the name as given -- i.e.
        against the cwd -- before `base_dir`). Declining and returning None is
        deliberate: the caller is responsible for telling the user to save
        the config first, not this Qt-free class.
        """
        lines = self._material_list_lines(names)
        if not lines:
            self.values.get('2D', {}).pop('materials_list', None)
            self.dirty = True
            return None

        if target is None:
            target = (self._existing_material_list_target()
                      or self._default_material_list_target())
        if target is None:
            return None

        target.parent.mkdir(parents=True, exist_ok=True)
        _write_atomically(target, '\n'.join(lines) + '\n')
        self.set('2D', 'materials_list', str(
            self._relative_to_config(target)))
        return target

    def _default_material_list_target(self) -> Optional[Path]:
        """This config's own list file: beside the saved config, else in the
        workspace, else nowhere at all -- see `write_material_names` for why
        the process's current directory is not a fallback."""
        if self.source_path is not None:
            return self.source_path.parent / \
                f"{self.source_path.stem}_materials.txt"
        if self.workspace_dir is not None:
            return Path(self.workspace_dir) / 'materials.txt'
        return None

    def _existing_material_list_target(self) -> Optional[Path]:
        """Where a currently configured ``materials_list`` actually lives on
        disk, resolved the same way `read_material_list` resolves it (via
        `base_dir`) -- so `write_material_names` rewrites that file rather
        than silently retargeting to the config-stem default. None when
        nothing is configured yet, or the configured value does not resolve
        to an existing file (e.g. it was typed but never actually read
        successfully) -- in either case there is nothing to rewrite, only a
        fresh default to create.
        """
        current = self.get('2D', 'materials_list')
        if not isinstance(current, str) or not current:
            return None
        return _find_existing(current, self.base_dir)

    def _relative_to_config(self, path: Path) -> Path:
        """A path relative to the config's directory where that is possible.

        The existing configs use relative paths and travel with their run
        directory; absolutising them on a round trip would break that.
        """
        if self.base_dir is None:
            return path
        try:
            return Path(path).resolve().relative_to(Path(self.base_dir).resolve())
        except ValueError:
            return Path(path)

    def apply_material_directories(self, structure_dir, potential_dir,
                                   potential_extension: str,
                                   pair_mode: Optional[bool] = None) -> None:
        """Point the per-material fields at two directories.

        This is what makes a material list actually sweep: `expand_config_sweeps`
        substitutes the placeholder per material, and without them every config
        in the sweep would build the same material.

        With a second material present the two sheets take *different*
        placeholders -- `{mat1}` and `{mat2}` -- because each line of the list
        names an ordered pair and the second sheet has to move with it.
        Templating only the first sheet (which `{mat}` alone does) leaves the
        second one pinned to whatever material it happened to hold, so every
        run in the sweep would silently build the same second material: wrong
        results, not a wrong-looking config.

        Both sheets read the same two directories: one list, one structure
        directory, one potential directory -- a pair whose members live apart
        has no way to be named on a single line.

        Args:
            pair_mode: What the caller believes it is configuring -- an
                ordered-pair sweep (True) or a single-material one (False).
                None, the default, derives it from the document, which is the
                only thing that can actually be right: which markers the
                sheets take follows the presence of ``[2D-2]``, not a flag.
                A stated `pair_mode` that disagrees with the document is
                refused rather than obeyed -- obeying True without a
                ``[2D-2]`` would conjure that section out of a template
                write, and obeying False with one would leave the second
                sheet pinned to a single material for every run of the sweep,
                which is silently wrong results rather than a wrong-looking
                config.

        Raises:
            DocumentError: If `pair_mode` contradicts the document's shape.
        """
        if pair_mode is not None and pair_mode != self.has_second_material():
            raise DocumentError(
                f"pair_mode={pair_mode} contradicts this config, which has "
                f"{'a' if self.has_second_material() else 'no'} "
                f"[{self.SECOND_SHEET}] section: the {{mat1}}/{{mat2}} "
                "placeholders follow the sheets that actually exist."
            )
        self._write_material_templates(
            self._relative_to_config(Path(structure_dir)),
            self._relative_to_config(Path(potential_dir)),
            potential_extension,
        )

    def _write_material_templates(self, structures, potentials,
                                  potential_extension: str) -> None:
        """Write the placeholders into whichever sheet sections exist.

        Split out of `apply_material_directories` because the directories
        arrive already spelled the way the config should hold them: that
        caller relativises them first, while `add_second_material` reads them
        back off a config that is already templated, where relativising a
        second time would resolve a relative path against the process's
        working directory instead of the config's own.
        """
        def apply(section: str, marker: str) -> None:
            self.set(section, 'mat', marker)
            self.set(section, 'cif_path',
                     f"{structures}/{marker}{STRUCTURE_EXTENSION}")
            self.set(section, 'pot_path',
                     f"{potentials}/{marker}{potential_extension}")

        if self.has_second_material():
            apply('2D', '{mat1}')
            apply(self.SECOND_SHEET, '{mat2}')
        else:
            # Single material keeps the original `{mat}` spelling: it is what
            # every existing config and every example uses, and it substitutes
            # identically to `{mat1}`.
            apply('2D', '{mat}')

    def clear_material_directories(self) -> bool:
        """Undo apply_material_directories: drop the {mat} placeholders.

        Turning the sweep off has to leave a config that still builds. Only
        keys that actually carry a placeholder are touched -- a hand-typed
        `mat` that never went through apply_material_directories is left
        alone -- and the key is dropped rather than left with a guessed,
        non-sensical value (e.g. stripping just the filename would leave
        `cif_path = cif/`, which looks like a path but is not one). An absent
        key reads as "not set yet", which is what it is: the user must type a
        real value in before the config can build again.

        Every placeholder counts, not the literal '{mat}': a pair sweep
        templates the first sheet on {mat1} and the second on {mat2}, and
        testing for '{mat}' alone would leave both behind on untick -- a
        config that still reads as swept but has no list to expand it.

        Returns:
            Whether anything was actually removed. Callers use this to decide
            whether the mat/cif_path/pot_path editors need resyncing from the
            document -- doing that unconditionally would revert a same-commit
            edit to a plain (non-templated) 'mat' on every commit of an
            ordinary, never-swept config.
        """
        changed = False
        for section_key in ('2D', self.SECOND_SHEET):
            section = self.values.get(section_key)
            if not section:
                continue
            for key in self.TEMPLATED_KEYS:
                if has_template_marker(section.get(key)):
                    del section[key]
                    self.dirty = True
                    changed = True
        return changed

    def material_directories(self):
        """The two directories the templates point at, read back.

        Returns ``(structure_dir, potential_dir, potential_extension)``, each
        None when the config is not templated. Opening a config has to
        repopulate the pickers, or every open looks like nothing is configured.
        """
        cif = self.get('2D', 'cif_path')
        pot = self.get('2D', 'pot_path')
        if not (has_template_marker(cif) and has_template_marker(pot)):
            return (None, None, None)
        return (Path(cif).parent, Path(pot).parent, Path(pot).suffix)

    def expansion(self) -> ExpansionSummary:
        """How many configs, builds and jobs this would generate."""
        return count_expansion(self.normalised_values(), self.model,
                               base_dir=self.base_dir)

    def expansion_output_dirs(self) -> ExpansionDirs:
        """Output directory each expanded config would build into.

        Uses the same expansion `expansion()` counts, mapped through the
        same `output_dir_for` a real run would use, so what the Generate
        confirmation lists matches what generation will actually produce --
        including duplicates, which are the exact hazard the "list is not
        sweepable" rule exists to prevent.

        A config mid-edit can fail to expand at all -- an unresolvable
        material, or a pair sweep whose sheets are templated on the wrong
        marker. That is advisory information, not a validation error, so it
        never raises; but it is *reported* rather than swallowed. Returning a
        bare empty list here is how a config that expansion could explain in
        one sentence ("the [2D-2] section templates on '{mat1}'...") used to
        reach the user as an unexplained absence of directories.

        The config-level failures (`ValueError` from the sweep guards,
        `OSError`/`KeyError` from a path or section that is not there yet) are
        reported verbatim. Anything else is a bug in this program rather than
        in the config, so it is logged with its traceback and reported as
        unexpected -- still without taking the window down, because this runs
        on the GUI thread.
        """
        try:
            expanded = expand_config_sweeps(
                self.normalised_values(), self.base_dir)
            return ExpansionDirs(
                [output_dir_for(run_dict, self.model) for run_dict in expanded])
        except (ValueError, OSError, KeyError) as exc:
            return ExpansionDirs([], str(exc))
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("expansion preview failed")
            return ExpansionDirs(
                [], f"Could not work out what this config would generate: {exc}")

    def loopability_of(self, section: str, key: str) -> str:
        """How a list in this field would be interpreted.

        ``sweep`` multiplies runs, ``layer`` multiplies builds, ``in_file`` is
        looped inside LAMMPS, and ``scalar`` means a list is not supported --
        because it would not change the output directory name, so its members
        would overwrite each other.
        """
        return loopability(section, key)

    def unsupported_lists(self) -> List[Tuple[str, str]]:
        """Fields holding a list that cannot be swept."""
        return [
            (section, key)
            for section, values in self.values.items()
            for key, value in values.items()
            if isinstance(value, list) and self.loopability_of(section, key) == 'scalar'
        ]

    # --------------------------------------------------------------- output

    def normalised_values(self) -> Dict[str, Dict[str, Any]]:
        """A deep copy in the shape the CLI reads.

        The document keeps the first sheet under the plain ``2D`` key whether
        or not there is a second one, so nothing that reads ``[2D]`` has to
        know about heterostructures. On the way out that only holds for a
        single material: with a second sheet present the pair must be written
        as ``[2D-1]`` and ``[2D-2]``, because the models fold *numbered*
        sections into ``sheets`` and a lone ``[2D]`` beside a ``[2D-2]`` would
        build a one-material stack out of the wrong section.

        A single-material config still comes out as a plain ``[2D]`` -- every
        config ever written by this program and every example in the repo has
        that shape, and a bare ``[2D-1]`` would be a gratuitous churn of all
        of them.
        """
        values = copy.deepcopy(self.values)
        second = values.pop(self.SECOND_SHEET, None)
        if second is None:
            return values

        out: Dict[str, Dict[str, Any]] = {}
        for key, section in values.items():
            if key == '2D':
                # Renamed in place, with the second sheet directly after it:
                # a config where the two materials are pages apart is a
                # config nobody can read.
                out[FIRST_SHEET] = section
                out[self.SECOND_SHEET] = second
            else:
                out[key] = section
        out.setdefault(self.SECOND_SHEET, second)
        return out

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return self.normalised_values()

    def comments(self) -> Dict[str, Dict[str, str]]:
        """Canonical inline comments, rebuilt from the override metadata.

        The reader strips comments, so a round trip cannot preserve the user's.
        Re-emitting unit and help text keeps a written config at least as
        readable as a hand-written one.

        Keyed by the *written* section names, so the two sheets of a pair are
        annotated as `[2D-1]` and `[2D-2]`. Which override entry each of
        those reads is `override_for_section`'s single answer, shared with
        the form -- the two used to decide it separately, so text added to
        the second sheet's entry alone would have appeared on screen and
        never in the file.
        """
        out: Dict[str, Dict[str, str]] = {}
        for section_key, values in self.normalised_values().items():
            section_override = override_for_section(self.model, section_key)
            if not section_override:
                continue
            annotations = {}
            for key in values:
                field_override = section_override.fields.get(key)
                if not field_override:
                    continue
                parts = []
                if field_override.unit:
                    parts.append(f"in {field_override.unit}")
                if field_override.help:
                    parts.append(field_override.help)
                if parts:
                    annotations[key] = ' -- '.join(parts)
            if annotations:
                out[section_key] = annotations
        return out

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Write the config, defaulting to where it was loaded from.

        Written in the CLI's own shape (`normalised_values`): a pair goes out
        as `[2D-1]` + `[2D-2]`, a single material as a plain `[2D]`.
        """
        target = Path(path) if path is not None else self.source_path
        if target is None:
            raise DocumentError("No path to save to.")
        self._rebase_relative_paths(target.parent)

        written = write_config(
            self.normalised_values(), target,
            comments=self.comments(),
            header=f"FrictionSim2D {self.model} configuration",
        )
        self.source_path = written
        self.dirty = False
        return written

    def _rebase_relative_paths(self, new_base: Path) -> None:
        """Keep relative paths pointing at the same files when `base_dir` moves.

        A relative path means "relative to `base_dir`", so saving into a
        different directory than the one they were typed against would
        silently repoint them. Each one that resolves is rewritten relative to
        `new_base` when both sit inside the workspace (the layout travels
        together), and absolute otherwise. Unresolvable values are left alone
        so validation still reports them.

        Only for a never-saved document, whose paths were typed against the
        workspace root. A loaded config keeps its text verbatim on Save As --
        the round trip writes back what it read (see `TestRoundTrip`).
        """
        if self.source_path is not None:
            return
        old_base = self.base_dir
        if old_base is None or Path(old_base).resolve() == Path(new_base).resolve():
            return
        from .staged_paths import absolutize_staged_paths  # noqa: PLC0415

        resolved = absolutize_staged_paths(self.values, old_base)
        root = Path(self.workspace_root).resolve() if self.workspace_root else None
        new_base = Path(new_base).resolve()
        for section, values in resolved.items():
            if not isinstance(values, dict):
                continue
            for key, value in values.items():
                before = self.values[section].get(key)
                if value == before or not Path(value).is_absolute():
                    continue
                absolute = Path(value).resolve()
                if root is not None and new_base.is_relative_to(root) \
                        and absolute.is_relative_to(root):
                    value = os.path.relpath(absolute, new_base)
                self.values[section][key] = str(value)
