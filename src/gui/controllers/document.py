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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ...core.config import (
    AFMSimulationConfig,
    SheetOnSheetSimulationConfig,
    load_settings,
    parse_config,
)
from ...core.config_io import write_config
from ...core.config_validation import ConfigValidation, validate_config
from ...core.run import (
    ExpansionSummary, count_expansion, expand_config_sweeps, loopability,
    output_dir_for, read_material_list,
)
from ...core.utils import _find_existing  # noqa: PLC2701  (intentional internal reuse)
from ..schema.introspect import SectionSpec, describe_model
from ..schema.overrides import overrides_for
from ..schema.simulation_types import MODES, sections_for
from .materials import STRUCTURE_EXTENSION

__all__ = ['ConfigDocument', 'DocumentError', 'HETERO_SECTION_HINT']

#: Sections that fold into the derived `sheets` list and route the build to the
#: heterostructure builders.
_HETERO_SECTION_PREFIX = '2D-'

HETERO_SECTION_HINT = (
    "This config defines numbered sheet sections ({sections}), which build a "
    "heterostructure. Those are not editable in this version -- open it in the "
    "CLI instead."
)


class DocumentError(Exception):
    """The document cannot represent this config."""


def root_model_for(model: str):
    """The pydantic root model a simulation model validates against."""
    from ...core.run import AFM_LIKE_MODELS  # noqa: PLC0415

    return AFMSimulationConfig if model in AFM_LIKE_MODELS else SheetOnSheetSimulationConfig


@dataclass
class ConfigDocument:
    """A run config under edit."""

    model: str = 'afm'
    values: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    source_path: Optional[Path] = None
    dirty: bool = False
    #: `unexpanded_template` issues held back by the most recent validate().
    #: Not an error when a material sweep is configured -- expansion
    #: substitutes them -- but worth reporting as pending work.
    pending_template_issues: List[Any] = field(default_factory=list)

    # ------------------------------------------------------------------ load

    @classmethod
    def from_file(cls, path: Union[str, Path], model: str) -> 'ConfigDocument':
        """Load an existing config.

        Raises:
            DocumentError: If the config builds a heterostructure, which this
                version cannot represent. Refusing is better than silently
                editing a config whose numbered sheet sections would route the
                build somewhere the GUI does not describe.
        """
        path = Path(path)
        values = parse_config(path)

        hetero = sorted(s for s in values if s.startswith(_HETERO_SECTION_PREFIX))
        if hetero:
            raise DocumentError(HETERO_SECTION_HINT.format(sections=', '.join(hetero)))

        return cls(model=model, values=values, source_path=path, dirty=False)

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

    def section_lock_reason(self, key: str) -> Optional[str]:
        override = overrides_for(self.model).get(key)
        return override.required_reason if override else None

    # ----------------------------------------------------------- simulation

    def set_model(self, model: str) -> None:
        """Switch simulation model, keeping the sections the new one shares.

        Sections the new model has no use for are dropped, and its required
        ones are added with defaults, so the document always matches its model.
        """
        if model == self.model:
            return
        if model not in MODES:
            raise DocumentError(f"Unknown simulation model: {model}")

        self.model = model
        relevant = set(sections_for(model))

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
        """Directory that relative paths in this config resolve against."""
        return self.source_path.parent if self.source_path else None

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
        """
        payload = copy.deepcopy(self.values)
        payload['settings'] = (settings or load_settings()).model_dump()
        result = validate_config(
            payload, root_model_for(self.model), base_dir=self.base_dir,
        )

        self.pending_template_issues = []
        if not self.has_material_list():
            return result

        pending, hard = [], []
        for issue in result.issues:
            (pending if issue.kind == 'unexpanded_template' else hard).append(issue)
        self.pending_template_issues = pending
        return ConfigValidation(issues=hard, config=result.config)

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

    def write_material_names(self, names: List[str]) -> Optional[Path]:
        """Write the material list and point the config at it.

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
        """
        names = [n.strip() for n in names if n and n.strip()]
        if not names:
            self.values.get('2D', {}).pop('materials_list', None)
            self.dirty = True
            return None

        target = self._existing_material_list_target()
        if target is None:
            if self.source_path is not None:
                target = self.source_path.parent / \
                    f"{self.source_path.stem}_materials.txt"
            else:
                target = Path.cwd() / 'materials.txt'

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('\n'.join(names) + '\n', encoding='utf-8')
        self.set('2D', 'materials_list', str(
            self._relative_to_config(target)))
        return target

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
                                   potential_extension: str) -> None:
        """Point the per-material fields at two directories.

        This is what makes a material list actually sweep: `expand_config_sweeps`
        substitutes {mat} per material, and without the placeholders every config
        in the sweep would build the same material.
        """
        structures = self._relative_to_config(Path(structure_dir))
        potentials = self._relative_to_config(Path(potential_dir))

        self.set('2D', 'mat', '{mat}')
        self.set('2D', 'cif_path', f"{structures}/{{mat}}{STRUCTURE_EXTENSION}")
        self.set('2D', 'pot_path', f"{potentials}/{{mat}}{potential_extension}")

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

        Returns:
            Whether anything was actually removed. Callers use this to decide
            whether the mat/cif_path/pot_path editors need resyncing from the
            document -- doing that unconditionally would revert a same-commit
            edit to a plain (non-templated) 'mat' on every commit of an
            ordinary, never-swept config.
        """
        section = self.values.get('2D')
        if not section:
            return False
        changed = False
        for key in self.TEMPLATED_KEYS:
            value = section.get(key)
            if isinstance(value, str) and '{mat}' in value:
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
        if not (isinstance(cif, str) and isinstance(pot, str)
                and '{mat}' in cif and '{mat}' in pot):
            return (None, None, None)
        return (Path(cif).parent, Path(pot).parent, Path(pot).suffix)

    def expansion(self) -> ExpansionSummary:
        """How many configs, builds and jobs this would generate."""
        return count_expansion(copy.deepcopy(self.values), self.model,
                               base_dir=self.base_dir)

    def expansion_output_dirs(self) -> List[Path]:
        """Output directory each expanded config would build into.

        Uses the same expansion `expansion()` counts, mapped through the
        same `output_dir_for` a real run would use, so what the Generate
        confirmation lists matches what generation will actually produce --
        including duplicates, which are the exact hazard the "list is not
        sweepable" rule exists to prevent. A config mid-edit can fail to
        expand at all; that is advisory information, not a validation
        error, so any exception yields an empty list rather than raising.
        """
        try:
            expanded = expand_config_sweeps(
                copy.deepcopy(self.values), self.base_dir)
            return [output_dir_for(run_dict, self.model) for run_dict in expanded]
        except Exception:  # pylint: disable=broad-except
            return []

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

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return copy.deepcopy(self.values)

    def comments(self) -> Dict[str, Dict[str, str]]:
        """Canonical inline comments, rebuilt from the override metadata.

        The reader strips comments, so a round trip cannot preserve the user's.
        Re-emitting unit and help text keeps a written config at least as
        readable as a hand-written one.
        """
        out: Dict[str, Dict[str, str]] = {}
        overrides = overrides_for(self.model)
        for section_key, values in self.values.items():
            section_override = overrides.get(section_key)
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
        """Write the config, defaulting to where it was loaded from."""
        target = Path(path) if path is not None else self.source_path
        if target is None:
            raise DocumentError("No path to save to.")

        written = write_config(
            self.values, target,
            comments=self.comments(),
            header=f"FrictionSim2D {self.model} configuration",
        )
        self.source_path = written
        self.dirty = False
        return written
