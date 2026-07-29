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
from ...core.run import ExpansionSummary, count_expansion, loopability
from ..schema.introspect import SectionSpec, describe_model
from ..schema.overrides import overrides_for
from ..schema.simulation_types import MODES, sections_for

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
        return {
            f.key: f.default
            for f in section.fields
            if f.default is not None or f.required
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

    def expansion(self) -> ExpansionSummary:
        """How many configs, builds and jobs this would generate."""
        return count_expansion(copy.deepcopy(self.values), self.model,
                               base_dir=self.base_dir)

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
