"""The global settings under edit.

Presents ``GlobalSettings`` through the same small interface as
:class:`~frictionsim2d.gui.controllers.document.ConfigDocument`, so the settings tab is
built by the same form machinery as the config tab rather than a parallel one.

**The file is always written complete.** ``load_settings`` takes the first file
it finds and does not merge, so a partial file would reset what it omitted.
Groups in `HIDDEN_SETTINGS` are kept in the values and the file, but get no
form section (the ``aiida`` CLI owns them).

**A settings.yaml in the working directory is not picked up automatically.**
The search is the explicit ``--settings-file`` argument, then
``~/.config/FrictionSim2D/settings.yaml``
(:func:`frictionsim2d.core.config._settings_paths_in_precedence_order`). The GUI therefore
passes its workspace file explicitly on every build.

Qt-free.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import ValidationError

from ...core.config import GlobalSettings, load_settings, settings_origin
from ..config_validation import ConfigIssue, ConfigValidation
from ..schema.introspect import SectionSpec, describe_section
from ..schema.overrides import SETTINGS_MODEL

__all__ = ['SettingsDocument', 'SETTINGS_FILENAME', 'HIDDEN_SETTINGS']

SETTINGS_FILENAME = 'settings.yaml'

#: Settings groups the Settings tab does not show (user ruling CFG-19).
HIDDEN_SETTINGS = ('integrity',)


@dataclass
class SettingsDocument:
    """Editable global settings, backed by a complete file."""

    values: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    source_path: Optional[Path] = None
    dirty: bool = False

    #: Duck-types ConfigDocument for the shared form widgets; also selects the
    #: settings override table so the groups get labels and help.
    model: str = SETTINGS_MODEL

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, settings_file: Optional[Union[str, Path]] = None) -> 'SettingsDocument':
        """Load the effective settings.

        Args:
            settings_file: A specific file to load. Without it the normal
                resolution applies.
        """
        settings = load_settings(settings_file=settings_file)
        origin = Path(settings_file) if settings_file else _origin_or_none()
        return cls(values=settings.model_dump(mode='json'), source_path=origin)

    @classmethod
    def defaults(cls) -> 'SettingsDocument':
        return cls(values=GlobalSettings().model_dump(mode='json'))

    # ---------------------------------------------------------------- schema

    def sections(self) -> List[SectionSpec]:
        """One section per settings sub-model, except `HIDDEN_SETTINGS`."""
        specs: List[SectionSpec] = []
        for name, info in GlobalSettings.model_fields.items():
            if name in HIDDEN_SETTINGS:
                continue
            annotation = info.annotation
            if isinstance(annotation, type) and hasattr(annotation, 'model_fields'):
                specs.append(describe_section(
                    annotation, name=name, key=name, optional=False))
        return specs

    def section(self, key: str) -> Optional[SectionSpec]:
        return next((s for s in self.sections() if s.key == key), None)

    # ---------------------------------------------------------------- values

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.values.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        current = self.values.setdefault(section, {})
        if current.get(key) == value:
            return
        current[key] = value
        self.dirty = True

    def has_section(self, key: str) -> bool:
        return key in self.values

    def enable_section(self, key: str, enabled: bool) -> None:
        """Settings sections are all required; nothing to toggle."""
        if enabled and key not in self.values:
            self.values[key] = {}

    def is_section_locked(self, _key: str) -> bool:
        return True

    def section_lock_reason(self, _key: str) -> Optional[str]:
        return 'Every settings group is always present.'

    # ------------------------------------------------------------ validation

    def validate(self):
        """Validate against ``GlobalSettings``, reporting per field."""
        try:
            settings = GlobalSettings.model_validate(copy.deepcopy(self.values))
        except ValidationError as exc:
            issues = []
            for error in exc.errors():
                loc = error.get('loc', ())
                section = str(loc[0]) if len(loc) >= 1 else None
                key = str(loc[1]) if len(loc) >= 2 else None
                issues.append(ConfigIssue(
                    message=error.get('msg', 'invalid value'),
                    section=section, key=key, kind='validation'))
            return ConfigValidation(issues=issues)

        return ConfigValidation(issues=[], config=settings)

    def as_settings(self) -> GlobalSettings:
        """The validated settings object."""
        return GlobalSettings.model_validate(copy.deepcopy(self.values))

    # --------------------------------------------------------------- output

    def save(self, path: Union[str, Path]) -> Path:
        """Write the complete settings (hidden groups included) to ``path``."""
        settings = self.as_settings()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(settings.model_dump(mode='json'), sort_keys=False),
            encoding='utf-8',
        )
        self.source_path = target
        self.dirty = False
        return target

    def differs_from_defaults(self) -> Dict[str, List[str]]:
        """Which values are not the built-in default, per section."""
        defaults = GlobalSettings().model_dump(mode='json')
        changed: Dict[str, List[str]] = {}
        for section, values in self.values.items():
            if not isinstance(values, dict):
                continue
            keys = [
                key for key, value in values.items()
                if defaults.get(section, {}).get(key) != value
            ]
            if keys:
                changed[section] = keys
        return changed


def _origin_or_none() -> Optional[Path]:
    try:
        return settings_origin()
    except Exception:  # pylint: disable=broad-except
        return None
