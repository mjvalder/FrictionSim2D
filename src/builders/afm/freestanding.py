"""Freestanding (substrate-free) tip-on-slab simulation builders.

A self-supported multilayer slab of the sheet material replaces the amorphous
substrate: the bottom layer (``layer_1``) is held fixed and the upper layers
relax, so the tip-felt PES corrugation is not contaminated by a disordered
sheet-substrate registry. Selected automatically when a ``pes-tip`` config has
no ``[sub]`` section (see ``run.simulate``). Reuses the entire AFM build via the
substrate-optional path in :class:`~src.builders.afm.AFMSimulation`.
"""

import logging

from .pes_tip import PESTipSimulation, PES_TIP_CONTACT_GAP

logger = logging.getLogger(__name__)


class FreestandingPESTipSimulation(PESTipSimulation):
    """Tip PES on a self-supported multilayer slab (no substrate)."""

    FAMILY = 'pes_tip'

    def _get_hpc_job_name(self) -> str:
        return f"pes_tip_free_{self.config.sheet.mat}"

    def _eval_mode(self) -> str:
        pes = getattr(self.config, 'pes', None)
        return getattr(pes, 'eval_mode', 'minimize') if pes is not None else 'minimize'

    def _md_steps(self) -> int:
        pes = getattr(self.config, 'pes', None)
        return int(getattr(pes, 'md_steps', 2000)) if pes is not None else 2000

    def _log_resolved_config(self, n_layers: int) -> None:
        pes = getattr(self.config, 'pes', None)
        load = getattr(pes, 'tip_load', None) if pes is not None else None
        logger.info(
            "Resolved: freestanding tip PES | %d-layer slab | no substrate | "
            "eval_mode=%s | contact gap=%.2f Angstrom (fixed; tip_load=%s nN is the "
            "external gap-calibration target, not enforced by this script)",
            n_layers, self._eval_mode(), PES_TIP_CONTACT_GAP, load,
        )

    def write_inputs(self, n_layers: int) -> None:
        """Render the freestanding tip PES scan to ``lammps/slide.in``."""
        logger.info("Writing freestanding tip PES scan inputs...")

        if n_layers < 2:
            raise ValueError(
                "A freestanding tip PES needs a fixed bottom layer + at least one "
                f"relaxed layer; set [2D] layers >= 2 (got {n_layers})."
            )

        self._log_resolved_config(n_layers)

        # Shared setup from PESTipSimulation._pes_scan_context; only the two
        # keys this family adds (and its own template) differ.
        context, out_dir = self._pes_scan_context(n_layers)
        context.update({
            'eval_mode': self._eval_mode(),
            'md_steps': self._md_steps(),
        })

        scan_script = self.render_template("afm_freestanding/pes_scan.lmp", context)
        self.write_file("lammps/slide.in", scan_script, out_dir)

        # Same as PESTipSimulation: overriding write_inputs dropped the
        # provenance record along with the rest of the parent body.
        self._write_provenance(n_layers, context, out_dir)
        logger.info("Freestanding tip PES scan written to %s/lammps/", out_dir)
