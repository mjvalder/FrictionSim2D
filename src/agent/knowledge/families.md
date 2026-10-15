Notation: [afm] etc. is a model name; [general] force means key force in config section [general].
AFM tip family (a tip pressed onto a supported sheet):
- Dynamic + indent [afm]: approach, indent, then slide under each load. Loads: [general] force (nN, list). Speed: [general] scan_speed. Tip radius: [tip] r. Tool: setup_afm.
- Indent only [indent]: approach, hold at each load, retract; no sliding. Tool: setup_indent.
- PES scan [pes_tip]: static tip energy map over one surface cell at one low load ([general] force) on an N x N grid ([pes] grid_n). Tool: setup_pes_tip.
Sheet-on-sheet family (sheets sliding on each other):
- Dynamic (slide) [sheetonsheet]: loads are pressures ([general] pressure, GPa, list); needs at least 3 layers (4 for a heterostructure pair). Tool: setup_sheet.
- PES scan [pes_sheet]: static sheet-over-sheet energy map ([pes] grid_n). Tool: setup_pes_sheet.
PES tip and PES sheet scans are different models: pick the tool the user means.
