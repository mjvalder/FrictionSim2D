"""Migration from schema v1 to v2.

Adds columns introduced in the Phase 1 data model unification:
- size_x, size_y, stack_type, substrate_material, substrate_amorphous
- potential_type
- std_lf, std_nf, mean_lfx, std_lfx, mean_lfy, std_lfy
- ntimesteps, time_series_hash, is_complete
- status (staging pipeline)
- data_url (federated file storage)

Also creates the api_keys and schema_version tables.
"""

# Each migration is a list of SQL statements executed in order.

UP = [
    # --- New geometry columns ---
    "ALTER TABLE simulations ADD COLUMN IF NOT EXISTS size_x REAL;",
    "ALTER TABLE simulations ADD COLUMN IF NOT EXISTS size_y REAL;",
    "ALTER TABLE simulations ADD COLUMN IF NOT EXISTS stack_type TEXT;",
    "ALTER TABLE simulations ADD COLUMN IF NOT EXISTS substrate_material TEXT;",
    "ALTER TABLE simulations ADD COLUMN IF NOT EXISTS substrate_amorphous BOOLEAN;",

    # --- Potential ---
    "ALTER TABLE simulations ADD COLUMN IF NOT EXISTS potential_type TEXT;",

    # --- Extended result statistics ---
    "ALTER TABLE simulations ADD COLUMN IF NOT EXISTS std_lf REAL;",
    "ALTER TABLE simulations ADD COLUMN IF NOT EXISTS std_nf REAL;",
    "ALTER TABLE simulations ADD COLUMN IF NOT EXISTS mean_lfx REAL;",
    "ALTER TABLE simulations ADD COLUMN IF NOT EXISTS std_lfx REAL;",
    "ALTER TABLE simulations ADD COLUMN IF NOT EXISTS mean_lfy REAL;",
    "ALTER TABLE simulations ADD COLUMN IF NOT EXISTS std_lfy REAL;",

    # --- Data provenance ---
    "ALTER TABLE simulations ADD COLUMN IF NOT EXISTS ntimesteps INTEGER;",
    "ALTER TABLE simulations ADD COLUMN IF NOT EXISTS time_series_hash TEXT UNIQUE;",
    "ALTER TABLE simulations ADD COLUMN IF NOT EXISTS is_complete BOOLEAN DEFAULT true;",

    # --- Staging pipeline ---
    "ALTER TABLE simulations ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'staged';",

    # --- Federated file storage ---
    "ALTER TABLE simulations ADD COLUMN IF NOT EXISTS data_url TEXT;",

    # --- API keys table ---
    """CREATE TABLE IF NOT EXISTS api_keys (
        id          SERIAL PRIMARY KEY,
        key_hash    TEXT UNIQUE NOT NULL,
        user_name   TEXT NOT NULL,
        created_at  TIMESTAMPTZ DEFAULT now(),
        is_active   BOOLEAN DEFAULT true
    );""",

    # --- Schema version table ---
    """CREATE TABLE IF NOT EXISTS schema_version (
        version     INTEGER PRIMARY KEY,
        applied_at  TIMESTAMPTZ DEFAULT now()
    );""",

    "INSERT INTO schema_version (version) VALUES (2) ON CONFLICT (version) DO NOTHING;",
]

DOWN = [
    "ALTER TABLE simulations DROP COLUMN IF EXISTS size_x;",
    "ALTER TABLE simulations DROP COLUMN IF EXISTS size_y;",
    "ALTER TABLE simulations DROP COLUMN IF EXISTS stack_type;",
    "ALTER TABLE simulations DROP COLUMN IF EXISTS substrate_material;",
    "ALTER TABLE simulations DROP COLUMN IF EXISTS substrate_amorphous;",
    "ALTER TABLE simulations DROP COLUMN IF EXISTS potential_type;",
    "ALTER TABLE simulations DROP COLUMN IF EXISTS std_lf;",
    "ALTER TABLE simulations DROP COLUMN IF EXISTS std_nf;",
    "ALTER TABLE simulations DROP COLUMN IF EXISTS mean_lfx;",
    "ALTER TABLE simulations DROP COLUMN IF EXISTS std_lfx;",
    "ALTER TABLE simulations DROP COLUMN IF EXISTS mean_lfy;",
    "ALTER TABLE simulations DROP COLUMN IF EXISTS std_lfy;",
    "ALTER TABLE simulations DROP COLUMN IF EXISTS ntimesteps;",
    "ALTER TABLE simulations DROP COLUMN IF EXISTS time_series_hash;",
    "ALTER TABLE simulations DROP COLUMN IF EXISTS is_complete;",
    "ALTER TABLE simulations DROP COLUMN IF EXISTS status;",
    "ALTER TABLE simulations DROP COLUMN IF EXISTS data_url;",
    "DROP TABLE IF EXISTS api_keys;",
    "DELETE FROM schema_version WHERE version = 2;",
]
