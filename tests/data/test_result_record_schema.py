"""ResultRecord is the one schema shared by the store, the DB, the API and the client."""
import pytest

from src.data.models import ResultRecord, ResultStatus

EXPECTED = [
    'id', 'uploaded_at', 'uploader', 'contact', 'run_uuid', 'result_uuid',
    'set_label', 'source_host', 'model', 'family',
    'materials', 'material', 'stack_type', 'layers', 'size_x', 'size_y',
    'temperature', 'tip_material', 'tip_radius', 'substrate_material',
    'substrate_amorphous', 'potential_type', 'potential_name', 'pot_sha256',
    'git_commit',
    'force_nN', 'pressure_gpa', 'scan_angle', 'scan_speed',
    'mean_cof', 'std_cof', 'mean_lf', 'std_lf', 'mean_nf', 'std_nf',
    'mean_lfx', 'std_lfx', 'mean_lfy', 'std_lfy',
    'ntimesteps', 'time_series_hash', 'is_complete',
    'extras', 'status', 'notes', 'data_url',
]


def test_field_list_and_order():
    assert list(ResultRecord.model_fields) == EXPECTED


def test_simulation_type_and_metadata_are_gone():
    assert 'simulation_type' not in ResultRecord.model_fields
    assert 'metadata' not in ResultRecord.model_fields


def test_status_defaults_to_staged():
    assert ResultRecord(material='h-MoS2').status == ResultStatus.STAGED.value


def test_pointer_fields_are_carried():
    rec = ResultRecord(material='h-MoS2', run_uuid='r-1', result_uuid='x-1',
                       set_label='260910-afm', source_host='tuxedo',
                       contact='matteo@example.org')

    assert rec.run_uuid == 'r-1'
    assert rec.to_upload_dict()['contact'] == 'matteo@example.org'


def test_force_and_pressure_are_still_exclusive():
    with pytest.raises(ValueError, match='not both'):
        ResultRecord(material='h-MoS2', force_nN=30.0, pressure_gpa=1.0)


def test_ddl_column_order_matches_the_record():
    from src.data.database import SCHEMA_VERSION, _COLUMN_NAMES, _CREATE_TABLE_SQL

    assert SCHEMA_VERSION == 3
    assert _COLUMN_NAMES == list(ResultRecord.model_fields)

    body = _CREATE_TABLE_SQL.split('(', 1)[1].rsplit(')', 1)[0]
    ddl_cols = [line.strip().split()[0] for line in body.splitlines()
                if line.strip() and not line.strip().startswith('--')]
    assert ddl_cols == _COLUMN_NAMES


def test_ddl_declares_the_new_types():
    from src.data.database import _CREATE_TABLE_SQL

    assert 'materials           TEXT[]' in _CREATE_TABLE_SQL
    assert 'extras              JSONB' in _CREATE_TABLE_SQL
    assert 'time_series_hash    TEXT UNIQUE' in _CREATE_TABLE_SQL
    assert "status              TEXT DEFAULT 'staged'" in _CREATE_TABLE_SQL
    assert 'simulation_type' not in _CREATE_TABLE_SQL


def test_upload_from_aiida_is_gone():
    from src.data.database import FrictionDB

    assert not hasattr(FrictionDB, 'upload_from_aiida')
