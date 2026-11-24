"""Store helpers for the analysis tests."""
from frictionsim2d.analysis.store import find_result_node, result_node, table_from_csv


def run_analysis(name, dataset, params=None, *, force=False):
    """The table of ``result_node``: a reuse and a fresh run read back alike."""
    return table_from_csv(result_node(name, dataset, params, force=force).get_content(mode='rb'))


def find_result(name, dataset, params_hash=None, version=None):
    """The newest stored table of ``find_result_node``, or None."""
    node = find_result_node(name, dataset, params_hash, version)
    return None if node is None else table_from_csv(node.get_content(mode='rb'))
