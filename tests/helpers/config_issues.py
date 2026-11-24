"""Filter a `ConfigValidation` report by section and key (test-side helper)."""


def issues_for(result, section, key=None):
    """Issues attached to ``section``, or to one ``key`` within it."""
    return [i for i in result.issues
            if i.section == section and (key is None or i.key == key)]
