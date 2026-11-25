"""Synthetic hook decorator (the audit detects it by name only)."""


def conference_hook(hook_id):
    def decorate(function):
        function.conference_hook_id = hook_id
        return function

    return decorate
