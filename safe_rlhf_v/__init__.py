"""Expose templates lazily so model-only use does not import audio dependencies."""

__all__ = ['ChatTemplate']


def __getattr__(name):
    if name == 'ChatTemplate':
        from safe_rlhf_v.configs.template import ChatTemplate
        return ChatTemplate
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
