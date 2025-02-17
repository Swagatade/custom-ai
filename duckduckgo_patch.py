from duckduckgo_search import Client as DDGSClient
_orig_init = DDGSClient.__init__

def _patched_init(self, *args, **kwargs):
    kwargs.pop('proxies', None)
    _orig_init(self, *args, **kwargs)

DDGSClient.__init__ = _patched_init
