"""One log line per hard-cap trigger: `[cap] <site> <what> limit=<..> got=<..> -> <action>`.

Logging only; a site that fires per batch prints once per (site, what) for the process (= one instance)."""
_SEEN = set()


def cap_log(site, what, limit, got, action):
    key = (site, what)
    if key in _SEEN:
        return
    _SEEN.add(key)
    print(f'[cap] {site} {what} limit={limit} got={got} -> {action}', flush=True)


def reset():
    _SEEN.clear()
