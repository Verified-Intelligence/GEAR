"""Raised where a code path of a removed (research / neural-network verification) feature would have run."""


class NotInRelease(NotImplementedError):
    pass


def not_in_release(feature):
    raise NotInRelease(f"{feature} is not part of this release")
