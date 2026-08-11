"""Bounded semantic planner and closed-loop controller."""

from tokenpoweragent.agent.controller import TokenPowerAgent

# Public ServeCompass name with a backward-compatible class alias.  The legacy
# name remains importable so existing experiments and serialized provenance do
# not change during the branding transition.
ServeCompass = TokenPowerAgent

__all__ = ["ServeCompass", "TokenPowerAgent"]
