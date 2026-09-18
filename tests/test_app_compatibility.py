"""Studio's min_app_version_for: an agent whose tools need a newer Android
build than another agent's must say so in its manifest/registry entry, and an
agent with no such tool must be completely unaffected (see
android_app/AgentRuntime/src/portal.ts isAppCompatible for the device side)."""
from __future__ import annotations

from studio.app import TOOL_MIN_APP_VERSION, min_app_version_for


def test_no_tools_needs_no_app_version():
    assert min_app_version_for([]) is None


def test_pre_existing_tools_are_unaffected():
    tools = [{"name": "create_action_item"}, {"name": "get_todays_meetings"},
             {"name": "draft_email"}, {"name": "jira_search"}]
    assert min_app_version_for(tools) is None


def test_a_gated_tool_reports_its_version():
    assert min_app_version_for([{"name": "schedule_event"}]) == "3.1"


def test_mixed_gated_and_ungated_tools_still_gates():
    tools = [{"name": "create_action_item"}, {"name": "schedule_event"}]
    assert min_app_version_for(tools) == "3.1"


def test_picks_the_highest_requirement_among_several_gated_tools(monkeypatch):
    # Prove the max-picking logic itself, independent of what versions the
    # table happens to hold today.
    monkeypatch.setitem(TOOL_MIN_APP_VERSION, "future_tool", "10.2")
    tools = [{"name": "schedule_event"}, {"name": "future_tool"}]
    assert min_app_version_for(tools) == "10.2"


def test_a_tool_with_no_name_field_is_ignored():
    assert min_app_version_for([{"kind": "http"}]) is None
