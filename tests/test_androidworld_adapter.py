import sys
import types

from src.agent.env_adapter import AndroidWorldAdapter


class _BBox:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max


class _Element:
    def __init__(
        self,
        *,
        text=None,
        content_description=None,
        class_name="android.widget.Button",
        editable=False,
        clickable=True,
        long_clickable=False,
        scrollable=False,
        enabled=True,
        selected=False,
        checked=False,
        focused=False,
        bbox=None,
    ):
        self.text = text
        self.content_description = content_description
        self.class_name = class_name
        self.bbox = bbox or _BBox(0.1, 0.2, 0.3, 0.4)
        self.hint_text = None
        self.is_checked = checked
        self.is_checkable = False
        self.is_clickable = clickable
        self.is_editable = editable
        self.is_enabled = enabled
        self.is_focused = focused
        self.is_focusable = True
        self.is_long_clickable = long_clickable
        self.is_scrollable = scrollable
        self.is_selected = selected
        self.is_visible = True
        self.package_name = "com.example"
        self.resource_name = None
        self.resource_id = "id/example"


class _State:
    def __init__(self, ui_elements):
        self.ui_elements = ui_elements


class _Env:
    def __init__(self):
        self.foreground_activity_name = "com.example.SettingsActivity"
        self.logical_screen_size = (1080, 2400)
        self.orientation = 0
        self.executed_actions = []
        self._state = _State(
            [
                _Element(text="Wi-Fi"),
                _Element(
                    text="Search",
                    editable=True,
                    focused=True,
                    class_name="android.widget.EditText",
                ),
            ]
        )

    def execute_action(self, action):
        self.executed_actions.append(action)

    def get_state(self, wait_to_stabilize=False):
        return self._state


class _JSONAction:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_androidworld_adapter_formats_observation_and_actions():
    env = _Env()
    adapter = AndroidWorldAdapter(
        env,
        env.get_state(),
        task_name="BrowserOpenUrl",
        goal="Open the browser and search for a cafe.",
        template="Open a browser and complete the requested search.",
        score_provider=lambda: 0.0,
    )

    assert "[Task Name] BrowserOpenUrl" in adapter.observation
    assert "[0] role=Button label=Wi-Fi" in adapter.observation
    assert "focused: ui-1" in adapter.observation
    assert "input_target: ui-1" in adapter.observation
    assert "text_entry_admissible: true" in adapter.observation
    assert "tap [0]" in adapter.admissible_actions
    assert 'type "..." into [1]' in adapter.admissible_actions


def test_androidworld_adapter_only_exposes_type_for_focused_field():
    env = _Env()
    env._state = _State(
        [
            _Element(
                text="First name",
                editable=True,
                focused=False,
                class_name="android.widget.EditText",
            ),
            _Element(
                text="Last name",
                editable=True,
                focused=True,
                class_name="android.widget.EditText",
            ),
        ]
    )
    adapter = AndroidWorldAdapter(
        env,
        env.get_state(),
        task_name="ContactsAddContact",
        goal="Create a contact.",
        score_provider=lambda: 0.0,
    )

    assert 'type "..." into [0]' not in adapter.admissible_actions
    assert 'type "..." into [1]' in adapter.admissible_actions


def test_androidworld_adapter_detects_input_method_detour():
    env = _Env()
    env._state = _State(
        [
            _Element(text="Choose input method", class_name="android.widget.TextView"),
            _Element(
                text="Show virtual keyboard",
                class_name="android.widget.TextView",
            ),
            _Element(
                text=None,
                class_name="android.widget.Switch",
                clickable=True,
                checked=True,
            ),
        ]
    )
    adapter = AndroidWorldAdapter(
        env,
        env.get_state(),
        task_name="ContactsAddContact",
        goal="Create a contact.",
        score_provider=lambda: 0.0,
    )

    assert "overlay: input_method_picker" in adapter.observation
    assert "text_entry_admissible: false" in adapter.observation
    assert "navigate back" in adapter.admissible_actions


def test_androidworld_adapter_executes_indexed_action(monkeypatch):
    env = _Env()
    android_world_module = types.ModuleType("android_world")
    android_world_env_module = types.ModuleType("android_world.env")
    android_world_json_action_module = types.ModuleType("android_world.env.json_action")
    android_world_json_action_module.JSONAction = _JSONAction
    android_world_json_action_module.CLICK = "click"
    android_world_json_action_module.INPUT_TEXT = "input_text"
    android_world_json_action_module.SCROLL = "scroll"
    android_world_json_action_module.LONG_PRESS = "long_press"
    android_world_json_action_module.NAVIGATE_BACK = "navigate_back"
    android_world_json_action_module.NAVIGATE_HOME = "navigate_home"
    android_world_json_action_module.KEYBOARD_ENTER = "keyboard_enter"
    android_world_json_action_module.WAIT = "wait"

    monkeypatch.setitem(sys.modules, "android_world", android_world_module)
    monkeypatch.setitem(sys.modules, "android_world.env", android_world_env_module)
    monkeypatch.setitem(
        sys.modules,
        "android_world.env.json_action",
        android_world_json_action_module,
    )

    score = {"value": 0.0}
    adapter = AndroidWorldAdapter(
        env,
        env.get_state(),
        task_name="BrowserOpenUrl",
        goal="Open the browser and search for a cafe.",
        score_provider=lambda: score["value"],
    )

    adapter.step("tap [1]")
    assert env.executed_actions[-1].action_type == "click"
    assert env.executed_actions[-1].index == 1

    adapter.step('type "hello" into [1]')
    assert env.executed_actions[-1].action_type == "input_text"
    assert env.executed_actions[-1].index == 1
    assert env.executed_actions[-1].text == "hello"
