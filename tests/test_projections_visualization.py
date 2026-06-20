import importlib.util
import os
import sys
import types

_src = os.path.join(os.path.dirname(__file__), "..", "src")

_px = types.ModuleType("plotly.express")
_px.colors = types.SimpleNamespace(qualitative=types.SimpleNamespace(Plotly=["#111111", "#222222", "#333333"]))


class _FakeScatter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


_go = types.ModuleType("plotly.graph_objects")
_go.Scatter = _FakeScatter
_go.Figure = type("Figure", (), {})

_subplots = types.ModuleType("plotly.subplots")
_subplots.make_subplots = lambda *a, **kw: None

_pd = types.ModuleType("pandas")
_pd.DataFrame = type("DataFrame", (), {})  # type: ignore[attr-defined]

_proj_pkg = types.ModuleType("projections")
_proj_pkg.__path__ = [os.path.join(_src, "projections")]  # type: ignore[attr-defined]

_added_stubs: list[str] = []
for _name, _module in [
    ("plotly", types.ModuleType("plotly")),
    ("plotly.express", _px),
    ("plotly.graph_objects", _go),
    ("plotly.subplots", _subplots),
    ("pandas", _pd),
    ("projections", _proj_pkg),
]:
    if _name not in sys.modules:
        sys.modules[_name] = _module
        _added_stubs.append(_name)

try:
    _spec = importlib.util.spec_from_file_location(
        "projections.visualization",
        os.path.join(_src, "projections", "visualization.py"),
    )
    assert _spec is not None and _spec.loader is not None
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
finally:
    for _name in _added_stubs:
        sys.modules.pop(_name, None)

_get_color_map = _mod._get_color_map


class TestGetColorMap:
    def test_assigns_colors_to_all_traditions(self):
        data = [{"tradition": "greek"}, {"tradition": "norse"}]
        cmap = _get_color_map(data)
        assert "greek" in cmap
        assert "norse" in cmap
