from pathlib import Path

from settings import ServerSettings, settings


class TestServerSettingsDefaults:
    def test_host(self):
        assert settings.server.host == "127.0.0.1"

    def test_port(self):
        assert settings.server.port == 8000

    def test_gzip_minimum_size(self):
        assert settings.server.gzip_minimum_size == 1024

class TestServerSettingsOverride:
    def test_override_via_constructor(self):
        cfg = ServerSettings(host="0.0.0.0", port=9000)
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 9000

    def test_settings_server_is_server_settings(self):
        assert isinstance(settings.server, ServerSettings)


class TestServerPaths:
    def test_web_root_points_to_server_web(self):
        assert settings.web_root == Path("src/server/web")

    def test_web_root_is_regular_field(self):
        assert "web_root" in settings.model_fields

