from pathlib import Path

from settings import ServerSettings, settings


class TestServerSettingsDefaults:
    def test_host(self):
        assert settings.server.host == "127.0.0.1"

    def test_port(self):
        assert settings.server.port == 8000

    def test_gzip_minimum_size(self):
        assert settings.server.gzip_minimum_size == 1024

    def test_cache_max_age(self):
        assert settings.server.cache_max_age == 86400

    def test_search_defaults(self):
        assert settings.server.search_job_ttl_seconds == 1800
        assert settings.server.search_max_workers == 1


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

    def test_assets_under_web_root(self):
        assert settings.assets_dir == settings.web_root / "assets"

