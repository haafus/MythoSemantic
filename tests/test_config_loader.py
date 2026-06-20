from settings import CorpusSettings, settings


class TestCorpusSettingsDefaults:
    def test_default_max_workers(self):
        assert settings.corpus.max_workers == 10

    def test_default_timeouts(self):
        assert settings.corpus.timeout_connect == 10
        assert settings.corpus.timeout_read == 30

    def test_default_retry(self):
        assert settings.corpus.retry_total == 4
        assert settings.corpus.retry_backoff_factor == 1.5
        assert settings.corpus.retry_status_forcelist == [429, 500, 502, 503, 504]

    def test_default_parsing(self):
        assert settings.corpus.html_include_comments is False
        assert settings.corpus.html_include_tables is True
        assert settings.corpus.pdf_extract_tables is False
        assert settings.corpus.pdf_preserve_layout is True


class TestCorpusSettingsOverride:
    def test_override_via_constructor(self):
        cfg = CorpusSettings(max_workers=20, timeout_connect=5)
        assert cfg.max_workers == 20
        assert cfg.timeout_connect == 5
        assert cfg.timeout_read == 30

    def test_settings_corpus_is_corpus_settings(self):
        assert isinstance(settings.corpus, CorpusSettings)


class TestPathsFromSettings:
    def test_corpus_dir(self):
        assert settings.corpus_dir is not None

    def test_config_dir(self):
        assert settings.config_dir is not None
