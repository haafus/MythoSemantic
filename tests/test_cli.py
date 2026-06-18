from click.testing import CliRunner

from cli import mytho

runner = CliRunner()


class TestMythoTopLevel:
    def test_help(self):
        result = runner.invoke(mytho, ["--help"])
        assert result.exit_code == 0
        assert "MythoScope" in result.output

    def test_lists_all_commands(self):
        result = runner.invoke(mytho, ["--help"])
        for cmd in ["corpus", "embeddings", "projections", "graphs", "server", "build", "status"]:
            assert cmd in result.output


class TestCorpusCommand:
    def test_help(self):
        result = runner.invoke(mytho, ["corpus", "--help"])
        assert result.exit_code == 0
        assert "--force" in result.output


class TestProjectionCommand:
    def test_help(self):
        result = runner.invoke(mytho, ["projections", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--no-plots" in result.output
        assert "--motif-analysis" in result.output


class TestGraphsCommand:
    def test_help(self):
        result = runner.invoke(mytho, ["graphs", "--help"])
        assert result.exit_code == 0
        assert "--force" in result.output


class TestServerCommand:
    def test_help(self):
        result = runner.invoke(mytho, ["server", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output


class TestBuildCommand:
    def test_help(self):
        result = runner.invoke(mytho, ["build", "--help"])
        assert result.exit_code == 0
        assert "--force" in result.output
        assert "--skip-corpus" in result.output
        assert "--skip-embeddings" in result.output
        assert "--skip-projections" in result.output
        assert "--skip-graphs" in result.output
