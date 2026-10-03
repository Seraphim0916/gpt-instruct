from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.8-3.10
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "codex_instruct",
    PROJECT_ROOT / "codex-instruct.py",
)
assert SPEC and SPEC.loader
codex_instruct = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(codex_instruct)


class CodexDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name).resolve()
        self.home = self.root / "home"
        self.codex_home = self.home / ".codex"
        self.codex_home.mkdir(parents=True)
        self.config_path = self.codex_home / "config.toml"
        self.original_config = 'model = "test-model"\n'
        self.config_path.write_text(self.original_config, encoding="utf-8")

        previous_cwd = Path.cwd()
        os.chdir(self.home)
        self.addCleanup(os.chdir, previous_cwd)
        home_patch = patch.object(Path, "home", return_value=self.home)
        home_patch.start()
        self.addCleanup(home_patch.stop)
        env_patch = patch.dict(os.environ, {"CODEX_HOME": ".codex"})
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def test_relative_and_absolute_aliases_are_returned_once(self) -> None:
        for env_home in [".codex", "./.codex", str(self.codex_home)]:
            with self.subTest(env_home=env_home):
                with patch.dict(os.environ, {"CODEX_HOME": env_home}):
                    self.assertEqual(codex_instruct.find_codex_dirs(), [self.codex_home])
        self.assertEqual(self.config_path.read_text(encoding="utf-8"), self.original_config)

    def test_symlink_alias_is_returned_once(self) -> None:
        alias = self.root / "codex-alias"
        try:
            alias.symlink_to(self.codex_home, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"Directory symlinks are unavailable: {exc}")
        with patch.dict(os.environ, {"CODEX_HOME": str(alias)}):
            self.assertEqual(codex_instruct.find_codex_dirs(), [self.codex_home])
        self.assertTrue(alias.is_symlink())

    def test_distinct_config_directories_are_both_returned_in_sorted_order(self) -> None:
        other_home = self.root / "another-codex"
        other_home.mkdir()
        (other_home / "config.toml").write_text('model = "other-model"\n', encoding="utf-8")
        with patch.dict(os.environ, {"CODEX_HOME": str(other_home)}):
            self.assertEqual(
                codex_instruct.find_codex_dirs(), sorted([self.codex_home, other_home])
            )

    def test_directories_without_config_are_excluded(self) -> None:
        other_home = self.root / "empty-codex"
        other_home.mkdir()
        with patch.dict(os.environ, {"CODEX_HOME": str(other_home)}):
            self.assertEqual(codex_instruct.find_codex_dirs(), [self.codex_home])
            self.config_path.unlink()
            self.assertEqual(codex_instruct.find_codex_dirs(), [])
        self.assertEqual(list(other_home.iterdir()), [])

    def test_deploy_visits_aliased_config_directory_once(self) -> None:
        source = self.root / "source.md"
        source.write_text("synthetic prompt\n", encoding="utf-8")
        destination = self.codex_home / "custom.md"
        args = SimpleNamespace(codex_dir=None, dry_run=False)
        with patch.object(
            codex_instruct, "backup_file", wraps=codex_instruct.backup_file
        ) as backup, patch.object(
            codex_instruct, "atomic_write_text", wraps=codex_instruct.atomic_write_text
        ) as write, patch("builtins.print"):
            result = codex_instruct.deploy_prompt(args, source, destination.name)

        self.assertEqual(result, 0)
        backup.assert_called_once_with(self.config_path)
        self.assertEqual(
            [call.args[0] for call in write.call_args_list].count(destination), 1
        )
        snapshots = list(self.codex_home.glob("config.toml.bak_*"))
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].read_text(encoding="utf-8"), self.original_config)
        self.assertEqual(destination.read_bytes(), source.read_bytes())
        self.assertEqual(
            tomllib.loads(self.config_path.read_text(encoding="utf-8")),
            {"model": "test-model", "model_instructions_file": "./custom.md"},
        )


class ManagedConfigTests(unittest.TestCase):
    def test_reset_refuses_unrecognized_equivalent_paths_before_writes(self) -> None:
        for absolute in (False, True):
            with self.subTest(absolute=absolute):
                directory, config = self.make_config('model = "fixture"\n')
                with directory:
                    root = config.parent
                    source = root / "source.md"
                    source.write_text("synthetic prompt\n", encoding="utf-8")
                    result = self.run_cli_in_fixture(root, ["--file", str(source),
                        "--name", "custom.md", "--codex-dir", str(root)])
                    self.assertEqual(result.returncode, 0, result.stderr)
                    reference = str(root / "custom.md") if absolute else "././custom.md"
                    config.write_text('model_instructions_file = ' + json.dumps(reference)
                                      + '\n', encoding="utf-8")
                    before = self.fixture_contents(root)
                    result = self.run_cli_in_fixture(root,
                        ["--reset", "--codex-dir", str(root)], answer="y\n")
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertEqual(self.fixture_contents(root), before)

    def test_reset_decodes_equivalent_toml_prompt_strings(self) -> None:
        values = ('"\\u002e/custom\\u002emd"', '"\\u002e\\u002fcustom.md"',
                  "'./custom.md'", '"""./custom.md"""', "'''./custom.md'''")
        for value in values:
            with self.subTest(value=value):
                directory, config = self.make_config('model = "fixture"\n')
                with directory:
                    root = config.parent
                    source = root / "source.md"
                    source.write_text("synthetic prompt\n", encoding="utf-8")
                    args = ["--file", str(source), "--name", "custom.md",
                            "--codex-dir", str(root)]
                    self.assertEqual(self.run_cli_in_fixture(root, args).returncode, 0)
                    config.write_text('model = "fixture"\nmodel_instructions_file = '
                                      + value + ' # hand edited\n', encoding="utf-8")
                    result = self.run_cli_in_fixture(
                        root, ["--reset", "--codex-dir", str(root)], answer="y\n"
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(config.read_text(encoding="utf-8"),
                                     'model = "fixture"\n')
                    self.assertFalse((root / "custom.md").exists())
                    self.assertFalse(codex_instruct.state_file_path(config).exists())

    def assert_deployment_refused_unchanged(self, config: str, baseline=None) -> None:
        for dry_run in (False, True):
            with self.subTest(dry_run=dry_run):
                directory, config_path = self.make_config(config)
                with directory:
                    root = config_path.parent
                    source = root / "source.md"
                    source.write_text("synthetic prompt\n", encoding="utf-8")
                    if baseline is not None:
                        codex_instruct.baseline_backup_path(config_path).write_text(
                            baseline, encoding="utf-8"
                        )
                    before = self.fixture_contents(root)
                    arguments = ["--file", str(source), "--name", "custom.md",
                                 "--codex-dir", str(root)]
                    if dry_run:
                        arguments.append("--dry-run")
                    result = self.run_cli_in_fixture(root, arguments)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertEqual(self.fixture_contents(root), before)

    def test_deploy_refuses_unterminated_config_before_writes(self) -> None:
        for text in ('model = "unfinished', 'values = [1, 2',
                     'value = { a = 1', 'value = """unfinished',
                     '[features]\nvalue = "unfinished'):
            with self.subTest(text=text):
                self.assert_deployment_refused_unchanged(text)

    def test_deploy_refuses_instruction_tables_before_writes(self) -> None:
        for header in ('[model_instructions_file]', '[[model_instructions_file]]',
                       '["model_instructions_file"]',
                       '[model_instructions_file.nested]'):
            with self.subTest(header=header):
                self.assert_deployment_refused_unchanged(header + '\nvalue = "x"\n')

    def test_deploy_refuses_unsupported_legacy_baseline_before_writes(self) -> None:
        for baseline in ('"model_instructions_file" = "./old.md"\n',
                         'model_instructions_file.path = "./old.md"\n',
                         'model_instructions_file = """\n./old.md\n"""\n'):
            with self.subTest(baseline=baseline):
                self.assert_deployment_refused_unchanged(
                    'model_instructions_file = "./custom.md"\n', baseline
                )

    def test_temporary_home_alias_has_canonical_expected_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            actual = root / "actual"
            actual.mkdir()
            alias = root / "alias"
            alias.symlink_to(actual, target_is_directory=True)
            with patch.object(tempfile, "tempdir", str(alias)):
                directory, config = self.make_config("")
                self.addCleanup(directory.cleanup)
                self.assertEqual(config, config.resolve())

    def make_config(self, text: str) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary_directory = tempfile.TemporaryDirectory()
        config_path = Path(temporary_directory.name).resolve() / "config.toml"
        config_path.write_text(text, encoding="utf-8")
        return temporary_directory, config_path

    def run_cli_in_fixture(self, root: Path, arguments: list[str], answer: str = ""):
        environment = os.environ.copy()
        environment.update(
            HOME=str(root), CODEX_HOME=str(root),
            XDG_CONFIG_HOME=str(root / "xdg"), TMPDIR=str(root),
        )
        return subprocess.run(
            [sys.executable, "-I", str(PROJECT_ROOT / "codex-instruct.py"), *arguments],
            cwd=root, env=environment, input=answer, capture_output=True,
            encoding="utf-8", timeout=10,
        )

    def fixture_contents(self, root: Path) -> dict:
        return {
            str(path.relative_to(root)): None if path.is_dir() else path.read_bytes()
            for path in root.rglob("*")
        }

    def test_cli_reports_file_type_errors_without_tracebacks_or_changes(self) -> None:
        for case in ["target-file", "config-directory-deploy", "config-directory-reset"]:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "source.md"
                source.write_text("synthetic prompt\n", encoding="utf-8")
                target = root / "codex"
                if case == "target-file":
                    target.write_text("keep this existing file\n", encoding="utf-8")
                else:
                    config = target / "config.toml"
                    config.mkdir(parents=True)
                    (config / "notes.txt").write_text("keep this folder\n", encoding="utf-8")
                arguments = (
                    ["--reset"] if case.endswith("reset")
                    else ["--file", str(source), "--name", "custom.md"]
                ) + ["--codex-dir", str(target)]
                before = self.fixture_contents(root)
                result = self.run_cli_in_fixture(root, arguments, "y\n")
                self.assertEqual(result.returncode, 2)
                self.assertIn("File operation failed", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertNotIn("keep this", result.stderr)
                self.assertEqual(self.fixture_contents(root), before)

    def test_cli_preserves_preview_cancel_success_and_argument_error_behavior(self) -> None:
        original = 'model = "test-model"\n'
        temporary_directory, config_path = self.make_config(original)
        self.addCleanup(temporary_directory.cleanup)
        root = config_path.parent
        source = root / "source.md"
        source.write_text("synthetic prompt\n", encoding="utf-8")
        deploy = ["--file", str(source), "--name", "custom.md", "--codex-dir", str(root)]
        reset = ["--reset", "--codex-dir", str(root)]
        before = self.fixture_contents(root)
        preview = self.run_cli_in_fixture(root, deploy + ["--dry-run"])
        self.assertEqual(preview.returncode, 0)
        self.assertEqual(preview.stderr, "")
        self.assertEqual(self.fixture_contents(root), before)
        self.assertEqual(self.run_cli_in_fixture(root, deploy).returncode, 0)
        deployed = self.fixture_contents(root)
        self.assertEqual(self.run_cli_in_fixture(root, reset, "n\n").returncode, 0)
        self.assertEqual(self.fixture_contents(root), deployed)
        self.assertEqual(self.run_cli_in_fixture(root, reset, "y\n").returncode, 0)
        self.assertEqual(config_path.read_text(encoding="utf-8"), original)
        self.assertFalse((root / "custom.md").exists())
        before_invalid_argument = self.fixture_contents(root)
        invalid = self.run_cli_in_fixture(root, ["--version", "gpt-6-v1"])
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("--version requires --apply", invalid.stderr)
        self.assertNotIn("File operation failed", invalid.stderr)
        self.assertEqual(self.fixture_contents(root), before_invalid_argument)

    def test_cli_does_not_swallow_non_filesystem_exceptions(self) -> None:
        for error in [ValueError("unexpected value"), RuntimeError("unexpected failure")]:
            with self.subTest(error=type(error).__name__), patch.object(
                sys, "argv", ["codex-instruct.py", "--file", "source.md"]
            ), patch.object(codex_instruct, "deploy_prompt", side_effect=error):
                with self.assertRaises(type(error)):
                    codex_instruct.main()

    def test_prompt_filename_validation_preserves_safe_unicode_names(self) -> None:
        invalid = [
            None, 7, "", ".", "..", "config.toml", "../prompt.md",
            "/tmp/prompt.md", "folder/prompt.md", r"folder\prompt.md",
            r"C:\prompt.md", r"\\server\prompt.md", 'quoted"name.md',
        ]
        invalid.extend(f"prompt{chr(code)}name.md" for code in range(32))
        invalid.extend(f"prompt{char}name.md" for char in "\x7f\x85\u2028\u2029\ud800\udfff")
        for filename in invalid:
            with self.subTest(filename=filename):
                self.assertFalse(codex_instruct.is_safe_prompt_filename(filename))
        for filename in ["提示詞.md", "author's notes.md", "test-🧪.md", "prompt.MD"]:
            with self.subTest(filename=filename):
                self.assertTrue(codex_instruct.is_safe_prompt_filename(filename))

    def test_unsafe_custom_name_is_rejected_before_any_deployment_write(self) -> None:
        for filename in [
            'prompt"\ninjected_setting = true\n# .md',
            'quoted"name.md', "nul\0name.md", "line\u2028name.md", "broken\ud800.md",
            r"..\prompt.md",
        ]:
            with self.subTest(filename=filename):
                temporary_directory, config_path = self.make_config(
                    'model = "test-model"\nmodel_provider = "fixture"\n'
                )
                with temporary_directory:
                    root = config_path.parent
                    source = root / "source.md"
                    source.write_text("synthetic prompt\n", encoding="utf-8")
                    before = {path.name: path.read_bytes() for path in root.iterdir()}
                    with patch.object(sys, "argv", [
                        "codex-instruct.py", "--file", str(source),
                        "--name", filename, "--codex-dir", str(root),
                    ]), patch("builtins.print"):
                        result = codex_instruct.main()
                    self.assertEqual(result, 2)
                    self.assertEqual(
                        {path.name: path.read_bytes() for path in root.iterdir()}, before
                    )
                    self.assertFalse(codex_instruct.state_file_path(config_path).exists())

    def test_unicode_custom_name_deploy_reset_preserves_unrelated_config(self) -> None:
        original = 'model = "test-model"\n[features]\nweb_search = true\n'
        temporary_directory, config_path = self.make_config(original)
        self.addCleanup(temporary_directory.cleanup)
        root = config_path.parent
        source = root / "source.md"
        source.write_text("synthetic prompt\n", encoding="utf-8")
        filename = "個人's notes 🧪.md"
        args = SimpleNamespace(codex_dir=str(root), dry_run=False)

        self.assertEqual(codex_instruct.deploy_prompt(args, source, filename), 0)
        deployed = tomllib.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(deployed["model_instructions_file"], f"./{filename}")
        self.assertEqual(deployed["model"], "test-model")
        self.assertTrue(deployed["features"]["web_search"])
        self.assertEqual((root / filename).read_bytes(), source.read_bytes())
        with patch("builtins.input", return_value="y"):
            self.assertEqual(codex_instruct.reset_managed_install(args), 0)
        self.assertEqual(config_path.read_bytes(), original.encode("utf-8"))
        self.assertFalse((root / filename).exists())
        self.assertFalse(codex_instruct.state_file_path(config_path).exists())

    def test_gpt56_v45_is_the_stable_default_release(self) -> None:
        self.assertEqual(codex_instruct.DEFAULT_PROMPT_VERSION, "gpt-5.6-v45")
        self.assertEqual(
            codex_instruct.DEFAULT_PROMPT_MD_FILENAME,
            "gpt-5.6-sol-v45.md",
        )
        self.assertEqual(
            codex_instruct.DEFAULT_PROMPT_ARCHIVE.name,
            "gpt-5.6-sol-v45.zip",
        )
        self.assertEqual(
            {
                version: (archive.name, md_filename)
                for version, (archive, md_filename) in codex_instruct.PROMPT_VERSIONS.items()
            },
            {
                "gpt-5.6-v45": ("gpt-5.6-sol-v45.zip", "gpt-5.6-sol-v45.md"),
                "gpt-6-v1": ("gpt-6-astra-v1.zip", "gpt-6-astra-v1.md"),
            },
        )
        self.assertIn("gpt-6-astra-v1.md", codex_instruct.MANAGED_PROMPT_FILENAMES)
        self.assertEqual(
            codex_instruct.LEGACY_MANAGED_PROMPT_FILENAMES,
            {
                "gpt-5.6-sol-unrestricted-v45.md",
                "gpt-5.6-sol-unrestricted-v5.md",
                "gpt-5.6-sol-unrestricted-v35.md",
                "gpt-5.6-sol-unrestricted-v41.md",
                "gpt-5.6-sol-unrestricted-v41-skills.md",
                "gpt-5.6-sol-unrestricted-v42.md",
                "gpt-6-astra-v1-rc1.md",
            },
        )

    def test_cli_deploys_selected_v1_archive(self) -> None:
        temporary_directory, config_path = self.make_config('model = "gpt-6-astra"\n')
        self.addCleanup(temporary_directory.cleanup)
        codex_home = config_path.parent

        with patch.object(
            sys,
            "argv",
            [
                "codex-instruct.py",
                "--apply",
                "--version",
                "gpt-6-v1",
                "--codex-dir",
                str(codex_home),
            ],
        ):
            result = codex_instruct.main()

        self.assertEqual(result, 0)
        deployed = codex_home / "gpt-6-astra-v1.md"
        archive_path, archive_md_filename = codex_instruct.PROMPT_VERSIONS[
            "gpt-6-v1"
        ]
        with zipfile.ZipFile(archive_path) as archive:
            self.assertEqual(archive.namelist(), [archive_md_filename])
            expected_prompt = archive.read(archive_md_filename)
        self.assertEqual(deployed.read_bytes(), expected_prompt)
        self.assertIn(
            'model_instructions_file = "./gpt-6-astra-v1.md"',
            config_path.read_text(encoding="utf-8"),
        )

    def test_interactive_menu_selects_each_packaged_version(self) -> None:
        with patch("builtins.input", return_value="1"):
            self.assertEqual(
                codex_instruct.interactive_action(),
                "apply:gpt-5.6-v45",
            )
        with patch("builtins.input", return_value="2"):
            self.assertEqual(
                codex_instruct.interactive_action(),
                "apply:gpt-6-v1",
            )

    # A pre-existing top-level instruction line, including its comment, survives deploy/reset.
    def test_round_trip_restores_previous_instruction_entry(self) -> None:
        original_line = 'model_instructions_file = "./personal-instructions.md" # keep me'
        temporary_directory, config_path = self.make_config(
            f'model = "gpt-5.5"\n{original_line}\n'
        )
        self.addCleanup(temporary_directory.cleanup)

        codex_instruct.prepare_deployment_state(
            config_path,
            "gpt-5.6-sol-unrestricted-v45.md",
            "test instructions\n",
        )
        codex_instruct.set_model_instructions(
            config_path,
            "gpt-5.6-sol-unrestricted-v45.md",
        )
        changed, status = codex_instruct.restore_managed_model_instructions(config_path)

        self.assertTrue(changed)
        self.assertEqual(status, "restored")
        self.assertIn(original_line, config_path.read_text(encoding="utf-8"))

    # Assignments inside TOML tables are not mistaken for the managed top-level field.
    def test_nested_assignment_is_not_rewritten(self) -> None:
        nested_line = 'model_instructions_file = "nested-value.md"'
        temporary_directory, config_path = self.make_config(
            f'[profile.test]\n{nested_line}\n'
        )
        self.addCleanup(temporary_directory.cleanup)

        codex_instruct.prepare_deployment_state(
            config_path,
            "gpt-5.6-sol-unrestricted-v45.md",
            "test instructions\n",
        )
        codex_instruct.set_model_instructions(
            config_path,
            "gpt-5.6-sol-unrestricted-v45.md",
        )
        codex_instruct.restore_managed_model_instructions(config_path)

        text = config_path.read_text(encoding="utf-8")
        self.assertEqual(text.count(nested_line), 1)
        self.assertNotIn("gpt-5.6-sol-unrestricted-v45.md", text)

    def test_multiline_toml_values_survive_deploy_and_reset(self) -> None:
        for delimiter in ['"""', "'''"]:
            for previous_line in [None, 'model_instructions_file = "./personal.md" # keep']:
                with self.subTest(delimiter=delimiter, previous_line=previous_line):
                    quoted_content = (
                        r'An escaped delimiter: \""" stays inside the string.'
                        if delimiter == '"""' else "One ' and two '' literal quotes."
                    )
                    unrelated = (
                        '# A comment containing """ is not a string opener.\n'
                        'label = "Escaped \\"quote\\" and # text"\n'
                        f'developer_instructions = {delimiter}\n'
                        f'{quoted_content}\n'
                        'model_instructions_file = "./example.md"\n'
                        '[example.table]\n'
                        'model = "example-only"\n'
                        f'{delimiter}\n'
                    )
                    original = unrelated + 'model = "test-model"\n'
                    if previous_line:
                        original += previous_line + "\n"
                    original += '\n[features]\nweb_search = true\n'
                    parsed_original = tomllib.loads(original)
                    temporary_directory, config_path = self.make_config(original)
                    with temporary_directory:
                        root = config_path.parent
                        source = root / "source.md"
                        source.write_text("synthetic prompt\n", encoding="utf-8")
                        args = SimpleNamespace(codex_dir=str(root), dry_run=False)
                        self.assertEqual(
                            codex_instruct.top_level_model_instructions_line(original), previous_line
                        )
                        self.assertEqual(codex_instruct.deploy_prompt(args, source, "custom.md"), 0)
                        deployed_text = config_path.read_text(encoding="utf-8")
                        deployed = tomllib.loads(deployed_text)
                        self.assertEqual(deployed["model_instructions_file"], "./custom.md")
                        self.assertEqual(
                            {key: value for key, value in deployed.items() if key != "model_instructions_file"},
                            {key: value for key, value in parsed_original.items() if key != "model_instructions_file"},
                        )
                        self.assertTrue(deployed_text.startswith(unrelated))
                        with patch("builtins.input", return_value="y"):
                            self.assertEqual(codex_instruct.reset_managed_install(args), 0)
                        self.assertEqual(config_path.read_bytes(), original.encode("utf-8"))

    def test_config_insertion_skips_multiline_array_string_contents(self) -> None:
        original = (
            'notes = [\n  """\nmodel = "example"\n[example.table]\n""",\n'
            '  "ordinary string", # ] is only a comment\n]\n'
            '\n[features]\nweb_search = true\n'
        )
        updated = codex_instruct.replace_top_level_model_instructions(
            original, 'model_instructions_file = "./custom.md"'
        )
        parsed = tomllib.loads(updated)
        self.assertEqual(parsed["model_instructions_file"], "./custom.md")
        self.assertEqual(parsed["notes"], tomllib.loads(original)["notes"])
        self.assertEqual(
            codex_instruct.replace_top_level_model_instructions(updated, None), original
        )

    def test_unicode_line_separators_in_comments_preserve_root_settings(self) -> None:
        for separator in ["\u0085", "\u2028", "\u2029"]:
            for previous_line in [None, 'model_instructions_file = "./personal.md" # keep']:
                with self.subTest(separator=repr(separator), previous_line=previous_line):
                    original = (
                        f'# Documentation excerpt:{separator}[features]\n'
                        f'model = "test-model" # Model notes:{separator}[profile.example]\n'
                    )
                    if previous_line:
                        original += previous_line + "\n"
                    original += '[profiles.work]\nmodel = "nested-model"\n'
                    parsed_original = tomllib.loads(original)
                    temporary_directory, config_path = self.make_config(original)
                    with temporary_directory:
                        root = config_path.parent
                        source = root / "source.md"
                        source.write_text("synthetic prompt\n", encoding="utf-8")
                        args = SimpleNamespace(codex_dir=str(root), dry_run=False)
                        with patch("builtins.print"):
                            result = codex_instruct.deploy_prompt(args, source, "custom.md")
                        self.assertEqual(result, 0)
                        expected = dict(parsed_original, model_instructions_file="./custom.md")
                        self.assertEqual(
                            tomllib.loads(config_path.read_text(encoding="utf-8")), expected
                        )
                        with patch("builtins.print"), patch("builtins.input", return_value="y"):
                            self.assertEqual(codex_instruct.reset_managed_install(args), 0)
                        self.assertEqual(config_path.read_bytes(), original.encode("utf-8"))

    def test_unicode_separators_in_multiline_values_survive_config_edits(self) -> None:
        for separator in ["\u0085", "\u2028", "\u2029"]:
            for delimiter in ['"""', "'''"]:
                with self.subTest(separator=repr(separator), delimiter=delimiter):
                    original = (
                        f'notes = {delimiter}First{separator}second\n'
                        f'Third{separator}[features]{delimiter}\n'
                        'model = "test-model"\n[profiles.work]\nmodel = "nested-model"\n'
                    )
                    expected = tomllib.loads(original)
                    expected["model_instructions_file"] = "./custom.md"
                    updated = codex_instruct.replace_top_level_model_instructions(
                        original, 'model_instructions_file = "./custom.md"'
                    )
                    self.assertEqual(tomllib.loads(updated), expected)
                    self.assertEqual(
                        codex_instruct.replace_top_level_model_instructions(updated, None), original
                    )

    def test_config_line_boundaries_preserve_crlf_and_unterminated_last_line(self) -> None:
        previous = 'model_instructions_file = "./personal.md"'
        replacement = 'model_instructions_file = "./custom.md"'
        for newline in ["\n", "\r\n"]:
            for last_newline in ["", newline]:
                with self.subTest(newline=repr(newline), last_newline=repr(last_newline)):
                    original = (
                        f'# Documentation excerpt:\u2028[features]{newline}'
                        f'model = "test-model"{newline}{previous}{last_newline}'
                    )
                    updated = codex_instruct.replace_top_level_model_instructions(original, replacement)
                    self.assertEqual(updated, original.replace(previous, replacement))
                    self.assertEqual(
                        codex_instruct.replace_top_level_model_instructions(updated, previous), original
                    )

    def test_multiline_managed_values_are_rejected_before_deploy_or_reset_writes(self) -> None:
        for value in ['"""\n./custom.md"""', "'''\n./custom.md'''", '"""\\\n./custom.md"""']:
            for action in ["deploy", "reset"]:
                with self.subTest(value=value, action=action):
                    original = f'model = "test-model"\nmodel_instructions_file = {value}\n'
                    self.assertEqual(tomllib.loads(original)["model_instructions_file"], "./custom.md")
                    temporary_directory, config_path = self.make_config(original)
                    with temporary_directory:
                        root = config_path.parent
                        source = root / "source.md"
                        source.write_text("new synthetic prompt\n", encoding="utf-8")
                        owned = root / "custom.md"
                        owned.write_text("old synthetic prompt\n", encoding="utf-8")
                        codex_instruct.save_state(config_path, {
                            "version": codex_instruct.STATE_VERSION,
                            "previous_model_instructions_line": None,
                            "managed_prompts": {owned.name: {
                                "sha256": codex_instruct.file_sha256(owned), "existed_before": False,
                            }},
                        })
                        before = {path.name: path.read_bytes() for path in root.iterdir()}
                        args = SimpleNamespace(codex_dir=str(root), dry_run=False)
                        with patch("builtins.input", return_value="y"), patch("builtins.print"):
                            result = (
                                codex_instruct.deploy_prompt(args, source, "new.md")
                                if action == "deploy" else codex_instruct.reset_managed_install(args)
                            )
                        self.assertEqual(result, 2)
                        self.assertEqual(
                            {path.name: path.read_bytes() for path in root.iterdir()}, before
                        )

    def test_unsupported_managed_keys_are_rejected_before_deploy_or_reset_writes(self) -> None:
        for key in [
            '"model_instructions_file"', "'model_instructions_file'",
            r'"\u006dodel_instructions_file"', r'"\U0000006dodel_instructions_file"',
            r'"model_instructions_\u0066ile"', '"model_instructions_file".nested',
            'model_instructions_file.nested',
        ]:
            for action in ["deploy", "reset"]:
                with self.subTest(key=key, action=action):
                    original = f'model = "test-model"\n{key} = "./custom.md"\n'
                    self.assertIn("model_instructions_file", tomllib.loads(original))
                    temporary_directory, config_path = self.make_config(original)
                    with temporary_directory:
                        root = config_path.parent
                        source = root / "source.md"
                        source.write_text("new synthetic prompt\n", encoding="utf-8")
                        owned = root / "custom.md"
                        owned.write_text("old synthetic prompt\n", encoding="utf-8")
                        codex_instruct.save_state(config_path, {
                            "version": codex_instruct.STATE_VERSION,
                            "previous_model_instructions_line": None,
                            "managed_prompts": {owned.name: {
                                "sha256": codex_instruct.file_sha256(owned), "existed_before": False,
                            }},
                        })
                        before = {path.name: path.read_bytes() for path in root.iterdir()}
                        args = SimpleNamespace(codex_dir=str(root), dry_run=False)
                        with patch("builtins.input", return_value="y"), patch("builtins.print"):
                            result = (
                                codex_instruct.deploy_prompt(args, source, "new.md")
                                if action == "deploy" else codex_instruct.reset_managed_install(args)
                            )
                        self.assertEqual(result, 2)
                        self.assertEqual(
                            {path.name: path.read_bytes() for path in root.iterdir()}, before
                        )

    def test_unrelated_quoted_and_nested_keys_are_preserved(self) -> None:
        for key in [
            '"unrelated key"', '"model_instructions_file.example"',
            r'"\\u006dodel_instructions_file"', '"profile".model_instructions_file',
            '"配置".option', '"Model_instructions_file"',
        ]:
            with self.subTest(key=key):
                previous = 'model_instructions_file = "./personal.md" # keep'
                original = f'{key} = "kept"\nmodel = "test-model"\n{previous}\n'
                updated = codex_instruct.replace_top_level_model_instructions(
                    original, 'model_instructions_file = "./custom.md"'
                )
                expected = tomllib.loads(original)
                expected["model_instructions_file"] = "./custom.md"
                self.assertEqual(tomllib.loads(updated), expected)
                self.assertEqual(codex_instruct.top_level_model_instructions_line(original), previous)
                self.assertEqual(
                    codex_instruct.replace_top_level_model_instructions(updated, previous), original
                )

    # Legacy backups contribute only the prior instruction entry, never stale provider data.
    def test_legacy_baseline_migrates_only_previous_instruction(self) -> None:
        temporary_directory, config_path = self.make_config(
            'model_provider = "openai"\n'
            'model_instructions_file = "./gpt-5.6-sol-unrestricted-v42.md"\n'
        )
        self.addCleanup(temporary_directory.cleanup)
        baseline = codex_instruct.baseline_backup_path(config_path)
        baseline.write_text(
            'model_provider = "custom"\n'
            'model_instructions_file = "./personal.md"\n\n'
            '[model_providers.custom]\nbase_url = "https://example.invalid/v1"\n',
            encoding="utf-8",
        )

        changed, status = codex_instruct.restore_managed_model_instructions(config_path)

        self.assertTrue(changed)
        self.assertEqual(status, "restored")
        restored = tomllib.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(restored["model_provider"], "openai")
        self.assertEqual(restored["model_instructions_file"], "./personal.md")
        self.assertNotIn("model_providers", restored)

    # One end-to-end reset covers custom-name ownership, owned-artifact cleanup,
    # and preservation of provider/feature settings changed after deployment.
    def test_full_reset_removes_state_and_prompt_without_reverting_provider(self) -> None:
        temporary_directory, config_path = self.make_config(
            'model_provider = "custom"\nmodel = "gpt-5.5"\n'
        )
        self.addCleanup(temporary_directory.cleanup)
        codex_home = config_path.parent
        source = codex_home / "source.md"
        source.write_text("test instructions\n", encoding="utf-8")
        args = SimpleNamespace(codex_dir=str(codex_home), dry_run=False)

        result = codex_instruct.deploy_prompt(args, source, "custom-prompt.md")
        self.assertEqual(result, 0)

        # CCSwitch changes provider state while retaining the common instruction entry.
        config_path.write_text(
            'model_provider = "openai"\n'
            'model = "gpt-5.5"\n'
            'model_instructions_file = "./custom-prompt.md"\n\n'
            '[features]\n'
            'web_search = true\n',
            encoding="utf-8",
        )
        with patch("builtins.input", return_value="y"):
            result = codex_instruct.reset_managed_install(args)

        self.assertEqual(result, 0)
        restored = tomllib.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(restored["model_provider"], "openai")
        self.assertEqual(restored["model"], "gpt-5.5")
        self.assertTrue(restored["features"]["web_search"])
        self.assertNotIn("model_instructions_file", restored)
        self.assertFalse((codex_home / "custom-prompt.md").exists())
        self.assertFalse(codex_instruct.state_file_path(config_path).exists())

    # Editing a CRLF config must not introduce mixed line endings.
    def test_crlf_line_endings_are_preserved(self) -> None:
        text = 'model = "gpt-5.5"\r\n[features]\r\nweb_search = true\r\n'
        updated = codex_instruct.replace_top_level_model_instructions(
            text,
            'model_instructions_file = "./prompt.md"',
        )

        self.assertNotIn("\n", updated.replace("\r\n", ""))

    def test_disk_config_newlines_survive_deploy_and_reset(self) -> None:
        for newline in ["\n", "\r\n"]:
            for trailing in [False, True]:
                for previous in [None, 'model_instructions_file = "./personal.md" # keep']:
                    for linked in [False, True]:
                        with self.subTest(newline=repr(newline), trailing=trailing, previous=previous, linked=linked):
                            lines = ['# keep this comment', 'model = "test-model"']
                            if previous:
                                lines.append(previous)
                            lines += ['', '[features]', 'web_search = true']
                            original = newline.join(lines) + (newline if trailing else "")
                            temporary_directory, config_path = self.make_config("")
                            with temporary_directory:
                                root = config_path.parent
                                config_path.write_bytes(original.encode("utf-8"))
                                config_path.chmod(0o640)
                                if linked:
                                    backing = root / "backing.toml"
                                    config_path.rename(backing)
                                    config_path.symlink_to(backing)
                                source = root / "source.md"
                                source.write_text("synthetic prompt\n", encoding="utf-8")
                                args = SimpleNamespace(codex_dir=str(root), dry_run=False)
                                with patch("builtins.print"):
                                    self.assertEqual(codex_instruct.deploy_prompt(args, source, "custom.md"), 0)
                                expected = codex_instruct.replace_top_level_model_instructions(
                                    original, 'model_instructions_file = "./custom.md"'
                                )
                                self.assertEqual(config_path.read_bytes(), expected.encode("utf-8"))
                                self.assertEqual(
                                    tomllib.loads(config_path.read_bytes().decode("utf-8")),
                                    dict(tomllib.loads(original), model_instructions_file="./custom.md"),
                                )
                                snapshots = list(root.glob("config.toml.bak_*"))
                                self.assertEqual(len(snapshots), 1)
                                self.assertEqual(snapshots[0].read_bytes(), original.encode("utf-8"))
                                with patch("builtins.print"), patch("builtins.input", return_value="y"):
                                    self.assertEqual(codex_instruct.reset_managed_install(args), 0)
                                self.assertEqual(config_path.read_bytes(), original.encode("utf-8"))
                                self.assertEqual(config_path.stat().st_mode & 0o777, 0o640)
                                self.assertEqual(config_path.is_symlink(), linked)

    def test_snapshot_restore_preserves_snapshot_newline_bytes(self) -> None:
        for newline in ["\n", "\r\n"]:
            for trailing in [False, True]:
                with self.subTest(newline=repr(newline), trailing=trailing):
                    temporary_directory, config_path = self.make_config("")
                    with temporary_directory:
                        root = config_path.parent
                        original = b'model = "current-model"\r\n'
                        config_path.write_bytes(original)
                        config_path.chmod(0o640)
                        snapshot = root / "config.toml.bak_fixture"
                        saved = newline.join(['# snapshot comment', 'model = "saved-model"'])
                        saved += newline if trailing else ""
                        snapshot.write_bytes(saved.encode("utf-8"))
                        args = SimpleNamespace(codex_dir=str(root), dry_run=False)
                        with patch("builtins.print"), patch("builtins.input", return_value="y"):
                            self.assertEqual(codex_instruct.restore_config_snapshot(args, snapshot), 0)
                        self.assertEqual(config_path.read_bytes(), snapshot.read_bytes())
                        backups = [path for path in root.glob("config.toml.bak_*") if path != snapshot]
                        self.assertEqual(len(backups), 1)
                        self.assertEqual(backups[0].read_bytes(), original)
                        self.assertEqual(config_path.stat().st_mode & 0o777, 0o640)

    def test_newline_preservation_is_opt_in_for_config_writes(self) -> None:
        temporary_directory, config_path = self.make_config("")
        self.addCleanup(temporary_directory.cleanup)
        config_path.write_bytes(b'model = "test-model"\r\n')
        root = config_path.parent
        source = root / "source.md"
        source.write_text("synthetic prompt\n", encoding="utf-8")
        args = SimpleNamespace(codex_dir=str(root), dry_run=False)
        with patch.object(
            codex_instruct, "atomic_write_text", wraps=codex_instruct.atomic_write_text
        ) as write, patch("builtins.print"):
            self.assertEqual(codex_instruct.deploy_prompt(args, source, "custom.md"), 0)
        calls = {call.args[0]: call.kwargs for call in write.call_args_list}
        self.assertTrue(calls[config_path]["preserve_newlines"])
        self.assertFalse(calls[root / "custom.md"].get("preserve_newlines", False))
        self.assertFalse(
            calls[codex_instruct.state_file_path(config_path)].get("preserve_newlines", False)
        )
        with patch.object(os, "fdopen", wraps=os.fdopen) as fdopen:
            codex_instruct.atomic_write_text(root / "exact.txt", "one\r\ntwo\n", preserve_newlines=True)
            self.assertEqual(fdopen.call_args.kwargs["newline"], "")
            codex_instruct.atomic_write_text(root / "default.txt", "ordinary text\n")
            self.assertIsNone(fdopen.call_args.kwargs["newline"])
        self.assertEqual((root / "exact.txt").read_bytes(), b"one\r\ntwo\n")

    # An invalid state file cannot nominate config.toml or another non-Markdown file for deletion.
    def test_tampered_state_cannot_nominate_config_for_deletion(self) -> None:
        temporary_directory, config_path = self.make_config(
            'model_instructions_file = "./gpt-5.6-sol-unrestricted-v41.md"\n'
        )
        self.addCleanup(temporary_directory.cleanup)
        state_path = codex_instruct.state_file_path(config_path)
        state_path.write_text(
            json.dumps(
                {
                    "version": codex_instruct.STATE_VERSION,
                    "previous_model_instructions_line": None,
                    "managed_prompts": {
                        "config.toml": {
                            "sha256": "0" * 64,
                            "existed_before": False,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        args = SimpleNamespace(codex_dir=str(config_path.parent), dry_run=False)

        with patch("builtins.input", return_value="y"):
            result = codex_instruct.reset_managed_install(args)

        self.assertEqual(result, 0)
        self.assertTrue(config_path.exists())

    # A user-replaced prompt fails the recorded SHA256 check and must be preserved.
    def test_modified_custom_prompt_is_preserved_on_reset(self) -> None:
        temporary_directory, config_path = self.make_config('model = "gpt-5.5"\n')
        self.addCleanup(temporary_directory.cleanup)
        codex_home = config_path.parent
        source = codex_home / "source.md"
        source.write_text("deployed content\n", encoding="utf-8")
        args = SimpleNamespace(codex_dir=str(codex_home), dry_run=False)
        self.assertEqual(
            codex_instruct.deploy_prompt(args, source, "custom-prompt.md"),
            0,
        )
        custom_prompt = codex_home / "custom-prompt.md"
        custom_prompt.write_text("user replacement\n", encoding="utf-8")
        config_path.write_text(
            'model_instructions_file = "./personal.md"\n',
            encoding="utf-8",
        )

        with patch("builtins.input", return_value="y"):
            result = codex_instruct.reset_managed_install(args)

        self.assertEqual(result, 0)
        self.assertEqual(custom_prompt.read_text(encoding="utf-8"), "user replacement\n")
        self.assertIn("./personal.md", config_path.read_text(encoding="utf-8"))

    # A replacement symlink is user-owned even when its target matches the old prompt.
    def test_reset_preserves_replacement_prompt_symlinks_without_reading_targets(self) -> None:
        for target_kind in ["matching", "modified", "missing"]:
            with self.subTest(target_kind=target_kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                home = root / "codex-home"
                home.mkdir()
                config_path = home / "config.toml"
                original = 'model = "test-model"\nmodel_provider = "fixture"\n'
                config_path.write_text(original, encoding="utf-8")
                source = root / "source.md"
                source.write_text("synthetic prompt\n", encoding="utf-8")
                args = SimpleNamespace(codex_dir=str(home), dry_run=False)
                self.assertEqual(codex_instruct.deploy_prompt(args, source, "custom.md"), 0)
                self.assertEqual(codex_instruct.deploy_prompt(args, source, "owned.md"), 0)

                target = root / "user-owned.md"
                content = b"synthetic prompt\n" if target_kind == "matching" else b"user changed\n"
                if target_kind != "missing":
                    target.write_bytes(content)
                prompt = home / "custom.md"
                prompt.unlink()
                prompt.symlink_to(target)
                with patch("builtins.input", return_value="y"), patch("builtins.print") as output, patch.object(
                    codex_instruct, "file_sha256", wraps=codex_instruct.file_sha256
                ) as hash_file:
                    result = codex_instruct.reset_managed_install(args)

                self.assertEqual(result, 0)
                self.assertTrue(prompt.is_symlink())
                self.assertEqual(Path(os.readlink(prompt)), target)
                hash_file.assert_called_once_with(home / "owned.md")
                self.assertFalse((home / "owned.md").exists())
                if target_kind == "missing":
                    self.assertFalse(target.exists())
                else:
                    self.assertEqual(target.read_bytes(), content)
                self.assertEqual(config_path.read_bytes(), original.encode("utf-8"))
                self.assertFalse(codex_instruct.state_file_path(config_path).exists())
                messages = "\n".join(" ".join(map(str, call.args)) for call in output.call_args_list)
                self.assertIn("Removed 1, preserved 1", messages)

    # Atomic config writes follow a trusted config symlink instead of replacing the link itself.
    def test_atomic_config_update_preserves_symlink(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        codex_home = Path(temporary_directory.name)
        target = codex_home / "shared-config.toml"
        config_path = codex_home / "config.toml"
        target.write_text('model = "gpt-5.5"\n', encoding="utf-8")
        config_path.symlink_to(target.name)

        codex_instruct.set_model_instructions(
            config_path,
            "gpt-5.6-sol-unrestricted-v45.md",
        )

        self.assertTrue(config_path.is_symlink())
        self.assertIn("model_instructions_file", target.read_text(encoding="utf-8"))

    # A matching basename outside CODEX_HOME is not treated as a script-owned prompt.
    def test_external_path_with_managed_basename_is_not_owned(self) -> None:
        line = 'model_instructions_file = "/tmp/gpt-5.6-sol-unrestricted-v42.md"'
        self.assertFalse(
            codex_instruct.line_references_managed_prompt(
                line,
                codex_instruct.MANAGED_PROMPT_FILENAMES,
            )
        )

    # Deployment refuses to overwrite a destination that is not already tracked as owned.
    def test_preexisting_unowned_prompt_is_not_overwritten(self) -> None:
        temporary_directory, config_path = self.make_config('model = "gpt-5.5"\n')
        self.addCleanup(temporary_directory.cleanup)
        codex_home = config_path.parent
        source = codex_home / "source.md"
        source.write_text("new content\n", encoding="utf-8")
        destination = codex_home / "custom-prompt.md"
        destination.write_text("personal content\n", encoding="utf-8")
        args = SimpleNamespace(codex_dir=str(codex_home), dry_run=False)

        result = codex_instruct.deploy_prompt(args, source, destination.name)

        self.assertEqual(result, 2)
        self.assertEqual(destination.read_text(encoding="utf-8"), "personal content\n")
        self.assertFalse(codex_instruct.state_file_path(config_path).exists())

    # Full-file recovery remains available only through the explicit snapshot command.
    def test_explicit_snapshot_restore_keeps_manual_recovery(self) -> None:
        temporary_directory, config_path = self.make_config(
            'model_provider = "openai"\n'
        )
        self.addCleanup(temporary_directory.cleanup)
        codex_home = config_path.parent
        snapshot = codex_home / "config.toml.bak_20260723_010203_000001"
        snapshot.write_text('model_provider = "custom"\n', encoding="utf-8")
        args = SimpleNamespace(codex_dir=str(codex_home), dry_run=False)

        with patch("builtins.input", return_value="y"):
            result = codex_instruct.restore_config_snapshot(args, snapshot)

        self.assertEqual(result, 0)
        self.assertIn('model_provider = "custom"', config_path.read_text(encoding="utf-8"))

    # Legacy deployments preserve prompt files that existed before ownership tracking.
    def test_legacy_preexisting_prompt_is_preserved(self) -> None:
        filename = "gpt-5.6-sol-unrestricted-v41.md"
        temporary_directory, config_path = self.make_config(
            f'model_instructions_file = "./{filename}"\n'
        )
        self.addCleanup(temporary_directory.cleanup)
        codex_home = config_path.parent
        codex_instruct.baseline_backup_path(config_path).write_text(
            'model = "gpt-5.5"\n',
            encoding="utf-8",
        )
        destination = codex_home / filename
        destination.write_text("legacy deployment\n", encoding="utf-8")
        source = codex_home / "source.md"
        source.write_text("updated deployment\n", encoding="utf-8")
        args = SimpleNamespace(codex_dir=str(codex_home), dry_run=False)

        self.assertEqual(codex_instruct.deploy_prompt(args, source, filename), 0)
        with patch("builtins.input", return_value="y"):
            self.assertEqual(codex_instruct.reset_managed_install(args), 0)

        self.assertTrue(destination.exists())
        self.assertEqual(destination.read_text(encoding="utf-8"), "updated deployment\n")


if __name__ == "__main__":
    unittest.main()
