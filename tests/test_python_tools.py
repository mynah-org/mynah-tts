from pathlib import Path


def test_converter_and_inspector_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "tools" / "convert_magpie.py").is_file()
    assert (root / "tools" / "inspect_nemo.py").is_file()
    assert (root / "tools" / "export_magpie_tokenizer.py").is_file()


if __name__ == "__main__":
    test_converter_and_inspector_exist()
    print("Python tooling smoke test: PASS")
