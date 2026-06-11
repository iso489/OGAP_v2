import json
from ogap.cli import main as cli_main


def test_nas_search_cli_runs(capsys):
    rc = cli_main(["nas-search", "--n", "3", "--patch", "16,16,16",
                   "--num-classes", "4", "--in-channels", "8", "--seed", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    report = json.loads(out)
    assert report["strategy"] == "random"
    assert len(report["pareto_front"]) >= 1
    a0 = report["pareto_front"][0]
    assert "arch" in a0 and "cost" in a0 and "accuracy_proxy" in a0
