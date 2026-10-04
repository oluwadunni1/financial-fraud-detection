"""The demo CLI prints only what the system already produced.

`results` reads committed reports, so it runs anywhere -- including CI -- and
breaks loudly if a report it cites is renamed or reshaped.
"""

from __future__ import annotations

import pathlib

import pytest


def test_results_reads_every_report_it_cites(capsys):
    from fraud.demo import main

    assert main(["results"]) == 0
    out = capsys.readouterr().out
    for expected in ("0.4665", "0.0418", "0.2501", "served == causal replay",
                     "Monitoring without labels"):
        assert expected in out


@pytest.mark.skipif(not pathlib.Path("data/processed").exists(), reason="processed data absent")
def test_payload_is_a_valid_predict_request(capsys):
    import json

    from fraud.api.main import Transaction
    from fraud.demo import main

    assert main(["payload", "18267417"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert Transaction.model_validate(body).txn_id == 18267417
