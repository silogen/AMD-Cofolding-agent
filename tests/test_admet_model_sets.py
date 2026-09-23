"""Dual ADMET model-stack contracts."""

from unittest.mock import MagicMock, patch

from reinvent4_mcp.config import AdmetConfig
from reinvent4_mcp.tools.external import predict_admet
from reinvent4_mcp.ttp.mapper import ttp_to_scorer_config
from reinvent4_mcp.ttp.schema import TTP


def test_admet_config_defaults_to_drug() -> None:
    assert AdmetConfig().model_set == "drug"


def test_ttp_mapper_injects_default_model_set() -> None:
    ttp = TTP.model_validate(
        {
            "id": "admet-model-set",
            "name": "ADMET model-set contract",
            "scoring": {
                "objectives": [
                    {
                        "name": "LogP",
                        "type": "external",
                        "service": "admet",
                        "service_url": "http://admet-predictor:8084/score",
                        "service_params": {"endpoint": "logp"},
                    }
                ]
            },
        }
    )
    config = ttp_to_scorer_config(ttp)
    endpoint = config["component"][0]["ExternalRESTScorer"]["endpoint"][0]
    assert endpoint["params"]["model_set"] == "drug"


@patch("reinvent4_mcp.tools.external.requests.post")
def test_predict_admet_forwards_agro_model_set(mock_post) -> None:
    response = MagicMock()
    response.json.return_value = [
        {
            "input_smiles": "CCO",
            "standardized_smiles": "CCO",
            "ok": True,
            "model_set": "agro",
            "predictions": {"bcf": {"mean": 1.2}},
        }
    ]
    response.raise_for_status.return_value = None
    mock_post.return_value = response

    with patch(
        "reinvent4_mcp.tools.external._get_service_url",
        return_value="http://admet-predictor:8084/score",
    ):
        result = predict_admet(["CCO"], ["bcf"], model_set="agro")

    assert result["model_set"] == "agro"
    assert mock_post.call_args.kwargs["json"]["model_set"] == "agro"
