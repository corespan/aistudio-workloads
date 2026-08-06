"""
costEngine.py — LLM inference cost estimation.

Produces three cost breakdowns per benchmark run:
  on_prem      — GPU capex amortised + electricity (for users running their own hardware)
  neocloud     — hourly GPU rental from neocloud providers (CoreWeave, Lambda, Vast.ai, etc.)
  hyperscaler  — on-demand rates from major cloud providers (AWS, Azure, GCP)

Rates are loaded from hardware_pricing.json in the same directory.
Update that file to reflect current market prices or your actual hardware cost.
"""

import json
import logging
import os

log = logging.getLogger(__name__)


class CostEngine:
    def __init__(self, config_path: str | None = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "hardware_pricing.json")
        self.config_path = config_path
        self._load_config()

    def _load_config(self) -> None:
        try:
            with open(self.config_path) as f:
                self.config = json.load(f)
        except Exception as exc:
            log.error("CostEngine: could not load %s — %s", self.config_path, exc)
            self.config = None

    def _get_gpu_key(self, gpu_name: str) -> str:
        name = gpu_name.upper()
        if "T4"   in name: return "Tesla T4"
        if "P40"  in name: return "Tesla P40"
        if "A100" in name: return "A100"
        if "H100" in name: return "H100"
        if "5090" in name: return "RTX 5090"
        return "DEFAULT"

    def _amortize_hourly(self, capex: float, lifespan_years: int) -> float:
        return capex / (lifespan_years * 365 * 24)

    def _calculate_costs(
        self,
        gpu_key: str,
        num_gpus: int,
        duration_hours: float,
        avg_power_watts: float,
    ) -> tuple[float, float, float]:
        """Return (on_prem_total, neocloud_total, hyperscaler_total) in USD."""
        elec_rate = self.config["meta"]["electricity_kwh_usd"]
        energy_usd = (avg_power_watts * num_gpus / 1000.0) * duration_hours * elec_rate

        # On-prem: GPU capex amortised over lifespan + electricity
        gpu_spec   = self.config["gpus"].get(gpu_key, self.config["gpus"]["DEFAULT"])
        gpu_hourly = self._amortize_hourly(gpu_spec["capex_usd"], gpu_spec["lifespan_years"])
        on_prem_total = (gpu_hourly * num_gpus * duration_hours) + energy_usd

        market = self.config["market_rates"]["gpus"].get(
            gpu_key, self.config["market_rates"]["gpus"]["DEFAULT"]
        )
        neocloud_total    = (market["neocloud_hourly_usd"]    * num_gpus * duration_hours) + energy_usd
        hyperscaler_total = (market["hyperscaler_hourly_usd"] * num_gpus * duration_hours) + energy_usd

        return on_prem_total, neocloud_total, hyperscaler_total

    def calculate_llm(
        self,
        gpu_name: str,
        num_gpus: int,
        duration_seconds: float,
        avg_power_watts: float,
        total_input_tokens: int,
        total_output_tokens: int,
    ) -> dict:
        """
        Calculate cost per million tokens for an LLM inference benchmark run.

        Uses a 3:1 output-to-input cost weighting — output tokens require
        substantially more compute to generate than input tokens require to process.

        Returns a dict with three breakdowns: on_prem, neocloud, hyperscaler.
        Each contains blended, input, and output cost per million tokens (USD).
        """
        if not self.config:
            return {"error": "hardware_pricing.json could not be loaded"}

        gpu_key        = self._get_gpu_key(gpu_name)
        duration_hours = duration_seconds / 3600.0

        on_prem_total, neocloud_total, hyperscaler_total = self._calculate_costs(
            gpu_key, num_gpus, duration_hours, avg_power_watts,
        )

        def _per_million(total_cost: float) -> tuple[float, float, float]:
            total_tokens = total_input_tokens + total_output_tokens
            if total_tokens <= 0:
                return 0.0, 0.0, 0.0
            blended = (total_cost / total_tokens) * 1_000_000
            # Output tokens cost 3x more than input tokens to generate
            denom       = total_input_tokens + (3 * total_output_tokens)
            base_input  = total_cost / denom if denom > 0 else 0.0
            base_output = base_input * 3
            return blended, base_input * 1_000_000, base_output * 1_000_000

        op_b, op_i, op_o = _per_million(on_prem_total)
        nc_b, nc_i, nc_o = _per_million(neocloud_total)
        hy_b, hy_i, hy_o = _per_million(hyperscaler_total)

        return {
            "gpu_detected":            gpu_name,
            "workload_duration_hours": round(duration_hours, 6),
            "on_prem": {
                "blended_cost_per_1m_tokens_usd": round(op_b, 4),
                "input_cost_per_1m_tokens_usd":   round(op_i, 4),
                "output_cost_per_1m_tokens_usd":  round(op_o, 4),
            },
            "neocloud": {
                "blended_cost_per_1m_tokens_usd": round(nc_b, 4),
                "input_cost_per_1m_tokens_usd":   round(nc_i, 4),
                "output_cost_per_1m_tokens_usd":  round(nc_o, 4),
            },
            "hyperscaler": {
                "blended_cost_per_1m_tokens_usd": round(hy_b, 4),
                "input_cost_per_1m_tokens_usd":   round(hy_i, 4),
                "output_cost_per_1m_tokens_usd":  round(hy_o, 4),
            },
        }
