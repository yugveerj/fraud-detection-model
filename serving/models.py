"""Pydantic request/response models for the scoring API (SPEC Section 5).

The input mirrors an IEEE-CIS transaction. Only ``TransactionDT`` and
``TransactionAmt`` are required; every other schema field is optional and any extra
fields (``C*``, ``V*``, ``id_*`` …) are accepted verbatim (``extra="allow"``) and
filled with NaN if absent. A single scored transaction has no entity history, so its
causal aggregates are the first-sighting values (count 0) — correct for a novel
transaction.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ScoreRequest(BaseModel):
    """A single transaction to score. Unlisted IEEE-CIS columns pass through."""

    model_config = ConfigDict(extra="allow")

    TransactionDT: int = Field(..., ge=0, description="relative timestamp (seconds)")
    TransactionAmt: float = Field(..., gt=0, description="transaction amount")
    ProductCD: str | None = None
    card1: int | None = None
    card4: str | None = Field(default=None, description="card network, e.g. visa")
    card6: str | None = Field(default=None, description="card type, e.g. debit/credit")
    P_emaildomain: str | None = None
    DeviceType: str | None = None


class Factor(BaseModel):
    feature: str
    value: float | None  # None when the underlying feature is missing (NaN)
    shap: float
    direction: str  # "increases" | "decreases"


class ScoreResponse(BaseModel):
    fraud_probability: float = Field(..., ge=0, le=1, description="CALIBRATED probability")
    decision: str = Field(..., description="'review' or 'approve'")
    threshold: float
    top_factors: list[Factor]
    model_version: str | None = None
    run_id: str | None = None
    calibration: str = "isotonic"
    provenance: str = "synthetic"
    disclaimer: str = (
        "Demonstration system on a public research dataset — not a production fraud decision."
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    model_version: str | None = None
    run_id: str | None = None
    calibration: str = "isotonic"
    provenance: str = "synthetic"
    operating_threshold: float | None = None
    review_cost: float | None = None
