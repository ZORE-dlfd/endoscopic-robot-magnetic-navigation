from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    decision_s: float = .4
    visibility_s: float = .1
    integration_s: float = .01
    horizon_s: float = 1.2
    coarse_deg: float = 10.
    fine_deg: float = 2.
    peak_count: int = 3
    candidate_count: int = 6
    gain_tolerance: float = .05
    atlas_size: int = 192
    include_delayed: bool = True

    def validate(self):
        if min(self.decision_s, self.visibility_s, self.integration_s, self.horizon_s, self.coarse_deg, self.fine_deg) <= 0:
            raise ValueError('Steps and horizon must be positive')
        if self.horizon_s < self.decision_s or self.visibility_s > self.decision_s:
            raise ValueError('Horizon must include one decision; visibility step cannot exceed decision')
        for upper, lower in [(self.decision_s, self.visibility_s), (self.visibility_s, self.integration_s)]:
            if abs(upper / lower - round(upper / lower)) > 1e-8:
                raise ValueError('Decision/visibility/integration steps must divide exactly')
        if not 0 <= self.gain_tolerance < 1 or self.peak_count < 1 or self.candidate_count < 1 or self.atlas_size < 16:
            raise ValueError('Invalid search parameters')
        return self

    def dictionary(self):
        return asdict(self)
