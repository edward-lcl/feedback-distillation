class LossConfig:
    LOSS_NAMES = ["lm_loss", "hidden_loss", "scoring_loss", "logit_loss"]
    SCALES = {"lm_loss": 1.0, "hidden_loss": 1.0, "scoring_loss": 1.0, "logit_loss": 0.03}

    def __init__(self, enabled_flags: list[bool]):
        if len(enabled_flags) != len(self.LOSS_NAMES):
            raise ValueError(f"Expected {len(self.LOSS_NAMES)} flags, got {len(enabled_flags)}")
        self.enabled = dict(zip(self.LOSS_NAMES, enabled_flags))
        self.num_enabled = sum(enabled_flags)

    def get_active_losses(self) -> tuple[list[str], dict[str, bool]]:
        active = [n for n, f in self.enabled.items() if f]
        return active, self.enabled

    def toggle_loss(self, name: str, state: bool):
        if name not in self.enabled:
            raise KeyError(f"{name} not valid. Choose from {self.LOSS_NAMES}")
        self.enabled[name] = state
        self.num_enabled = sum(self.enabled.values())
