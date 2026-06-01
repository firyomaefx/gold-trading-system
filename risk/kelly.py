import numpy as np
from typing import Tuple, Optional


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    avg_loss = abs(avg_loss)
    if avg_loss < 1e-15:
        return 0.0

    b = avg_win / avg_loss
    if b < 1e-15:
        return 0.0

    q = 1.0 - win_rate
    f_star = win_rate - q / b

    return max(0.0, min(f_star, 0.25))


def fractional_kelly(win_rate: float, avg_win: float, avg_loss: float, fraction: float = 0.5) -> float:
    f_star = kelly_fraction(win_rate, avg_win, avg_loss)
    return f_star * fraction


def position_size(
    account_equity: float,
    kelly_f: float,
    atr: float,
    risk_pct: float = 0.02,
    max_position_risk: float = 0.25,
) -> float:
    kelly_f = min(kelly_f, max_position_risk)

    risk_amount = account_equity * risk_pct
    trade_risk = min(kelly_f * account_equity, risk_amount)

    if atr < 1e-15:
        atr = 1.0

    position = trade_risk / atr

    return position


def calculate_trade_stats(trades: list) -> Tuple[float, float, float, float]:
    if not trades:
        return 0.5, 0.01, 0.01, 0.0

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]

    win_rate = len(wins) / len(trades) if trades else 0.5
    avg_win = np.mean(wins) if wins else 0.01
    avg_loss = np.mean(losses) if losses else 0.01

    kf = kelly_fraction(win_rate, avg_win, abs(avg_loss))

    return win_rate, avg_win, abs(avg_loss), kf


class BayesianKelly:
    """Bayesian-updated win rate + payoff ratio for Kelly sizing.

    Maintains a Beta prior on win rate and a Normal-Normal conjugate
    prior on the log-odds of the payoff ratio. With each new trade,
    the posterior is updated and the Kelly fraction is recomputed.

    This avoids the brittle "use last 10 trades" problem and produces
    a smoothed estimate that converges to the truth as data accumulates
    while shrinking toward the prior when data is sparse.
    """

    def __init__(
        self,
        prior_win_rate: float = 0.55,
        prior_strength: float = 20.0,
        prior_payoff: float = 1.2,
        prior_payoff_strength: float = 20.0,
        max_fraction: float = 0.25,
    ):
        self.prior_alpha = prior_win_rate * prior_strength
        self.prior_beta = (1.0 - prior_win_rate) * prior_strength
        self.alpha = self.prior_alpha
        self.beta = self.prior_beta

        self.prior_payoff = float(prior_payoff)
        self.prior_payoff_strength = float(prior_payoff_strength)
        self.payoff_sum_w = prior_payoff_strength * np.log(prior_payoff) if prior_payoff > 0 else 0.0
        self.payoff_n = prior_payoff_strength
        self.payoff_var_n = prior_payoff_strength

        self.max_fraction = max_fraction
        self.n_updates = 0
        self.last_win_rate = prior_win_rate
        self.last_payoff = prior_payoff

    def reset(self) -> None:
        self.alpha = self.prior_alpha
        self.beta = self.prior_beta
        self.payoff_sum_w = self.prior_payoff_strength * np.log(self.prior_payoff) if self.prior_payoff > 0 else 0.0
        self.payoff_n = self.prior_payoff_strength
        self.payoff_var_n = self.prior_payoff_strength
        self.n_updates = 0
        self.last_win_rate = self.prior_alpha / (self.prior_alpha + self.prior_beta)
        self.last_payoff = self.prior_payoff

    def update(self, pnl: float) -> None:
        if pnl > 0:
            self.alpha += 1.0
        else:
            self.beta += 1.0
        self.n_updates += 1
        self.last_win_rate = self.alpha / (self.alpha + self.beta)
        if pnl > 0:
            self.payoff_n += 1.0
        payoff = abs(pnl) + 1e-6
        self.payoff_sum_w += np.log(payoff)
        self.last_payoff = float(np.exp(self.payoff_sum_w / max(self.payoff_n, 1e-9)))

    def update_batch(self, pnls) -> None:
        for p in pnls:
            self.update(p)

    def expected_win_rate(self) -> float:
        return float(self.alpha / (self.alpha + self.beta))

    def win_rate_variance(self) -> float:
        a, b = self.alpha, self.beta
        return float((a * b) / (((a + b) ** 2) * (a + b + 1.0)))

    def payoff_lower_bound(self, confidence: float = 0.05) -> float:
        if self.payoff_n < 2:
            return 0.0
        mu = self.payoff_sum_w / self.payoff_n
        var = 1.0 / self.payoff_n
        std = float(np.sqrt(var))
        from scipy.stats import norm
        z = norm.ppf(confidence)
        return float(np.exp(mu + z * std))

    def kelly_fraction(self) -> float:
        w = self.expected_win_rate()
        b = self.last_payoff
        if b <= 0:
            return 0.0
        f_star = w - (1.0 - w) / b
        return float(max(0.0, min(f_star, self.max_fraction)))

    def conservative_kelly(self, shrink: float = 0.7) -> float:
        """Shrink the Kelly fraction to defend against estimation error."""
        return self.kelly_fraction() * shrink

    def diagnostic(self) -> dict:
        return {
            "n": self.n_updates,
            "win_rate": self.expected_win_rate(),
            "win_rate_std": float(np.sqrt(self.win_rate_variance())),
            "payoff": self.last_payoff,
            "payoff_lb_5pct": self.payoff_lower_bound(0.05),
            "kelly_full": self.kelly_fraction(),
            "kelly_shrunk": self.conservative_kelly(),
        }
