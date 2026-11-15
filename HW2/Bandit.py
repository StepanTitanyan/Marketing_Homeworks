"""A/B testing with multi-armed bandits: Epsilon Greedy vs Thompson Sampling.

Scenario: four advertisement options (bandits) with true mean rewards
``Bandit_Reward = [1, 2, 3, 4]``. Every algorithm gets ``NumberOfTrials = 20000``
trials. The reward of a bandit is Gaussian with a *known* precision ``tau``
(precision = 1 / variance), so pulling bandit ``k`` gives

    reward ~ Normal(mean = Bandit_Reward[k], variance = 1 / tau)

Run this file at first, in order to see what it is printing::

    python Bandit.py              # experiment + csv + plots + comparison
    python Bandit.py --runs 100   # more repetitions in comparison()
    python Bandit.py --tune       # extra: how epsilon0 and the prior were chosen
    python Bandit.py --debug      # also show the debug messages

Nothing is printed with print(), everything goes through the loguru logger
(debug / info / warning / error / critical).

Outputs
    rewards.csv   columns {Bandit, Reward, Algorithm}, one row per trial
    img/*.png     plot1 (learning process), plot2 (cumulative reward / regret)
                  and the multi-run comparison

Regret
    The regret of one trial is ``max(Bandit_Reward) - Bandit_Reward[chosen bandit]``
    (expected regret), so the cumulative regret is 0 only if the best bandit
    was chosen in every trial.
"""
import argparse
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # we only save the figures to png files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

############################### SETTINGS
Bandit_Reward = [1, 2, 3, 4]  # true mean reward of the 4 ads
NumberOfTrials = 20000

PRECISION = 1.0  # known precision of the reward (tau = 1 / variance), std = 1
EPSILON_0 = 10.0  # epsilon at trial t is EPSILON_0 / t (decays by 1/t), capped at 1
PRIOR_MEAN = 0.0  # Thompson Sampling: prior mean of every bandit mean
PRIOR_PRECISION = 0.1  # Thompson Sampling: prior precision (vague prior, std ~ 3.2)
SEED = 42
N_RUNS = 50  # repetitions used in comparison() and tune()

HERE = Path(__file__).resolve().parent
IMG_DIR = HERE / "img"
CSV_PATH = HERE / "rewards.csv"
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


############################### BANDIT (base class)
class Bandit(ABC):
    """Abstract experiment with several bandits (ads) whose rewards are unknown.

    ``EpsilonGreedy`` and ``ThompsonSampling`` inherit from this class. The
    abstract methods are the ones from the homework template, the concrete
    methods at the end of the class (``_setup``, ``_record``, ``to_frame``...)
    are extra helpers shared by both algorithms.
    """
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        """Create the experiment.

        :param p: true mean reward of every bandit, e.g. ``[1, 2, 3, 4]``
        :type p: list
        """
        pass

    @abstractmethod
    def __repr__(self):
        """Return a short text with the algorithm and its settings.

        :returns: description of the object
        :rtype: str
        """
        pass

    @abstractmethod
    def pull(self):
        """Pull one bandit and get a (random) reward.

        :returns: the reward of this pull
        :rtype: float
        """
        pass

    @abstractmethod
    def update(self):
        """Update what the algorithm knows after one pull."""
        pass

    @abstractmethod
    def experiment(self):
        """Run all the trials of the experiment."""
        pass

    @abstractmethod
    def report(self):
        """Store the data and print the results of the experiment."""
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

    # ------------------------------------------------------------------
    # Everything below is added by me. It is shared by both algorithms so
    # that EpsilonGreedy and ThompsonSampling only contain what is different.
    # ------------------------------------------------------------------
    @abstractmethod
    def choose_arm(self, t):
        """Decide which bandit to pull in trial ``t`` (the policy).

        :param t: trial number, starts at 1
        :type t: int
        :returns: index of the chosen bandit (0 based)
        :rtype: int
        """
        pass

    def _setup(self, p, tau, n_trials, seed, name):
        """Set everything that both algorithms need.

        :param p: true mean reward of every bandit
        :type p: list
        :param tau: known precision of the reward noise (1 / variance)
        :type tau: float
        :param n_trials: number of trials of the experiment
        :type n_trials: int
        :param seed: seed of the random number generator
        :type seed: int
        :param name: name of the algorithm, used in the csv and in the plots
        :type name: str
        :raises ValueError: if one of the inputs makes no sense
        """
        means = np.asarray(p, dtype=float)
        if means.ndim != 1 or means.size < 2:
            logger.error(f"p must be a list with at least 2 true rewards, got {p!r}")
            raise ValueError("p must be a list with at least 2 true rewards")
        if tau <= 0:
            logger.error(f"precision must be positive, got {tau}")
            raise ValueError("precision must be positive")
        if n_trials < 1:
            logger.error(f"number of trials must be at least 1, got {n_trials}")
            raise ValueError("number of trials must be at least 1")

        self.name = name
        self.true_means = means
        self.n_bandits = means.size
        self.tau = float(tau)
        self.n_trials = int(n_trials)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.best_mean = means.max()
        self.best_bandit = int(means.argmax())

        # what was logged during the experiment (one entry per trial)
        self.chosen = np.zeros(self.n_trials, dtype=int)
        self.rewards = np.zeros(self.n_trials)
        self.regrets = np.zeros(self.n_trials)
        self.estimates = np.zeros((self.n_trials, self.n_bandits))
        self.pulls = np.zeros(self.n_bandits, dtype=int)
        self.finished = False

    def _draw_reward(self, arm):
        """Draw a reward from the true distribution of a bandit.

        :param arm: index of the bandit
        :type arm: int
        :returns: ``Normal(true mean, 1 / tau)`` random number
        :rtype: float
        """
        return float(self.rng.normal(self.true_means[arm], 1.0 / np.sqrt(self.tau)))

    def _record(self, t, arm, reward, estimates):
        """Save one trial in the logs.

        :param t: trial number, starts at 1
        :type t: int
        :param arm: chosen bandit
        :type arm: int
        :param reward: reward that was received
        :type reward: float
        :param estimates: current estimate of the mean of every bandit
        :type estimates: numpy.ndarray
        """
        self.chosen[t - 1] = arm
        self.rewards[t - 1] = reward
        self.regrets[t - 1] = self.best_mean - self.true_means[arm]
        self.estimates[t - 1] = estimates

    def _run_trials(self):
        """Loop over all trials: choose, pull, update, record."""
        for t in range(1, self.n_trials + 1):
            arm = self.choose_arm(t)
            reward = self.pull(arm)
            self.update(arm, reward)
            self._record(t, arm, reward, self._live_estimates)
            if t % 5000 == 0:
                logger.debug(f"{self.name}: trial {t}/{self.n_trials}, "
                             f"estimates = {np.round(self._live_estimates, 3)}")
        self.finished = True

    def _check_finished(self):
        """Stop with a clear message if the experiment was not run yet.

        :raises RuntimeError: if ``experiment()`` was not called
        """
        if not self.finished:
            logger.error(f"{self.name}: call experiment() before using the results")
            raise RuntimeError("experiment() was not run yet")

    @property
    def cumulative_rewards(self):
        """Cumulative reward after every trial.

        :returns: array with ``n_trials`` values
        :rtype: numpy.ndarray
        """
        self._check_finished()
        return np.cumsum(self.rewards)

    @property
    def cumulative_regrets(self):
        """Cumulative regret after every trial.

        :returns: array with ``n_trials`` values
        :rtype: numpy.ndarray
        """
        self._check_finished()
        return np.cumsum(self.regrets)

    def to_frame(self):
        """Put the rewards in a table with the columns {Bandit, Reward, Algorithm}.

        The bandit is numbered from 1 to 4 (bandit k has true mean reward ``p[k - 1]``).

        :returns: one row per trial
        :rtype: pandas.DataFrame
        """
        self._check_finished()
        return pd.DataFrame({"Bandit": self.chosen + 1,
                             "Reward": self.rewards,
                             "Algorithm": self.name})

    def _report(self, csv_path, append):
        """Store the rewards in the csv file and log the results.

        :param csv_path: file to write
        :type csv_path: str or pathlib.Path
        :param append: ``True`` adds the rows to an existing file (no new header)
        :type append: bool
        """
        self._check_finished()
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(csv_path, mode="a" if append else "w",
                               header=not append, index=False, float_format="%.6f")
        logger.info(f"{self.name}: {self.n_trials} rewards saved in {csv_path.name}")

        cum_reward = self.cumulative_rewards[-1]
        cum_regret = self.cumulative_regrets[-1]
        logger.info(f"{self.name}: cumulative reward = {cum_reward:.2f}")
        logger.info(f"{self.name}: cumulative regret = {cum_regret:.2f}")
        logger.info(f"{self.name}: average reward = {cum_reward / self.n_trials:.4f} "
                    f"(best possible = {self.best_mean:.1f})")
        logger.info(f"{self.name}: average regret = {cum_regret / self.n_trials:.4f}")
        share = 100 * self.pulls / self.n_trials
        text = ", ".join(f"bandit {k + 1}: {int(n)} ({s:.1f}%)"
                         for k, (n, s) in enumerate(zip(self.pulls, share)))
        logger.info(f"{self.name}: pulls per bandit -> {text}")

#--------------------------------------#


class Visualization():
    """Plots of the experiment. Figures are saved as png files in ``out_dir``."""

    def __init__(self, algorithms, out_dir=IMG_DIR):
        """Keep the algorithms that were already run.

        :param algorithms: algorithms whose ``experiment()`` was called
        :type algorithms: list
        :param out_dir: folder for the png files (Default value = IMG_DIR)
        :type out_dir: str or pathlib.Path
        """
        self.algorithms = list(algorithms)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def plot1(self):
        """Visualize the learning process of each algorithm (linear and log scale).

        For every algorithm one figure is saved. The coloured lines are the
        estimated mean reward of each bandit after every trial (dashed line =
        true mean), the black line is the average reward received so far.
        """
        for algo in self.algorithms:
            x = np.arange(1, algo.n_trials + 1)
            running_avg = algo.cumulative_rewards / x
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
            for ax, scale in zip(axes, ["linear", "log"]):
                for k in range(algo.n_bandits):
                    ax.plot(x, algo.estimates[:, k], color=COLORS[k], lw=1.3,
                            label=f"bandit {k + 1} estimate")
                    ax.axhline(algo.true_means[k], color=COLORS[k], ls="--", lw=0.8)
                ax.plot(x, running_avg, color="black", lw=1.6, label="average reward so far")
                ax.set_xscale(scale)
                ax.set_xlabel("trial" + (" (log scale)" if scale == "log" else ""))
                ax.set_ylabel("estimated mean reward")
                ax.set_title(f"{algo.name} - {scale} scale")
                ax.grid(alpha=0.3)
            axes[0].legend(loc="lower right", fontsize=8)
            fig.tight_layout()
            path = self.out_dir / f"plot1_{algo.name.lower().replace(' ', '_')}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            logger.info(f"plot1 saved: {path.name}")

    def plot2(self):
        """Compare the algorithms: cumulative rewards and cumulative regrets.

        The cumulative reward lines are almost on top of each other (the rewards are
        around 4 per trial for both), so a third panel shows the average reward per
        trial on a log scale, where the difference at the beginning is visible.
        """
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
        for algo, color in zip(self.algorithms, ["#1f77b4", "#d62728"]):
            x = np.arange(1, algo.n_trials + 1)
            axes[0].plot(x, algo.cumulative_rewards, color=color, label=algo.name)
            axes[1].plot(x, algo.cumulative_regrets, color=color, label=algo.name)
            axes[2].plot(x, algo.cumulative_rewards / x, color=color, label=algo.name)
        axes[0].set_title("Cumulative reward")
        axes[1].set_title("Cumulative regret")
        axes[2].set_title("Average reward per trial so far")
        axes[2].set_xscale("log")
        for ax in axes:
            ax.set_xlabel("trial" + (" (log scale)" if ax is axes[2] else ""))
            ax.grid(alpha=0.3)
            ax.legend()
        axes[0].set_ylabel("cumulative reward")
        axes[1].set_ylabel("cumulative regret")
        axes[2].set_ylabel("average reward")
        fig.tight_layout()
        path = self.out_dir / "plot2_cumulative_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"plot2 saved: {path.name}")

#--------------------------------------#


class EpsilonGreedy(Bandit):
    """Epsilon Greedy with epsilon that decays by 1/t.

    In trial ``t`` the algorithm explores (random bandit) with probability
    ``epsilon_t = min(1, epsilon0 / t)`` and otherwise exploits (bandit with the
    highest sample mean so far).
    """

    def __init__(self, p, epsilon0=EPSILON_0, tau=PRECISION,
                 n_trials=NumberOfTrials, seed=SEED):
        """Create the experiment.

        :param p: true mean reward of every bandit
        :type p: list
        :param epsilon0: epsilon at the first trials, ``epsilon_t = epsilon0 / t`` (Default value = EPSILON_0)
        :type epsilon0: float
        :param tau: known precision of the reward noise (Default value = PRECISION)
        :type tau: float
        :param n_trials: number of trials (Default value = NumberOfTrials)
        :type n_trials: int
        :param seed: seed of the random number generator (Default value = SEED)
        :type seed: int
        :raises ValueError: if epsilon0 is not positive or the other inputs are wrong
        """
        self._setup(p, tau, n_trials, seed, name="Epsilon Greedy")
        if epsilon0 <= 0:
            logger.error(f"epsilon0 must be positive, got {epsilon0}")
            raise ValueError("epsilon0 must be positive")
        self.epsilon0 = float(epsilon0)
        self.m = np.zeros(self.n_bandits)  # sample mean of every bandit
        self._live_estimates = self.m  # same array, so it is always up to date

    def __repr__(self):
        """Describe the algorithm and its settings.

        :returns: text like ``EpsilonGreedy(bandits=[...], epsilon_t=min(1, 10/t), ...)``
        :rtype: str
        """
        return (f"EpsilonGreedy(bandits={self.true_means.tolist()}, "
                f"epsilon_t=min(1, {self.epsilon0:g}/t), precision={self.tau:g}, "
                f"trials={self.n_trials})")

    def epsilon(self, t):
        """Exploration probability in trial ``t`` (decays by 1/t).

        :param t: trial number, starts at 1
        :type t: int
        :returns: ``min(1, epsilon0 / t)``
        :rtype: float
        """
        return min(1.0, self.epsilon0 / t)

    def choose_arm(self, t):
        """Choose the bandit of trial ``t``: explore with probability epsilon_t, else exploit.

        :param t: trial number, starts at 1
        :type t: int
        :returns: index of the chosen bandit (0 based)
        :rtype: int
        """
        # explore with probability epsilon_t, otherwise take the best sample mean
        if self.rng.random() < self.epsilon(t):
            return int(self.rng.integers(self.n_bandits))
        best = np.flatnonzero(self.m == self.m.max())  # random choice if there is a tie
        return int(self.rng.choice(best))

    def pull(self, arm):
        """Pull a bandit.

        :param arm: index of the bandit
        :type arm: int
        :returns: the reward
        :rtype: float
        """
        return self._draw_reward(arm)

    def update(self, arm, reward):
        """Update the sample mean of the pulled bandit (running mean).

        :param arm: index of the bandit that was pulled
        :type arm: int
        :param reward: reward that was received
        :type reward: float
        """
        self.pulls[arm] += 1
        self.m[arm] += (reward - self.m[arm]) / self.pulls[arm]

    def experiment(self):
        """Run the experiment: ``n_trials`` trials with decaying epsilon."""
        logger.info(f"start: {self!r}")
        self._run_trials()
        logger.info(f"{self.name}: done, final epsilon = {self.epsilon(self.n_trials):.5f}, "
                    f"final estimates = {np.round(self.m, 3).tolist()}")

    def report(self, csv_path=CSV_PATH, append=False):
        """Store the rewards in the csv file and log cumulative/average reward and regret.

        :param csv_path: csv file, columns {Bandit, Reward, Algorithm} (Default value = CSV_PATH)
        :type csv_path: str or pathlib.Path
        :param append: ``True`` adds the rows to an existing file (Default value = False)
        :type append: bool
        """
        self._report(csv_path, append)

#--------------------------------------#


class ThompsonSampling(Bandit):
    """Thompson Sampling for Gaussian rewards with a known precision.

    The unknown mean of every bandit has a Normal prior with mean ``mu0`` and
    precision ``lambda0``. Because the precision ``tau`` of the reward is known,
    the posterior stays Normal after every pull of bandit ``k`` with ``n`` pulls
    and reward sum ``S``::

        precision_k = lambda0 + n * tau
        mean_k      = (lambda0 * mu0 + tau * S) / precision_k

    In every trial one value is sampled from the posterior of each bandit and the
    bandit with the highest sample is pulled.
    """

    def __init__(self, p, tau=PRECISION, mu0=PRIOR_MEAN, lambda0=PRIOR_PRECISION,
                 n_trials=NumberOfTrials, seed=SEED):
        """Create the experiment.

        :param p: true mean reward of every bandit
        :type p: list
        :param tau: known precision of the reward noise (Default value = PRECISION)
        :type tau: float
        :param mu0: prior mean of the mean reward of every bandit (Default value = PRIOR_MEAN)
        :type mu0: float
        :param lambda0: prior precision, small = vague prior (Default value = PRIOR_PRECISION)
        :type lambda0: float
        :param n_trials: number of trials (Default value = NumberOfTrials)
        :type n_trials: int
        :param seed: seed of the random number generator (Default value = SEED)
        :type seed: int
        :raises ValueError: if lambda0 is not positive or the other inputs are wrong
        """
        self._setup(p, tau, n_trials, seed, name="Thompson Sampling")
        if lambda0 <= 0:
            logger.error(f"lambda0 must be positive, got {lambda0}")
            raise ValueError("lambda0 must be positive")
        self.mu0 = float(mu0)
        self.lambda0 = float(lambda0)
        self.mu = np.full(self.n_bandits, self.mu0)  # posterior mean
        self.lam = np.full(self.n_bandits, self.lambda0)  # posterior precision
        self.sum_x = np.zeros(self.n_bandits)  # sum of the rewards of every bandit
        self._live_estimates = self.mu

    def __repr__(self):
        """Describe the algorithm and its settings.

        :returns: text like ``ThompsonSampling(bandits=[...], precision=1, prior=Normal(...), ...)``
        :rtype: str
        """
        return (f"ThompsonSampling(bandits={self.true_means.tolist()}, "
                f"precision={self.tau:g}, prior=Normal({self.mu0:g}, precision {self.lambda0:g}), "
                f"trials={self.n_trials})")

    def choose_arm(self, t):
        """Choose the bandit of trial ``t``: sample every posterior, take the highest sample.

        :param t: trial number, starts at 1 (not used, the posterior already has the history)
        :type t: int
        :returns: index of the chosen bandit (0 based)
        :rtype: int
        """
        # one sample from the posterior of every bandit, pull the highest
        samples = self.rng.normal(self.mu, 1.0 / np.sqrt(self.lam))
        return int(np.argmax(samples))

    def pull(self, arm):
        """Pull a bandit.

        :param arm: index of the bandit
        :type arm: int
        :returns: the reward
        :rtype: float
        """
        return self._draw_reward(arm)

    def update(self, arm, reward):
        """Update the Normal posterior of the pulled bandit (known precision).

        :param arm: index of the bandit that was pulled
        :type arm: int
        :param reward: reward that was received
        :type reward: float
        """
        self.pulls[arm] += 1
        self.sum_x[arm] += reward
        self.lam[arm] += self.tau
        self.mu[arm] = (self.lambda0 * self.mu0 + self.tau * self.sum_x[arm]) / self.lam[arm]

    def experiment(self):
        """Run the experiment: ``n_trials`` trials of Thompson Sampling."""
        logger.info(f"start: {self!r}")
        self._run_trials()
        logger.info(f"{self.name}: done, posterior means = {np.round(self.mu, 3).tolist()}, "
                    f"posterior std = {np.round(1 / np.sqrt(self.lam), 3).tolist()}")

    def report(self, csv_path=CSV_PATH, append=False):
        """Store the rewards in the csv file and log cumulative/average reward and regret.

        :param csv_path: csv file, columns {Bandit, Reward, Algorithm} (Default value = CSV_PATH)
        :type csv_path: str or pathlib.Path
        :param append: ``True`` adds the rows to an existing file (Default value = False)
        :type append: bool
        """
        self._report(csv_path, append)


############################### COMPARISON
def _repeat(make_algorithm, n_runs, first_seed):
    """Run the same algorithm several times with different seeds.

    :param make_algorithm: function that gets a seed and returns a new algorithm
    :type make_algorithm: callable
    :param n_runs: number of repetitions
    :type n_runs: int
    :param first_seed: seed of the first run (the others are first_seed + 1, ...)
    :type first_seed: int
    :returns: algorithms after their experiment
    :rtype: list
    """
    runs = []
    for i in range(n_runs):
        algo = make_algorithm(first_seed + i)
        algo._run_trials()
        runs.append(algo)
    return runs


def comparison(n_runs=N_RUNS):
    """Compare the two algorithms visually and with numbers over many repetitions.

    One run can be lucky or unlucky, so both algorithms are repeated ``n_runs``
    times (different seeds). The figure has three panels: mean cumulative regret
    with a 95% confidence band, share of trials where the best bandit was chosen,
    and the final regret of every run (box plot). The numbers are logged as a table.

    :param n_runs: number of repetitions of each algorithm (Default value = N_RUNS)
    :type n_runs: int
    :returns: one row per algorithm with the summary numbers
    :rtype: pandas.DataFrame
    """
    logger.info(f"comparison: {n_runs} runs of each algorithm, {NumberOfTrials} trials each")
    makers = {
        "Epsilon Greedy": lambda s: EpsilonGreedy(Bandit_Reward, seed=s),
        "Thompson Sampling": lambda s: ThompsonSampling(Bandit_Reward, seed=s),
    }
    runs = {name: _repeat(make, n_runs, SEED + 1000) for name, make in makers.items()}

    x = np.arange(1, NumberOfTrials + 1)
    colors = {"Epsilon Greedy": "#1f77b4", "Thompson Sampling": "#d62728"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    rows, final_regret, final_reward = [], {}, {}
    for name, algos in runs.items():
        regret = np.array([a.cumulative_regrets for a in algos])
        reward = np.array([a.cumulative_rewards for a in algos])
        optimal = np.array([np.cumsum(a.chosen == a.best_bandit) / x for a in algos])
        mean_regret = regret.mean(axis=0)
        half_width = 1.96 * regret.std(axis=0, ddof=1) / np.sqrt(n_runs) if n_runs > 1 else 0 * mean_regret
        axes[0].plot(x, mean_regret, color=colors[name], label=name)
        axes[0].fill_between(x, mean_regret - half_width, mean_regret + half_width,
                             color=colors[name], alpha=0.25)
        axes[1].plot(x, 100 * optimal.mean(axis=0), color=colors[name], label=name)
        final_regret[name] = regret[:, -1]
        final_reward[name] = reward[:, -1]
        rows.append({"Algorithm": name,
                     "runs": n_runs,
                     "mean cumulative reward": final_reward[name].mean(),
                     "mean cumulative regret": final_regret[name].mean(),
                     "std of regret": final_regret[name].std(ddof=1) if n_runs > 1 else 0.0,
                     "median regret": np.median(final_regret[name]),
                     "% best bandit chosen": 100 * optimal[:, -1].mean()})
    axes[0].set_title("Mean cumulative regret (95% band)")
    axes[0].set_ylabel("cumulative regret")
    axes[1].set_title("Share of trials with the best bandit")
    axes[1].set_ylabel("% of trials so far")
    axes[1].set_xscale("log")
    for ax in axes[:2]:
        ax.set_xlabel("trial" + (" (log scale)" if ax is axes[1] else ""))
        ax.grid(alpha=0.3)
        ax.legend()
    axes[2].boxplot([final_regret[n] for n in runs])
    axes[2].set_xticklabels(list(runs))  # (the labels= argument changed between matplotlib versions)
    axes[2].set_title(f"Final cumulative regret of {n_runs} runs")
    axes[2].set_ylabel("cumulative regret")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    path = IMG_DIR / "comparison_multi_run.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"comparison figure saved: {path.name}")

    summary = pd.DataFrame(rows)
    logger.info("comparison table:\n" + summary.round(2).to_string(index=False))
    eg, ts = final_regret["Epsilon Greedy"], final_regret["Thompson Sampling"]
    logger.info(f"Thompson Sampling had the lower cumulative regret in "
                f"{100 * np.mean(ts < eg):.0f}% of the {n_runs} paired runs")
    return summary


def tune(n_runs=N_RUNS, epsilons=(1, 2, 5, 10, 20, 50, 100), priors=(1.0, 0.1, 0.01)):
    """Extra: check how sensitive the two algorithms are to epsilon0 and the prior.

    A run is called *stuck* when its cumulative regret is above 1000, i.e. it spent
    a long time on a bad bandit. This is how ``EPSILON_0`` and ``PRIOR_PRECISION``
    were chosen.

    :param n_runs: repetitions for every setting (Default value = N_RUNS)
    :type n_runs: int
    :param epsilons: values of ``epsilon0`` to test (Default value = (1, 2, 5, 10, 20, 50, 100))
    :type epsilons: tuple
    :param priors: values of the prior precision ``lambda0`` to test (Default value = (1.0, 0.1, 0.01))
    :type priors: tuple
    :returns: one row per tested setting
    :rtype: pandas.DataFrame
    """
    rows = []
    for e in epsilons:
        regret = np.array([a.cumulative_regrets[-1] for a in _repeat(
            lambda s: EpsilonGreedy(Bandit_Reward, epsilon0=e, seed=s), n_runs, SEED + 2000)])
        rows.append({"Algorithm": "Epsilon Greedy", "setting": f"epsilon0={e:g}", "mean regret": regret.mean(),
                     "std": regret.std(ddof=1), "median": np.median(regret),
                     "% stuck": 100 * np.mean(regret > 1000)})
        logger.info(f"tune: epsilon0={e:g} done")
    for lam in priors:
        regret = np.array([a.cumulative_regrets[-1] for a in _repeat(
            lambda s: ThompsonSampling(Bandit_Reward, lambda0=lam, seed=s), n_runs, SEED + 3000)])
        rows.append({"Algorithm": "Thompson Sampling", "setting": f"lambda0={lam:g}", "mean regret": regret.mean(),
                     "std": regret.std(ddof=1), "median": np.median(regret),
                     "% stuck": 100 * np.mean(regret > 1000)})
        logger.info(f"tune: lambda0={lam:g} done")
    table = pd.DataFrame(rows)
    logger.info(f"tuning results ({n_runs} runs per setting):\n" + table.round(1).to_string(index=False))
    return table


def main():
    """Run the whole homework: experiment, report, plots and comparison."""
    parser = argparse.ArgumentParser(description="Epsilon Greedy vs Thompson Sampling")
    parser.add_argument("--runs", type=int, default=N_RUNS, help="repetitions in comparison() (0 = skip it)")
    parser.add_argument("--tune", action="store_true", help="also run the epsilon0 / prior sensitivity check")
    parser.add_argument("--debug", action="store_true", help="show the debug messages")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.debug else "INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    logger.debug(f"bandit rewards = {Bandit_Reward}, trials = {NumberOfTrials}, seed = {SEED}")
    epsilon_greedy = EpsilonGreedy(Bandit_Reward)
    thompson = ThompsonSampling(Bandit_Reward)

    epsilon_greedy.experiment()
    thompson.experiment()

    epsilon_greedy.report(CSV_PATH, append=False)  # first one creates the csv
    thompson.report(CSV_PATH, append=True)  # second one adds its rows

    viz = Visualization([epsilon_greedy, thompson])
    viz.plot1()
    viz.plot2()

    if args.runs > 0:
        comparison(args.runs)
    if args.tune:
        tune(args.runs if args.runs > 0 else N_RUNS)
    logger.info("finished")


if __name__ == '__main__':
    try:
        main()
    except Exception:
        logger.critical("the experiment stopped because of an error")
        raise
