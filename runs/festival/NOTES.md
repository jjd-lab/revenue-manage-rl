# Festival passes: does RL beat the planner once products share seats? — NOTES

**Scenario:** `reservation_pricing.festival`, `FestivalConfig` defaults. Three nights (Fri, Sat, Sun), 10,000 seats each, six passes: every run of consecutive nights. Saturday is the strong night, Sunday the soft one.
**Demand:** each day a Poisson number of buyers arrive (100,000 a season on average, more of them close to the date). Each buys one pass on sale, or nothing, by multinomial logit. A dearer or closed pass sends buyers to other nights and lengths.
**Levers:** a per-night price for each pass ($80–$120 a night) and a selling limit for each night (10,000–15,000): nine numbers a day.
**Seasons differ:** each draws its own market size (15% spread) and night appeal. Decision code knows only the usual values.
**Held-out:** seeds 0–29. Training envs shift every seed by 1,000,000, so no training season is a test season.
**Score:** revenue − $200 per empty seat − $400 per denied admission, summed over the three nights. Nothing is capped.
**Regenerate:** `rprl-train -c configs/festival_sac.yaml`, `rprl-bc-sac -c configs/festival_bc_sac.yaml`, then `python runs/festival/run_festival.py`.

## The question

Recent work finds RL catches up with a fitted planner once the problem grows past a single product (Razumovskiy and Karenin, 2026). Does sharing seats across six passes, with buyers moving between nights, change §14–§19's answer?

## The policies

- **planner:** each day re-solves a fluid program over the rest of the season in ten blocks, then charges the first block's prices. In logit market shares the program is convex, so the solver finds its optimum. Each night's limit stops sales once expected show-ups, at the usual keep rate, reach capacity.
- **planner + pickup:** also rescales the market size from booking requests seen so far.
- **fixed prices:** the same program solved once, for one price per pass all season.
- **SAC:** joint SAC on the score, gamma 1, seed 7, 200k steps (2,000 seasons).
- **BC → SAC:** SAC warm-started by cloning the planner on 100 training seasons, then the same 200k steps. **clone only** is that warm start before fine-tuning.
- "kept checkpoint" is the best of ten checks during training, each on 10 training-time seasons.

## Results

| policy | score | vs planner | 95% interval |
| --- | ---: | ---: | --- |
| **planner** | **2,421,254** | — | — |
| planner + pickup | 2,439,165 | +17,910 | [−10,259, +45,305] |
| fixed prices | 2,096,618 | −324,636 | [−375,455, −275,225] |
| BC → SAC | 2,075,890 | −345,365 | [−449,129, −261,840] |
| BC → SAC, kept checkpoint | 2,014,592 | −406,662 | [−489,823, −329,266] |
| clone only | 1,994,116 | −427,139 | [−493,764, −357,518] |
| SAC | 1,956,530 | −464,725 | [−507,709, −424,387] |
| SAC, kept checkpoint | 1,767,089 | −654,165 | [−730,049, −582,594] |

Mean per-night price and passes sold per season:

| policy | Fri | Sat | Sun | Fri+Sat | Sat+Sun | all 3 | empty seats | denied |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| planner | $86 · 9,134 | $107 · 7,685 | $80 · 7,661 | $83 · 2,426 | $80 · 2,004 | $80 · 414 | 2,934 | 108 |
| BC → SAC | $86 · 9,367 | $101 · 8,413 | $81 · 7,547 | $90 · 1,706 | $82 · 1,830 | $82 · 348 | 3,998 | 186 |
| SAC | $85 · 9,170 | $98 · 8,726 | $82 · 7,526 | $85 · 1,905 | $84 · 1,492 | $88 · 186 | 4,431 | 74 |

Full CSVs: `table.csv`, `per_season.csv`, `paired_intervals.csv`.

## Reading

- **Every learned policy trails the planner, and each interval is below zero.** The best, BC → SAC, is 14.3% behind; plain SAC 19.2%. On one night (§14–§18) the gap was 4–8%. Six passes sharing seats widened it.
- **Re-planning is worth 13%.** Fixed prices lose 324,636 to the planner. The best learned policy scores about the same as fixed prices (point estimates; no interval computed between them).
- **The learned policies sell Saturday too cheaply and bundles too dearly.** SAC charges $98 for a Saturday night against the planner's $107. It sells about 1,000 more 1-day Saturday passes and 21–55% fewer 2- and 3-day passes, and ends with about 1,500 more empty seats a season. The planner uses Saturday's demand to carry the weaker nights: it keeps the Saturday night dear and prices the bundles near the floor. The learned policies have not found that trade. (Empty seats are recorded per season, not per night.)
- **Fine-tuning helped the clone, a little.** The clone alone is 17.6% behind; 200k SAC steps took it to 14.3%. The clone prices Saturday at $118, too high, and leaves 4,777 seats empty.
- **The kept checkpoints are worse than the final ones.** Ten training-time seasons are too few to pick a checkpoint from; the final weights are the ones to report.
- **Pickup ties.** The planner already re-solves from its own bookings each day, so rescaling the market adds nothing measurable (§18 found the same).

## Is it the network, or the training? DAgger

The clone matches the planner closely on the planner's own seasons: prices within $0.40 a night, limits within about 45 seats. Played alone, it drifts. Asking the planner what it would do in the clone's own states, the Saturday price is $19 off by days 25–49, and the limits are 700–850 seats off in the last 25 days. The clone only ever saw states the planner reaches.

`run_dagger.py`: each round the clone plays 30 new training seasons, the planner labels every state it reaches, and the clone is refit on everything labelled so far (30 epochs, same MSE fit as `rprl-bc-sac`). Round 0 is the shipped clone. Output: `dagger.csv`, `dagger_per_season.csv`.

| round | labels | score | vs planner | 95% interval |
| ---: | ---: | ---: | ---: | --- |
| 0 | 10,000 | 1,994,116 | −427,139 | [−493,764, −357,518] |
| 1 | 13,000 | 2,379,324 | −41,930 | [−68,367, −17,027] |
| 2 | 16,000 | 2,411,444 | −9,811 | [−36,093, +17,221] |
| 3 | 19,000 | 2,397,811 | −23,443 | [−47,168, +316] |
| 4 | 22,000 | 2,406,063 | −15,191 | [−32,258, +2,792] |
| 5 | 25,000 | 2,398,254 | −23,001 | [−43,958, −2,144] |

- **One round takes the clone from 17.6% behind to 1.7%; from round 2 it sits 0.4–1.0% behind.** Rounds 2–4 tie the planner; round 5's interval just excludes zero.
- **The observation is enough.** A network seeing what the RL agents see can play the planner's strategy almost exactly. SAC's 14–19% gap is not missing information; it is that trial and error over 2,000 seasons does not find the strategy. The suspects are the training budget, the empty-seat and denied-admission charges arriving only on the last day, and settings carried over from the one-night runs.
- **This copies the planner; it does not beat it.**

## Does trial and error improve on the DAgger clone?

`train_dagger_sac.py`: SAC starting from the round-5 clone (the last round, fixed in advance). The replay buffer is filled with 100 of the clone's own training seasons, the critic is warmed for 5,000 steps with the actor frozen, the entropy coefficient starts at 0.01, then the same 200k steps as the other runs. Scored in `run_festival.py` as "DAgger -> SAC".

| policy | score | vs planner | 95% interval |
| --- | ---: | ---: | --- |
| DAgger clone | 2,398,254 | −23,001 | [−43,958, −2,144] |
| DAgger → SAC | 2,139,864 | −281,391 | [−321,250, −238,709] |

- **Fine-tuning undid most of the clone.** It lost 258,000 a season of the clone's value and finished 11.6% behind the planner, below the planner on all 30 seasons. The warm start held for the first 20k steps; after that the training-time checks swung between 1.66M and 2.64M.
- **It drifted toward the same pattern as every other learned policy.** Saturday went from $106 to $103 a night; the 2- and 3-day passes rose $1–3 a night and sold 7–21% less than under the clone. The planner's trade, a dear Saturday that carries the bundles and the weaker nights, is one SAC's updates move away from, not toward.
- **The likely reason is the learning signal, not the network.** Selling a 1-day Saturday pass pays the same day. What a bundle is worth — seats filled on the other nights — is only charged on the last day, 20–100 days later, and gamma is 1, so the critic has to carry that back through the whole season. A critic that underrates the bundles makes SAC's update steer away from them. The next section tests this and does not bear it out.
- One training seed; the swings in the training curve are larger than the gap to the clone, so a second seed could land elsewhere.

## Does an earlier signal help? Reward shaping

`FestivalConfig.shape_reward`: each day's reward also carries the change in the end-of-season charge if selling stopped now, from the operator's show-up estimate (potential-based shaping, Ng, Harada & Russell 1999; the potential is zero once the season ends). With gamma 1 the shaping sums to a constant over a season (`test_shaping_adds_only_a_constant_over_a_season`), so the best policy is unchanged; filling a seat pays the day it is sold. Runs: `configs/festival_sac_shaped.yaml` (SAC from scratch) and `train_dagger_sac.py --shaped` (from the DAgger clone), otherwise as before.

| policy | score | vs planner | vs the same run unshaped | 95% interval |
| --- | ---: | ---: | ---: | --- |
| SAC, shaped | 1,806,794 | −614,461 | −149,736 | [−262,646, −43,105] |
| DAgger → SAC, shaped | 1,917,265 | −503,989 | −222,599 | [−304,702, −148,300] |

- **Shaping made both runs worse, and each gap is real.** The late-charge explanation is not supported, at least not in a form this shaping reaches.
- **SAC from scratch started overbooking.** Denied admissions rose from 74 to 908 seats a season, on 1.1 nights on average against 0.3.
- **The fine-tune still drifted to the same pattern:** Saturday at $100 against the clone's $106, 299 three-day passes against 410. The shaped fine-tune fell to 1.26M at the first training-time check, lower than the unshaped one ever went.
- **What is left as an explanation is SAC's own updates on this problem** — how its critic generalizes across nine continuous actions, and its exploration — rather than when the reward arrives. One training seed per run; a critic that learns the shaped values badly could also explain it, and was not checked.

## Correction (2026-09-25)

The first evaluation scored BC → SAC and the clone on seasons they had trained on: the planner demonstrations were collected on seeds 7–106, overlapping 23 of the 30 test seeds. Training envs now shift every seed by 1,000,000 (`make_festival_env`), both policies were retrained, and every row above comes from the rerun. The leak had flattered the clone by about 217,000 (−210,083 then, −427,139 now).

## Caveats

- One training seed per learned policy, 200k steps each. The single-night policies needed similar budgets; nine controls may need more.
- The planner is handed the true logit structure and the usual rates; seasons differ from them only in market size, night appeal and show-up rates. A planner that has to fit its demand model from data is the next test.
- The planner's program is fluid: it plans with expected demand and ignores its spread. It is a strong heuristic, not the exact optimum, so the true gap to optimal may be larger.
- The cutback when a night fills treats every pass on that night alike, and buyers cut are lost rather than moved to another pass.
