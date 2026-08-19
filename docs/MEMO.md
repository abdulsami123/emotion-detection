# Technical Memo — Voice Tone & Background Noise

**AutoAce AI technical trial.** Prepared 2026-08-17.

---

## 1. Headline result

Measured on the three provided labelled calls. This is a **regression check, not a validation
set** — with n=3 it establishes that the system reproduces known-good behaviour and nothing more.

| field group | metric | result |
|---|---|---|
| `background_noise_present` | accuracy | **1.00** (3/3) |
| `background_noise_severity` | macro-F1 | **1.00** |
| `audio_quality` | accuracy | **1.00** (3/3) |
| `speaker_overlap_present` | accuracy | **1.00** (3/3) |
| `long_silence_present` | accuracy | **1.00** (3/3) |
| `background_noise_type` | accuracy | 0.67 (2/3) |
| `emotional_intensity` | accuracy | 0.67 — **equal to** the constant-`medium` baseline |
| `emotional_tone` | accuracy | 0.33, macro-F1 0.25 |

**The six deterministic signal fields are effectively solved; tone is not.** That split is the
honest summary and it is not an accident — it is what the architecture was designed around after
early analysis showed the tone problem is far harder on this data than it first appears.

*Tone and intensity are from the **live** `gpt-4o-mini` path, measured 2026-08-17.*

**The tone classifier is OpenAI `gpt-4o-mini`.** An earlier Anthropic Haiku 4.5 implementation was
never verifiable — that key returned HTTP 401 — and the provider was switched at the trial owner's
request. Only `classify_tone` is vendor-specific: `build_prompt` and `parse_response` are
provider-agnostic, so the prompt design survived the swap untouched.

**Tone improved from 0/3 to 1/3 during that work, and the cause was not the model.** Agent and
customer roles were **inverted on call_001**, so the tone branch was analysing the bot's uniformly
flat TTS delivery as the customer's. See §6(e) — the single largest defect found in this build.

---

## 2. Approaches compared

The brief requires comparing at least two materially different approaches. Three were built and
measured; several more were built and **refuted**, which is reported in §6 because the negative
results shaped the final design more than the positive ones.

| approach | role | status |
|---|---|---|
| **Prosody + SER + LLM over an annotated transcript** | primary tone | built; **unverified live** (401) |
| **Zero-shot NLI (`bart-large-mnli`) over the transcript** | second approach, no-API fallback | built and measured: tone 0.33 |
| **Deterministic DSP + AudioSet tagging** | all six signal fields | built and measured: 5 of 6 at 1.00 |

### Why the tone design is prosody-primary

Text alone scores at or below chance on this data. These are **AI-voice-agent calls** — a TTS bot
("Erica", a dealership receptionist) talking to human callers:

| call | label | what the customer actually says | what a text classifier reads |
|---|---|---|---|
| call_001 | **upset / high** | "Are you a real person?" then "Hello?" ×5 — ~8 words | neutral or confused |
| call_002 | **neutral / medium** | "Spanish, please." — two words | no signal at all |
| call_003 | **satisfied / medium** | long; repeatedly told the dealership is closed | **frustrated** |

call_003 is decisive: semantically the caller is blocked over and over, and the label is
`satisfied` because they stay warm and close with "Thank you". The lexical content points the
wrong way. So the design makes prosody primary and lets the transcript supply *situation* rather
than sentiment — and the system prompt encodes both failure directions explicitly (politeness
masking, and escalating repetition with neutral words).

The NLI result confirms the analysis: it returns `distressed` for both call_001 and call_002,
i.e. it reads short, fragmentary caller speech as distress. Text-only is not viable here.

---

## 3. Final architecture

```
raw audio -> decode 16k mono (no loudness normalization, ever)
             |
             +-- Silero VAD ---------------- shared speech / non-speech segmentation
             +-- ECAPA 2-cluster ----------- agent vs customer
             +-- faster-whisper ------------ transcript, word timestamps, language
                        |
      +-----------------+------------------+
      |                                    |
  SIGNAL BRANCH (raw audio)          TONE BRANCH (customer segments)
  floor -> present, severity         eGeMAPS -> speaker-relative z -> tags
  AST   -> type                      audeering SER -> arousal/dominance/valence
  SQUIM + DSP -> audio_quality       gpt-4o-mini over annotated transcript
  ECAPA intra-segment -> overlap        \-> NLI fallback if unavailable
  dead air -> long_silence           activation x trajectory -> intensity
      |                                    |
      +-----------------+------------------+
                        |
              merge + confidence from voter agreement -> 9-field JSON
```

**Absolute level is never normalized.** The noise floor in dB is both the primary presence
detector and the primary severity signal; normalizing gain destroys it. The tone branch obtains
level-invariance internally by z-scoring against each speaker's own baseline.

**The signal branch never touches an API or a network.** That is why it still returned all six
fields correctly while the tone branch was degraded by an invalid API key — the worst available
failure mode would have been losing the reliable points because a remote service was down.

---

## 4. Cost analysis

**Ceiling: $0.003 per audio minute.**

**Token usage is measured against the live API, not estimated.** A 30.9-second call sends
**2712 input tokens** and returns 124 output tokens (`gpt-4o-mini`) or 205 (`gpt-4.1-mini`). The
earlier figure of ~830 input tokens was an estimate and was wrong by 3.3× — the third cost-model
error in this build, all three found by measuring rather than reasoning.

Roughly **2500 of those input tokens are the fixed system + few-shot prefix**, so LLM cost per
*audio minute* is dominated by fixed prompt overhead on short calls and amortises on long ones:

| call length | input tokens | `gpt-4o-mini` $/call | $/audio-min |
|---|---|---|---|
| 0.5 min (call_001) | 2712 | $0.00048 | **$0.00093** |
| 2.9 min (call_003) | ~3700 | $0.00071 | **$0.00025** |

**Prompt caching is reachable here and was not on the previous provider.** OpenAI caches prefixes
over 1024 tokens automatically at half price and the ~2500-token prefix qualifies, so the few-shots
sit in a stable `_system_content()` prefix with only the per-call payload in the user message.
Haiku's 4096-token cache minimum was never met by this prompt, so this is a genuine gain from the
switch rather than a wash.

Rates assumed: `gpt-4o-mini` $0.15 / $0.60 per MTok, `gpt-4.1-mini` $0.40 / $1.60. **These should be
re-verified against OpenAI's current pricing page before submission** — they are the one input here
not measured directly from the API.

`gpt-4.1-mini` is the documented upgrade candidate at roughly 3× the input cost. Both models
returned the correct `upset / high` on call_001 in an isolated smoke test, so the choice is not
decidable on three labelled calls and is recorded as an A/B for a larger set.

### Measured compute

| path | instance | $/hr | s/audio-min | compute | **total** | vs ceiling |
|---|---|---|---|---|---|---|
| CPU | 4 vCPU spot | $0.05 | 62 | $0.00086 | **$0.00219** | 27% headroom |
| CPU | 4 vCPU on-demand | $0.10 | 62 | $0.00172 | **$0.00305** | ❌ breaches by 2% |
| GPU | T4 spot | $0.20 | ~14 | $0.00078 | **$0.00211** | 30% headroom |
| GPU | T4 on-demand | $0.35 | ~14 | $0.00136 | **$0.00269** | 10% headroom |

**CPU break-even: $0.0968/hr.** GPU figures are projected from stage profiles, not measured — no
CUDA device was available. **Recommended deployment: GPU**, which holds across the realistic
price range where CPU-on-demand does not.

### Excluded, rather than silently omitted

One-time model download and warm-up; dashboard hosting (fixed, not per-minute); storage. Other
assumptions: batch utilisation ≥70%, model weights resident across calls, no retry overhead.

### The cost model was wrong twice, in opposite directions

Worth stating plainly, because it is the most transferable lesson here:

| stage | estimated | measured | error |
|---|---|---|---|
| ASR | 60 s/audio-min | **24.5** | 2.4× too pessimistic |
| SQUIM quality | 1.5 s/audio-min | **28.6** | 19× too optimistic |
| AST typing | 1.0 s/audio-min | **30.9** | 31× too optimistic |

The ASR figure was inherited from an `openai-whisper` run rather than measured on the
CTranslate2 build actually shipped. SQUIM and AST were guessed. Correcting them **inverted** the
original conclusion that "GPU is both faster and cheaper" — at spot pricing the two are within
noise, and GPU is now recommended on **latency**, not cost.

Both heavyweight stages also scaled with call *duration*. But `background_noise_type` and
`audio_quality` are call-**level** properties, so neither needs every window: sampling a fixed
number of evenly-spaced windows (AST 6, SQUIM 3) made both **O(1) per call**, taking the 172-second
call from 1.13× to 0.36× realtime with every assertion preserved.

---

## 5. Latency analysis

Measured end to end, CPU only (torch 2.13.0+cpu, no CUDA), models resident:

| call | audio | wall-clock | × realtime |
|---|---|---|---|
| call_001 | 30.9s | 57.3s | 1.85× |
| call_002 | 35.0s | 58.4s | 1.67× |
| call_003 | 171.9s | 129.2s | **0.75×** |
| **total** | 237.8s | 244.9s | **1.03×** |

Short calls are dominated by fixed per-call costs (model warm-up, the O(1) AST/SQUIM samples), so
the marginal rate on long calls is the meaningful one. A 50-call batch of 3-minute calls runs in
roughly 1.9 hours on one CPU worker; stages are per-file independent, so a pool of 4 gives close
to linear speedup.

**CPU-only meets a batch target but not an interactive one.** GPU is the recommendation for
throughput.

---

## 6. What measurement refuted

Four design decisions did not survive contact with the data. These are reported because a design
doc that only records what worked is not a rigorous artifact.

**(a) "Tag non-speech regions only" made typing worse.** The argument was that speech masks
background noise. Measured, it inverts: whole-clip AST maps call_002 to its ground-truth `TV`,
while non-speech-only returns `music`. In telephony the gaps are near-**silent** — which is
exactly *why* the noise floor works as a presence detector — while the television is audible
*underneath* the speech. Removing the speech removed the evidence.

**(b) The spectral transmission-noise classifier does not work.** AudioSet genuinely is blind to
line artifacts (`static` scores 0.0013–0.0059 on every call, including the one whose ground truth
*is* "sharp static"). But no proposed replacement discriminates either:

| call | truth | flatness | transients/s | crest | kurtosis |
|---|---|---|---|---|---|
| call_001 | *(none)* | 0.0531 | **0.47** | **36.5** | **542** |
| call_002 | TV | 0.1157 | 0.28 | 12.9 | 81 |
| call_003 | **sharp static** | **0.0426** | 0.29 | 21.0 | 109 |

Flatness is *lowest* on the static call where broadband hiss should be highest; transient rate,
crest and kurtosis are all *highest* on the clean call; a 6 dB mains-hum test fired on all three.
The detector is **disabled** rather than shipped as an arbitrary-firing label, and call_003's type
is pinned as `xfail(strict=True)` with the measurements in the reason. This is the one field where
we knowingly emit a wrong answer (`radio` instead of `sharp static`).

**(c) Two threshold sets were internally inconsistent with their own anchors.**
`NOISE_SEVERITY_BANDS` placed the low/medium boundary at −50.0 while annotating −52.1 as a
`medium` anchor — so that anchor fell into `low` by construction, and call_002's severity was
wrong. Separately, the coherence-check valence bounds (0.45 / 0.55) sat *inside* the measured SER
range (valence spans only 0.534–0.638 on this audio), so the check would have fired on roughly
half of all calls regardless of correctness, penalising correct answers through the confidence
formula. Both corrected, both now covered by tests that assert the label the schema actually
emits rather than merely that values are ordered.

**(d) The intensity self-baseline was degenerate on short calls.** `BASELINE_WINDOW_S` is 25s but
the two short calls have only 7.1s and 12.4s of customer speech, so the baseline was built from
**all** the segments it then scored. Z-scoring a set against its own mean gives exactly zero *by
construction*:

| call | customer speech | baseline coverage | mean activation z |
|---|---|---|---|
| call_001 | 7.1s | **100%** | **−0.000** |
| call_002 | 12.4s | **100%** | **+0.000** |
| call_003 | 73.8s | 37% | +0.240 |

Zero activation trips the `low` threshold, so both short calls reported `low` from no evidence at
all — dragging intensity to 0.33, **below** the 0.67 majority-class baseline. Degenerate baselines
are now detected and routed to the Tier C path (emit the prior, cap confidence). Intensity rose to
0.67 and confidence on those calls dropped to 0.25, so the *absence* of a measurement is visible
instead of hidden behind a fabricated label.

---

**(e) Agent and customer roles were inverted on one of three calls — the largest defect found.**
Diarization clusters voices correctly, then has to decide which cluster is the bot, and its anchor
was *"the agent greets first"*. On call_001 the caller opens with an impatient `"Come on."` at 1.2s
and the bot greets at 3.7s, so that anchor picked the wrong cluster and **inverted every role in
the call**:

| segment | text | assigned (before) |
|---|---|---|
| 1.2–2.0 | "Come on." | agent ✗ |
| 3.7–6.9 | "Hi, I'm Erica from Toyota of Braintree. How can I help?" | customer ✗ |
| 8.7–11.3 | "Yes, hi. Are you a real person?" | agent ✗ |

The tone branch therefore spent the whole run analysing the bot's uniformly flat TTS delivery as
if it were the customer's, which is why tone scored **0/3 regardless of provider** — the same
model, given the caller's actual utterances by hand, returned the correct `upset / high`.

Fixed by `pipeline.verify_roles`, which re-anchors on **transcript content** (phrases a
receptionist structurally says — "how can I help", "transferring you", plus Spanish equivalents)
rather than on turn order, and only swaps on a strict majority of evidence. Phrases are
deliberately *not* name-specific, since the hidden set may use a different bot or dealership.
Measured: call_001 swaps and its tone becomes correct; calls 002 and 003 were already right and
are untouched.

Two lessons worth carrying: an assumption that holds on two of three examples is not a rule, and
the test asserting it (`test_agent_speaks_first`) claimed it was *"true on all three provided
calls"* while only ever checking the one where it held. Both are now corrected, with the
counter-example pinned.

---

## 7. Calibration position

**Thresholds are derived from definitions and physics, then validated against three points — not
fitted.** Every threshold lives in `autoace/config.py` and is annotated `MEASURED`, `DERIVED` or
`UNFITTED`. Class coverage in the labelled set is severely incomplete:

- `emotional_tone`: 3 of 5 classes (`frustrated`, `distressed` never appear as truth)
- `emotional_intensity`: 2 of 3 (`low` never appears)
- `background_noise_severity`: 2 of 4 (`low`, `high` never appear)
- `audio_quality`: 1 of 3 (only `clear`)
- `long_silence_present`: 1 of 2 (only `false`)

Two boolean fields are fully covered. **Five of the eight classified fields lack full coverage**,
so the `low`/`medium` intensity boundary and the `low`/`high` severity boundaries have no
supporting example at all.

**Leakage prevention.** No model is trained; the LLM is prompted and the thresholds are reasoned,
so leakage risk is confined to prompt engineering against three examples. Mitigations: the three
calls are a regression test rather than a tuning set; few-shot examples are **synthetic** rather
than drawn from the labelled calls; and the ECAPA agent-reference mechanism uses speaker identity
only and never reads `labels.csv`. No production module under `autoace/` reads the labels file.

**The three-call figures are therefore not an unbiased accuracy estimate**, and the confusion
matrices are a formality at this n.

---

## 8. Failure modes and limitations

1. **Tone is unverified on its primary path.** The API key returns 401; the measured 0.33 is the
   NLI fallback. Highest-priority item.
2. **Code-switching breaks speaker assignment.** On call_002, ECAPA separates the bot's *English*
   voice from its own *Spanish* voice rather than bot from human: **12.36s attributed to the
   customer against a true ~0.9s**. The `degraded` flag does not fire because the clusters are
   well separated — along the wrong axis. Proven unfixable at cluster level (the customer's single
   0.9s segment never forms its own cluster, so the floor is 5.2s). Pinned `xfail(strict=True)`.
   **The fix is now proven viable:** `asr.detect_language()` returns `en` on call_002's opening 8s
   and `es` on its final 12s, so per-segment language ID can merge the two clusters.
3. **Line-artifact typing is wrong by design** (§6b). `background_noise_type` will name the
   nearest environmental source for a call whose real noise is a codec or line artifact.
4. **Speaker assignment can be decided by noise.** On call_001 a reference similarity of 0.5529
   sits 0.003 above the 0.55 threshold.
5. **SER axes are compressed** on telephony audio (valence 0.534–0.638, arousal 0.594–0.646), far
   from the [0,1] the model nominally emits, so absolute-threshold logic against them is fragile.
6. **Averaging dilutes short emotional peaks.** SER and AST both summarise at call level from a
   fixed window sample; a brief escalation outside those windows is invisible.
7. **Multilingual prosody.** eGeMAPS speech-rate features are language-sensitive, and both the SER
   model and the LLM's prosody interpretation are English-centric. call_002 is Spanish, so this is
   live on the provided set, not hypothetical.
8. **No cross-validation is possible.** Leave-one-call-out on n=3 is not meaningful.

---

## 9. External API disclosure (brief §11)

| item | detail |
|---|---|
| Model | **`gpt-4o-mini` (OpenAI)** |
| Pricing assumed | $0.15 / MTok input, $0.60 / MTok output — **re-verify before submission** |
| Measured usage | 2712 input / 124 output tokens on a 30.9s call |
| Per audio minute | **$0.00093** on a 0.5-min call, **$0.00025** on a 2.9-min call |
| Upgrade candidate | `gpt-4.1-mini` at ~3× input cost; A/B pending a larger labelled set |
| Retention | zero-retention / no-training to be confirmed on the account before production use |
| **Does audio leave AutoAce infrastructure?** | **No.** Audio is never transmitted. |
| What does leave? | The customer-side transcript, discretised prosody tags, the SER triple, and an agent-behaviour summary |
| Fallback if egress is refused | `autoace/tone_nli.py` runs `bart-large-mnli` locally; no network at all |

The signal branch makes no external calls whatsoever. Confidential audio and transcripts derived
from it are excluded from version control (`reference/`, `tests/fixtures/asr_baseline.json`) because
the target repository is public.

---

## 10. Next steps, in priority order

1. **Fix code-switched speaker assignment** using per-segment `detect_language()` — viability
   already demonstrated (§8.2). It is now the **direct cause of one of the two remaining tone
   errors**: call_002 attributes 12.4s to a customer who speaks ~0.9s, so the classifier reads the
   bot's Spanish as the caller's.
2. **Attack politeness masking**, the other remaining tone error. call_003 is labelled `satisfied`
   while the caller is repeatedly blocked; the prompt carries an explicit rule for this and the
   model still reads the situation as frustration. Candidate signals: closing-turn sentiment
   weighted above mid-call content, and SER valence trend rather than call-level mean.
3. **Collect 100–300 labelled calls**, weighted toward the absent classes. Every remaining item is
   gated on this.
4. **Fit rather than reason** the activation weights and severity bands, and run a genuine
   stratified evaluation with per-class figures.
5. **A/B `gpt-4.1-mini` against `gpt-4o-mini`.** LLM cost is now a small fraction of the ceiling
   ($0.00025–$0.00093 per audio minute against $0.003), so the budget comfortably permits the
   stronger model — the constraint is evidence, not money. The OpenAI Batch API halves cost again
   if throughput allows.
6. **Find a real line-noise discriminator** for `background_noise_type`, which needs labelled
   examples of static, hum and crackle.
