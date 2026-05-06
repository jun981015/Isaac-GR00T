# Subgoal-Aware Progress Estimator

> Living doc. 큰 흐름: **Motivation → Concept → Related work → Dataset → Stage 1 (VLM) → Stage 2 (label) → Stage 3 (model) → Evaluation → Open questions**.

---

## 1. Motivation & contribution

### 1.1 Why progress estimator
Manipulation policy 학습/평가에 dense reward signal이 필요하다. Progress estimator는 영상에서 task 진행도 `P(t) ∈ [0, 1]`를 예측하는 모델이고, 이를 dense reward로 쓰면 sparse reward 대비 학습 효율이 올라간다.

### 1.2 Limitation of linear progress
기존 progress estimator (Robometer, GVL 등)는 progress를 **시간에 따라 균일하게 증가**하도록 모델링한다. 하지만 실제로 policy가 실패하는 지점은 특정 contact-bearing subtask (grasp, insert, place 등) 완료 순간에 집중되어 있다. Linear progress는 이런 critical moment에 다른 구간과 동일한 reward 변화를 할당하기 때문에 학습 신호가 약하다.

### 1.3 Hypothesis
> Progress 함수를 **critical 구간에서 더 가파르게 증가**하도록 non-linear하게 설계하면, 그 critical moment에 reward 변화가 집중되어 downstream RL policy가 그 지점을 더 잘 학습한다.

### 1.4 Three design knobs (contribution overview)

| 축 | 결정 | Default | Section |
|---|---|---|---|
| **Label** | slope ratio `r = w_crit / w_noncrit` 로 critical 구간 slope을 r배 가파르게. r=1이 linear baseline. | `r=3` | §6 |
| **Sampling** | per-video pair 추출 시 α 비율을 critical 안에 강제 sample, 나머지 (1-α)는 random uniform. **순서 무관 (t_i ⋛ t_j)** → forward/backward 양쪽 학습. | `α=0.5` | §7.2 |
| **Architecture** | pair input `(f_i, f_j) → Δp ∈ [-1, 1]` (음수 = regression). Full fine-tuning of `Qwen3-VL-4B`. | FFT | §7.1 |

세 축 모두 선행 연구에 없는 novel 결정 (§3 sampling survey).

---

## 2. Critical subtask — definition & scope

Critical = **policy가 자주 실패하는 어려운 subtask** (정밀 정렬이 필요한 contact 직전 fine-align + contact event를 한 구간으로 묶음). Stage 1 VLM이 task instruction + video로부터 추출 (§5).

| | 예시 |
|---|---|
| ✅ critical | pick/grasp, place/release, insert/peg-in-hole, stack-on, button-press, latch-release, plug-in, screw-tighten |
| ❌ non-critical | approach, transport(이미 잡은 채), hover, return-to-home, retract, gripper-open-prep |

**Scope**:
- **In-scope** (`r > 1` 의미 있음): pick-and-place, drawer/door, button, stack, insert 등 critical subtask가 명확한 task. **여기가 contribution.**
- Critical 0개 (pour, wipe, sweep, free-play 등): `r = 1` 강제 → linear baseline. degenerate case.
- Phase 1은 ≥1 critical 강제 (Stage 1 데이터 가공에서 보장, §4).

---

## 3. Related work — training-time frame sampling

각 reward model 논문이 학습할 때 video frame을 어떻게 뽑는지 (자세한 페이지 인용 + 비교: `docs/sampling_survey.md`).

| 논문 | T (frames) | 방식 | Variable-length | Segment-aware? | Loss reweight? |
|---|---|---|---|---|---|
| **Robometer** | max 32 | uniform-in-time downsample (cap) | max-cap uniform | ❌ | ❌ |
| **RoboReward** | 미명시 | VLM 기본 처리 위임 (label gen은 1 fps) | 미명시 | ❌ | ❌ |
| **Robo-Dopamine (GRM)** | variable (segment+chunk) | **keyframe-aware adaptive** + (BEFORE,AFTER) **frame pair** + ∆t stride sweep | segment당 균등 분배 | ✅ (유일) | ❌ |
| **SARM** | 9 (1 init + 8 cons.) | 30fps 원본에서 30-frame stride (1초 간격) | 고정 sliding window | ❌ | ❌ |
| **TOPReward** | K (가변) | uniformly-spaced prefix (zero-shot) | per-episode min-max | ❌ | — |
| **GVL** | 30 | uniform subsample to 30 + initial-anchor + shuffle (zero-shot) | 강제 30 | ❌ | — |

**Pattern**: Fixed T uniform-in-time이 dominant (Robometer 32, GVL 30, sequence input).
**Outlier**: GRM만 segment-aware + pair-input.

→ **우리 design 위치 (§1.4 표와 일치)**:
- pair-input (GRM에 가까움) + critical-aware sampling (어떤 논문도 안 함) + r-fixed non-linear label (어떤 논문도 안 함).
- Resolution은 Qwen3-VL-4B dynamic-resolution에 위임 + shortest-edge 256~336px cap (ablate, §7.1). Robometer 240px / SARM 224px와 같은 범위라 cross-baseline 비교에도 무리 없음.

---

## 4. Dataset

### 4.1 Reward-model dataset survey

| 논문 | 데이터셋 | 규모 | 라벨 | 공개 |
|---|---|---|---|---|
| Robometer (2026) | RBM-1M | 1M traj, 21 embodiments | per-frame 선형 | ✅ processed (5.87TB) |
| **RoboReward** | RoboReward-train + Bench | 45K + 6K + 2.8K | per-episode 1-5 (human) | ✅ `teetone/RoboReward` |
| Robo-Dopamine | GRM corpus | 35M samples | pair-wise relative | ⚠️ 모델만 |
| SARM | T-shirt folding | 700 | stage-aware dense | ⚠️ 미명시 |
| TOPReward | failure set | 156 | (zero-shot) | ❌ |
| GVL | (zero-shot) | — | — | — |

### 4.2 Why RoboReward
- **Raw video parquet 포맷** → frame sampling rate 자유 (RBM-1M은 32 frame 사전 양자화로 event 시점 손실).
- Per-episode 1-5 score → `reward=5` 필터로 success-only 깔끔히 추출.
- **RoboRewardBench (test split, 2,831 human-verified)** 으로 cross-dataset eval 자동 확보.
- RoboReward-4B/8B baseline 공개 → 직접 비교.

### 4.3 Filter pipeline & final manifest

```
Total: 54,135 episodes (train 45,072 / val 6,232 / test 2,831)
```

| Filter | 기준 | 제외 이유 |
|---|---|---|
| **F1** reward=5 | success-only (per-episode) | counterfactual / partial 제외 |
| **F2** reach-only 5 task | berkeley_mvp/tokyo_u_lsmo/utokyo_xarm/nyu_rot의 "Reach …" task | contact-event 없음, critical anchor 부재 |
| **F3** 반복/유체 7 keyword | `pour, stir, wipe, dust, sweep, rub, mix` | progress 단조 가정 위배, sustained-contact / cyclic motion |
| **F4** roboturk "Create tower" | 365 traj | VLM이 24~32 task로 잘게 쪼개고 tasks_time 절반만 채움 (회복 불가능) |
| **F5** N_tasks ∈ [2, 8] | VLM annotation 후 컷 | 너무 짧거나 길면 학습 노이즈 |

| Split | 원본 | F1 (=5) | F2 (reach) | F3 (kw) | F4 (tower) | VLM 통과 | F5 (N∈[2,8]) |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 45,072 | 8,425 | 8,329 | 7,536 | 7,171 | 7,084 | **6,887** |
| val   |  6,232 |   974 |   973 |   881 |   881 |   877 |   **859** |
| test  |  2,831 |   527 |   503 |   498 |   498 |   495 |   **482** |
| **합** | **54,135** | **9,926** | **9,805** | **8,915** | **8,550** | **8,456** | **8,228** |

→ **최종 학습/평가 manifest 8,228 traj** (자세한 step별 reject 분석은 `datasets/our_subset/annotation_filtering.md`).

**규모 충분 여부**: 27 source (OXE+RoboArena) → cross-embodiment 자연 확보. Task 다양성: pick-and-place / stack / insert / open-close / push / press / rotate / fold / handover 거의 전부 포함.

### 4.4 산출물 위치

```
datasets/our_subset/{train,val,test}/
├── metadata.jsonl              (8,550 — F1–F4 통과)
├── annotations/validated/      (8,456 — VLM Stage 1 통과, §5)
├── training_set.jsonl          (8,228 — F5 적용 학습 manifest)
└── summary.json                (per-source / excluded counts)
```

생성 스크립트:
- `scripts/filter_roboreward_success.py` — F1–F4
- `scripts/build_training_manifest.py` — F5

### 4.5 Evaluation env
- **Sim**: LIBERO (Franka pick-and-place suites) — online RL + offline metric
- **Real**: Franka real robot (2-3 pick-and-place task)
- **Cross-dataset offline**: RoboRewardBench test split (498 traj, 학습 미사용)

### 4.6 Paper framing
> "RoboReward train의 perfect-success episodes (8.2K traj, 27 sources) 에서 **slope-ratio r-fixed non-linear progress label + critical-stratified pair sampling** 으로 fine-tune. r은 critical/non-critical slope 비율 hyperparameter, α는 critical-anchored pair 비율. RoboRewardBench cross-dataset + LIBERO sim + Franka real 양쪽에서 linear baseline (r=1, α=0) 대비 우위."

---

## 5. Stage 1 — VLM annotation

### 5.1 Pipeline overview

```
metadata.jsonl  (F1–F4 통과, 8,550)
  │
  ▼  Stage 1 — VLM annotator (per-trajectory, 1 call + retry up to 3)
     in:  task_instruction + video frames (5 FPS uniform) + duration_sec
     out: JSON {reasoning, tasks, critical_tasks, tasks_time}
     →   raw/{traj_id}/attempt_{n}.json + validated/ + rejected/
  │
  ▼  Stage 2 — Per-frame progress label generator (deterministic, §6)
     in:  validated tasks_time + r ∈ {1, 2, 3, 5, 10} + frame_timestamps
     out: event_labels/{traj_id}.json (r별 progress + critical_mask)
```

### 5.2 Frame sampling (Stage 1)
- **5 FPS uniform**: `frame_timestamps = [k / 5.0 for k in range(int(duration_sec * 5))]`.
- 5초 영상 → 25 frame, 10초 → 50, 30초 → 150.
- Critical event (보통 1~3초) 안에 5~15 frame 들어가서 contact onset + fine-align 양자화 충분.
- max_frames cap 미설정.

### 5.3 Output schema (4 fields)

vLLM `response_format={"type":"json_schema","strict":true}` 로 schema 강제.

```json
{
  "reasoning": "The video shows a robot manipulating a blue block. Approach (0-3s) is easy free-space. Picking up (3-4s) is critical—gripper has to fine-align then close. Transport (4-8s) is easy. Placing on yellow plate (8-10s) is critical. So critical_tasks = [pick, place].",
  "tasks": ["approach the blue block", "pick the blue block", "transport to yellow plate", "place on yellow plate"],
  "critical_tasks": ["pick the blue block", "place on yellow plate"],
  "tasks_time": [
    {"task": "approach the blue block",   "start_sec": 0.0, "end_sec": 3.0},
    {"task": "pick the blue block",       "start_sec": 3.0, "end_sec": 4.0},
    {"task": "transport to yellow plate", "start_sec": 4.0, "end_sec": 8.0},
    {"task": "place on yellow plate",     "start_sec": 8.0, "end_sec": 10.0}
  ]
}
```

| 필드 | 목적 |
|---|---|
| `reasoning` | chain-of-thought. critical 판단 + 시간 구간 추론. 가장 앞에 둬서 후속 구조화 필드 정확도 ↑. Training 미사용. |
| `tasks` | 시간순 전체 subtask 이름 (critical/non-crit 구분 없음) |
| `critical_tasks` | `tasks`의 부분집합. **strict string match**으로 1:1 대응 |
| `tasks_time` | 모든 task의 시간 구간. `tasks`와 동일 순서. 인접·비중복 (`end_i == start_{i+1}`), 첫 항목 `start=0`, 마지막 `end=duration_sec` |

**Critical 판단 기준** (system prompt v3 = `prompts/stage1_system.txt`): §2 ✅/❌ 카테고리 그대로. 구간 길이 가이드: critical 보통 0.5~3초, 5초+ 단일 critical은 free-space approach 섞였을 가능성 → 경계 뒤로 미룸.

### 5.4 Validation

JSON 파싱 후 다음 모두 통과 시 `validated/{traj_id}.json` 으로 저장:

- 4 필드 모두 존재
- `len(tasks) ≥ 1`, `len(critical_tasks) ≥ 1`
- `len(tasks_time) == len(tasks)` 동일 순서 1:1 매칭 (strict string match)
- `critical_tasks ⊆ tasks` (strict)
- `tasks_time[0].start_sec ≈ 0.0`, `tasks_time[-1].end_sec ≈ duration_sec` (ε=0.1s)
- 인접·비중복 (ε=0.05s): 모든 i, `tasks_time[i].end_sec ≈ tasks_time[i+1].start_sec`
- 각 segment: `0 ≤ start_sec < end_sec ≤ duration_sec`

### 5.5 Retry & output structure

매 traj 최대 N=3회 시도. 각 시도는 `raw/{traj_id}/attempt_{n}.json` 으로 별도 저장 (validation 무관).

```
datasets/our_subset/{split}/annotations/
├── raw/{traj_id}/attempt_{1..3}.json   # VLM 응답 보존
├── validated/{traj_id}.json             # 통과 (학습 사용)
├── rejected/{traj_id}.json              # 3회 실패 (reason + history)
└── summary.csv                          # traj_id, status, n_attempts, ...
```

**reject_reason enum** (간단 요약): `vlm_timeout`, `json_parse_error`, `missing_field`, `empty_tasks`, `empty_critical`, `tasks_time_count_mismatch`, `tasks_time_name_mismatch`, `critical_not_in_tasks`, `time_invalid_range`, `tasks_not_contiguous`, `tasks_coverage_incomplete`. 자세한 기준은 §5.4 validation 규칙과 1:1 대응.

### 5.6 Production VLM
- **Model**: **Qwen3.6-27B** (multimodal). 4× B200 (sub-extra QoS), TP=4, vLLM 0.19.1.
- **Smoke test only**: Qwen3.5-9B (1 GPU, 5-traj 파이프라인 검증용).
- **JSON 강제**: OpenAI 표준 `response_format={"type":"json_schema","strict":true}` (옛 vLLM 전용 `extra_body={"guided_json":...}`은 0.19.1에서 무시됨).
- **2-pass annotation**:
  - **1차** (88.2% validated): `image=300` + `max-model-len=32768` + `max_tokens=4096`
  - **2차 retry** (96.4% validated): `image=600` + `max-model-len=65536` + `max_pixels=200704` + `max_tokens=8192`
- 최종 8,456 / 8,550 = **98.9% validated** (94 rejected).

---

## 6. Stage 2 — Progress label generation (r-fixed)

§5에서 추출한 `tasks_time` + `critical_tasks` 를 받아서 **deterministic** 하게 per-frame progress array를 생성한다.

### 6.1 r-fixed formula

**Hyperparameter**: slope ratio `r = slope_critical / slope_noncritical ∈ {1, 2, 3, 5, 10}`. `r=1`이 linear baseline.

**계산** (`Σ Δ = 1` 제약 + `w_c = r·w_n` 으로 유일해):

```
L_crit    = Σ (end - start) over critical segments
L_noncrit = Σ (end - start) over non-critical segments

w_n = 1 / (r · L_crit + L_noncrit)
w_c = r · w_n

Δ_i  = slope_i · (end_i - start_i)        where slope_i ∈ {w_c, w_n}
P(t) = cumulative_i + (t - start_i) · slope_i    (segment i 내부)
P(0)=0, P(duration_sec)=1                  (postcondition)
```

**Edge cases**:
- `L_noncrit = 0` (모두 critical): `w_c = 1/L_crit`, `w_n = 0`. r 무시 → 사실상 linear.
- `L_crit = 0` (Phase 1 미발생, 안전망): `w_n = 1/L_noncrit`, `w_c = 0`.

**보장되는 성질** (segment 길이 분포 무관):
- 모든 critical slope 동일 = `w_c`. 모든 non-critical slope 동일 = `w_n`. **`w_c / w_n ≡ r`** 항상.
- `r > 1` ⇒ critical이 항상 r배 가파름. 우연히 긴 critical도 non-critical보다 완만해지지 **않음** (이전 j-fixed 안의 결함을 제거).
- `r = 1` ⇒ `w_c = w_n = 1/duration` ⇒ `P(t) = t/duration` 자동 환원 (별도 baseline 분기 불필요).

### 6.2 Concrete example (pick-and-place, r=4)

`approach 0~3s, pick 3~4s (crit), transport 4~8s, place 8~10s (crit)` → `ΣL_crit=3s, ΣL_noncrit=7s`.
`w_n = 1/(4·3 + 7) = 1/19 ≈ 0.0526/s`,  `w_c = 4·w_n ≈ 0.2105/s`.

| 구간 | 시간 | 종류 | slope | Δ | cumulative |
|---|---|---|---|---|---|
| approach | 0~3s | non-crit | 0.0526/s (`w_n`) | 0.158 | → 0.158 |
| **pick** | 3~4s | **crit** | **0.2105/s (`w_c`)** | 0.211 | → 0.368 |
| transport | 4~8s | non-crit | 0.0526/s | 0.211 | → 0.579 |
| **place** | 8~10s | **crit** | **0.2105/s** | 0.421 | → 1.000 ✓ |

길이가 긴 place(2s)가 짧은 pick(1s)보다 더 많은 progress(0.421 vs 0.211)를 차지하지만 **slope은 동일** = `w_c`. 짧으면 짧지만 가파르게 jump, 길면 같은 가파른 slope으로 progress가 더 길게 쌓임.

```
progress
  1.0 ┤                                                  ╱  ← place: slope 0.21 (w_c)
      │                                                ╱
  0.58┤                                  ───────────
      │                              ───            ← transport: 0.05 (w_n)
  0.37┤                       ╱
      │                     ╱                       ← pick: 0.21 (w_c)
  0.16┤              ───
      │      ────────                               ← approach: 0.05 (w_n)
  0.0 ┤────
      └────────┬──┬──────────────┬─────────
       approach pick  transport  place
       0~3s    3~4s  4~8s        8~10s
```

### 6.3 Speed-invariance corollary

같은 trajectory를 2배 느리게 찍은 영상 B (duration 2배) 에서 같은 fraction `τ ∈ [0,1]` 의 frame은 같은 P 값을 가진다.

증명: 모든 segment 길이가 2배 → `w_*_B = w_*_A / 2`. 그러나 길이도 2배라 `Δ_seg_B = (w_*_A/2) · (2L_seg_A) = Δ_seg_A`. 즉 **P는 fraction의 함수, 절대 시간의 함수가 아님**.

→ Pair label 예 (r=4): A의 (2s, 5s) [fraction 0.2→0.5] = B의 (4s, 10s) [같은 fraction] = `0.316` 동일.

**Sampling 함의** (§7.2 default가 이 성질을 자연 만족):
- ✅ `t_j ~ Uniform(0, duration)` → τ_j 균등 → 영상 길이 무관 학습 분포 일관.
- ❌ 절대 시간 gap (`t_j = t_i + 3s` 고정) → 영상 길이에 따라 fraction 분포 달라짐. ∆t sweep이 필요하면 **fraction 단위 ∆τ** 로 정의해야 안전.

### 6.4 Generator code

```python
def generate_progress(tasks_time, critical_tasks, duration_sec, r, frame_timestamps):
    """r-fixed piecewise-linear progress label generator."""
    crit_set = set(critical_tasks)
    subtasks = [{**seg, "critical": seg["task"] in crit_set} for seg in tasks_time]

    L_crit    = sum(s["end_sec"] - s["start_sec"] for s in subtasks if s["critical"])
    L_noncrit = sum(s["end_sec"] - s["start_sec"] for s in subtasks if not s["critical"])

    if L_noncrit == 0:                # 모두 critical
        w_c, w_n = 1.0 / L_crit, 0.0
    elif L_crit == 0:                 # Phase 1 미발생
        w_c, w_n = 0.0, 1.0 / L_noncrit
    else:
        w_n = 1.0 / (r * L_crit + L_noncrit)
        w_c = r * w_n

    cumulative, pieces = 0.0, []
    for s in subtasks:
        slope = w_c if s["critical"] else w_n
        pieces.append({"start": s["start_sec"], "end": s["end_sec"],
                       "start_p": cumulative, "slope": slope})
        cumulative += slope * (s["end_sec"] - s["start_sec"])
    # postcondition: cumulative ≈ 1.0

    progress = []
    for t in frame_timestamps:
        if t <= 0.0: progress.append(0.0); continue
        if t >= duration_sec: progress.append(1.0); continue
        for p in pieces:
            if p["start"] <= t <= p["end"]:
                progress.append(p["start_p"] + (t - p["start"]) * p["slope"])
                break
    return progress
```

각 `r ∈ {1, 2, 3, 5, 10}` 마다 호출해서 sidecar에 저장 (`r=1` 자동 linear).

### 6.5 Sidecar format

`datasets/our_subset/{split}/event_labels/{traj_id}.json`:

```jsonc
{
  "traj_id": "bridge_originalsplit_train_index_42",
  "split": "train",
  "duration_sec": 10.0,
  "frame_timestamps": [0.0, 0.2, ..., 9.8],   // 5 fps
  "tasks": [...], "critical_tasks": [...], "tasks_time": [...],
  "N_total": 4, "N_critical": 2,
  "L_crit_sec": 3.0, "L_noncrit_sec": 7.0,
  "progress_r1":  [0.0, ..., 1.0],   // linear baseline (auto from r=1)
  "progress_r2":  [...],
  "progress_r3":  [...],
  "progress_r5":  [...],
  "progress_r10": [...],
  "critical_mask": [0, 0, ..., 1, 1, 0, ..., 1, 1, 1]   // per-frame {0,1}
}
```

각 progress array의 첫 값=0.0, 마지막 값=1.0 (postcondition 검증). `critical_mask[k] = 1 iff t_k ∈ critical segment` — 학습 단계 **critical loss reweight ablation** 용 (§7.3).

---

## 7. Stage 3 — Model training

### 7.1 Architecture

| 항목 | 결정 |
|---|---|
| **Backbone** | `Qwen/Qwen3-VL-4B-Instruct`, **full fine-tuning (no LoRA)** |
| **Input** | pair `(frame_i, frame_j)` 두 RGB. timestamp 모델에 미입력 (visual only). **순서 제약 없음** (`t_i ⋛ t_j` 모두 허용) |
| **Output** | scalar `Δp = P(t_j) - P(t_i) ∈ [-1, 1]` |
| **Frame resolution** | RoboReward 원본 raw video (소스마다 다양 — bridge 320×240, droid 1280×720 등). **Qwen3-VL-4B의 dynamic-resolution 처리에 위임** + 학습 throughput을 위해 **shortest-edge 256~336px 범위에서 cap**. 정확한 값은 ablate (240/256/336 후보). RBM-1M 240px 호환은 "있으면 좋음" 정도이고 결정 요인은 아님. |
| **Frame extraction** | **on-demand decoding** — `t_i, t_j ~ Uniform(0, duration)` 연속값으로 sample 후 ffmpeg seek으로 mp4에서 직접 디코딩. 사전 grid 양자화 없음 → critical boundary 와 정확히 일치하는 pair 가능. 학습 throughput 문제 생기면 Phase 1.5에서 5 FPS grid로 전환 (Stage 1 annotation 그리드 reuse). |
| **Frame budget per video** | 한 forward pass = **2 frame**. 한 epoch 당 video 1편에서 `K=16` pair × 2 = **32 frame** (sampled 새로 매 epoch). Pre-extracted pool 없음. |

**왜 [-1, 1] output (= 왜 순서 제약 없음)**:
- 학습 시 pair `(t_i, t_j)` 를 **순서 무관**으로 sample. `t_i > t_j` 인 pair도 들어옴 → 라벨 `Δp = P(t_j) - P(t_i) < 0`. 자연스럽게 음수 Δp 학습.
- **Online RL inference에서 regression 신호로 활용**: policy가 잘못된 행동으로 progress가 뒤로 가는 상황 (예: 잡았던 물체를 떨어뜨림, transport 중 후진)에서, 모델이 `(f_{t-1}, f_t)` 의 visual 변화로부터 음수 Δp를 출력 → reward 감소. 즉 **forward step 보상 + backward step 패널티**가 한 모델로 통합.
- Phase 1 success trajectory만 사용하더라도 trajectory 내부에서 forward/backward pair 둘 다 만들 수 있으므로 학습 분포는 양쪽 다 포함.

**왜 full fine-tuning over LoRA**:
- Contribution 측정에 head + visual encoder 양쪽 모두 적응 필요. r-fixed label과 critical-stratified sampling이 visual representation에 영향 → LoRA로는 부족할 수 있음.
- 4B 모델 + B200 4× TP면 FFT가 GPU memory 측면에서 충분.
- LoRA는 low-priority fallback ablation row.

**Inference 사용 패턴**:
- **Online RL** (예: LIBERO): 매 step `t`에서 `(f_{t-1}, f_t)` forward → 그 step의 instantaneous Δp가 reward (음수면 regression 패널티). 또는 `(f_0, f_t)` forward → 누적 progress 자체.
- **Offline eval**: 임의 pair `(f_i, f_j)` 로 Δp 예측 정확도 평가.

### 7.2 Sampling — α-mixed pair

각 video에서 **K pair**를 다음 비율로 추출. 모든 pair는 **순서 무관** (`t_i ⋛ t_j` 모두 허용) → label `Δp = P(t_j) - P(t_i)` 는 양수 / 음수 / 0 모두 가능.

| 비율 | 종류 | 정의 |
|---|---|---|
| **α** | **Critical-spanning** | pair `(t_i, t_j)` 의 시간 구간 `[min(t_i,t_j), max(t_i,t_j)]` 가 어떤 critical segment `[c_s, c_e]` 와 겹침: `min(t_i,t_j) < c_e` AND `max(t_i,t_j) > c_s`. 즉 segment 안에 들어가거나 boundary를 가로지름. 두 frame 중 어느 것이 critical 안인지는 무관. |
| **1−α** | **Random uniform** | `t_i, t_j ~ Uniform(0, duration)` 독립. 순서·critical 무관. |

**Default**: `K=16`, `α=0.5`. **Ablation**: `α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}` (α=0이 random uniform baseline, α=1이 extreme).

**왜 이 sampling이 contribution과 fit**:
- Label은 critical 구간에 progress mass를 몰아주지만 (slope `w_c = r·w_n`), random uniform sampling은 critical 구간을 **시간 비율 (`ΣL_crit / duration`) 만큼만** 가로지름. 즉 progress 변화가 큰 영역과 학습 분포의 비중이 어긋남. α-가중으로 **label의 critical-mass × sampling의 critical-coverage** 곱셈 효과를 줘서 contribution 신호 강화.
- 순서 무관 sampling이라 forward / backward pair 모두 학습 분포에 자연 포함 → §7.1 음수 Δp 학습 자동 만족.
- §6.3 speed-invariance: `t_j ~ Uniform(0, duration)` 이라 fraction 균등 → 영상 길이 무관 학습 분포 일관.

**Ablation 분리**: `r` (label) × `α` (sampling) 가 직교 → 둘의 효과를 따로 측정 가능.

### 7.3 Loss

| Loss | 정의 | 비고 |
|---|---|---|
| **MSE (default)** | `(pred − label)²` | 단순, baseline |
| **C51 (ablation)** | `[-1, 1]` 을 20-bin discretize, soft 1-hot CE | multi-modal uncertainty 표현, Robometer 호환 |
| **Critical reweight (ablation)** | `w_loss = w_crit if frame ∈ critical else 1.0`, `w_crit ∈ {2, 3, 5}` | sampling 그대로 두고 loss 단계에서 critical 강조. Robometer는 균일이 최적이라 ablate했었지만 우리는 frame-level reweight라 다른 축. |
| **Auxiliary (Phase 2)** | event classification (`critical_mask` binary) + ranking (`P(t_j) > P(t_i) for t_j > t_i`) | monotonicity 보강 |

### 7.4 Ablation rows

| 축 | 후보 | Default |
|---|---|---|
| Label `r` (slope ratio) | `{1, 2, 3, 5, 10}` | r=3 |
| Sampling `α` (critical-stratified 비율) | `{0.0, 0.25, 0.5, 0.75, 1.0}` | α=0.5 |
| Loss | MSE / C51-20bin | MSE |
| Critical loss reweight | none / `w_loss ∈ {2, 3, 5}` | none |
| Segment 내 분배 | linear / sigmoid | linear |
| Backbone | `base Qwen3-VL-4B` (FFT) / `Robometer-4B` warm-start (confounded) | base FFT |
| Adapter | FFT / LoRA-r32 (low-priority) | FFT |

대표 row 이름: `base-r{r}-α{αα}-{loss}-fft` (e.g., `base-r3-α05-mse-fft`).

핵심 비교:
- **Linear baseline**: `base-r1-α00-mse-fft`
- **Ours main (default)**: `base-r3-α05-mse-fft`

### 7.5 Fallback (학습 안 될 시)
- Preference head 추가해 multi-task (Robometer `L_prog + L_pref + L_succ`)
- C51 bin 수 증가 (20 → 50)
- Auxiliary loss (event classification, ranking) 적극 활용
- Backbone을 Robometer-4B warm-start로 변경 (confounded이지만 학습 수렴 가속)

---

## 8. Evaluation

### 8.1 Reward model quality (offline)
- **MAE / MSE** on held-out progress label (RoboReward val + 우리 라벨)
- **VOC** — GVL/Robometer 공통 metric
- **Kendall τ** — trajectory ranking accuracy
- **Critical event detection accuracy** — 예측 reward curve의 sharp jump frame이 GT critical event 근처 (±3 frame)
- **Jump sharpness metric** (proposed) — critical event 전후 progress 차이

### 8.2 Cross-dataset offline (RoboRewardBench)
- RoboReward test split (498 traj, 학습 미사용)
- 우리 모델 vs Robometer-4B / RoboReward-4B / Gemini zero-shot

### 8.3 Downstream RL (online)
- **Sim**: LIBERO (Franka pick-and-place suites)
- **Real**: Franka 2-3 pick-and-place tasks
- 비교: sparse / linear progress / ours
- **핵심 metric**: pick 단계 실패율 감소

### 8.4 Generalization (Phase 2)
- Non-prehensile task (drawer open 등)에서 동일 실험

---

## 9. Open questions / Next actions

### 즉시 (To-do)
- [ ] **Stage 2 generator 구현** — `scripts/generate_progress_labels.py` (§6.4 코드대로, `r ∈ {1, 2, 3, 5, 10}` + `critical_mask`). sidecar `event_labels/{traj_id}.json` 생성.
- [ ] Synthetic learnability test (pair model이 8-frame step function을 학습 가능한가)
- [ ] Critical 정확도 human spot-check (50 sample, viz_server로 검수 — `scripts/viz_server.py` 사용)
- [ ] Stage 3 학습 코드 (pair sampler `α` configurable + FFT loop)
- [ ] 영상 길이 / critical 길이 분포 분석 (sampling K 결정에 활용)

### Done
- [x] `teetone/RoboReward` 다운로드
- [x] F1–F5 적용 → `our_subset/{train,val,test}/training_set.jsonl` (8,228 traj)
- [x] Stage 1 system prompt v3 (`prompts/stage1_system.txt`)
- [x] Qwen3.6-27B vLLM 서빙 sbatch + OpenAI-호환 client (`scripts/vlm_*`)
- [x] 전체 our_subset annotation 완료 (8,456 validated / 94 rejected, 98.9%)
- [x] N_tasks ∈ [2, 8] 필터 → `training_set.jsonl` (8,228)

### 설계 결정 (확정)
- [x] Dataset = RoboReward train F1–F4 (8,550)
- [x] Final manifest = `training_set.jsonl` with N∈[2,8] (8,228)
- [x] Labeling = 2-stage VLM (Stage 1 4-field schema) + deterministic Stage 2
- [x] Critical 정의 = "policy가 자주 실패하는 어려운 subtask" (난이도 기준), contact 직전 fine-align + onset 한 구간
- [x] ≥1 critical 강제
- [x] Frame sampling (Stage 1) = 5 FPS uniform, max_frames cap 없음
- [x] VLM = Qwen3.6-27B (4× B200 TP=4 sub-extra), `response_format=json_schema strict`, retry up to 3
- [x] **Label hyperparameter = `r ∈ {1, 2, 3, 5, 10}`** (changelog 2026-04-28)
- [x] **Architecture = pair input, output ∈ [-1, 1], full fine-tuning**. 순서 무관 sampling (`t_i ⋛ t_j` 모두) 으로 음수 Δp 학습 → online RL에서 backward step에 자연 패널티.
- [x] **Sampling = α-mixed critical-spanning + random uniform pair** (`α ∈ {0, 0.25, 0.5, 0.75, 1.0}`, default 0.5). 순서 무관.
- [x] **Loss = MSE default + C51-20bin ablation**
- [x] Backbone = base Qwen3-VL-4B FFT (primary), Robometer-4B warm-start (confounded ablation)

### 검증 (파일럿 종료 조건)
- [ ] VLM이 critical subtask를 안정적으로 추출하는가? (50 sample human spot-check)
- [ ] critical event 시간 구간 boundary 정확도 ±0.4s 80% 이상
- [ ] Fine-tune 후 critical event 근처 reward curve sharp 한가?
- [ ] Linear baseline 대비 pick 단계 실패율 의미 있게 감소?

### 미결 조사
- [ ] Critical subtask GT 검증 데이터 어디서? (AGIBot은 our_subset에 없음)
- [ ] kaist_nonprehensile (push/topple/rotate/flip) 가장자리 케이스 critical 라벨 정확도
- [x] Frame 수 분포 (5 FPS 기준 max ~600 frame → 64K context + max_pixels 200704로 해결)

---

## §A.1 Failure trajectory — future work (Phase 2)

Phase 1은 **success trajectory만** 사용. (단 음수 Δp는 Phase 1에서도 forward/backward pair sampling 으로 이미 학습됨 — §7.1, §7.2). Phase 2에서 실패 trajectory 자체를 라벨링하는 옵션:

| 옵션 | 설명 |
|---|---|
| B: cap at pre-event level | 실패 시 해당 critical event 직전 progress로 유지 |
| C: jump + regression | Robometer 스타일. 잠깐 jump 후 원복 (monotonicity 깨짐) |
| D: explicit negative-progress region | 실패 traj에서 critical event 후 P(t)가 감소하는 구간을 라벨로 부여. Δp ∈ [-1, 1] output range가 이를 자연 지원. |

---

## Changelog

- 2026-04-28 (later): **doc 전면 재구조화 + design 명확화**. (1) 섹션 순서를 motivation → concept → related work → dataset → Stage 1 → Stage 2 → Stage 3 → eval → open으로 재배치 (이전엔 Research goal 안에 r 수식 + concrete example + speed-invariance가 다 들어가 있어 motivation/detail 구분이 흐릿함). (2) **§1.4 contribution overview 표** 신설 — label/sampling/architecture 세 축 명시. (3) **§7.1 architecture 명확화**: pair input + **output `Δp ∈ [-1, 1]`** + **full fine-tuning (LoRA 사용 안 함)**. 이전 doc은 LoRA-r32 default로 적혀 있었으나 contribution이 visual encoder까지 영향을 주는 점 + 4B 모델이라 FFT가 가능한 점을 고려해 FFT default로 변경. LoRA는 low-priority ablation row로 강등. (4) **§7.2 sampling 정의 명세화**: 기존엔 "anchor=0 + critical-stratified t_j" 로 모호하게 적혀 있었음. 새 정의 = pair `(t_i, t_j)` 일반 sampling + α 비율은 critical-spanning (segment overlap) + (1−α) 비율은 random uniform. critical-spanning 조건 = `t_i < c_e AND t_j > c_s`. (5) **§7.3 loss table 정리**: MSE default, C51 ablation, critical reweight ablation, auxiliary (Phase 2) 분리. (6) §2.3 (이전) → §3 (현재 related work). §1 progress profile + concrete example + speed-invariance → §6 Stage 2 안으로 이동. §1.1 failure → §A.1 future work.
- 2026-04-28: **Progress profile 재정의 — j-fixed (count-based) → r-fixed (slope-ratio).** 이전 안은 모든 critical에 동일한 progress 양 `j`를 할당하고 segment 내부에서 시간 균등 분배했음. 이 경우 segment 길이가 우연히 critical은 길고 non-critical은 짧으면 `slope_critical < slope_noncritical` 이 되어 "critical = sharper jump" narrative가 깨짐. 새 정의: hyperparameter를 slope ratio `r = w_c / w_n` 으로 두고 `Σ Δ = 1` 제약 + `w_c = r·w_n` 으로 풀어냄. 보장: (a) 모든 critical slope = w_c, 모든 non-critical = w_n, (b) `r > 1` 이면 항상 critical이 r배 가팔라짐 (길이 분포 무관), (c) `r = 1` 이면 자동 linear (별도 baseline 분기 불필요). 후보 `r ∈ {1, 2, 3, 5, 10}`. r 제약 없음 (이전 `j ≤ 1/N_critical` skip 룰 폐기).
- 2026-04-27: **Stage 1 VLM annotation 실행 + production manifest 확정.** Qwen3.6-27B (smoke test만 Qwen3.5-9B). vLLM 0.19.1 호환 이슈 해결 (libstdcxx, FlashInfer cu13, Triton JIT). JSON 강제 = OpenAI 표준 `response_format=json_schema strict` (옛 `extra_body=guided_json`은 무시됨). 2-pass annotation: 1차 88.2% → 2차 retry (image=600, max-model-len=65536, max_pixels=200704) 96.4% → 최종 98.9% validated. roboturk "Create tower" 365 traj 제외 (잘게 쪼개고 tasks_time 절반만 채움, 회복 불가). N_tasks ∈ [2, 8] 추가 컷 — multi-step traj (austin_sirius, cmu_play_fusion 등)가 7~8 task로 자연 분해되며 N_critical=4 발생 → 보존. 최종 manifest 8,228.
- 2026-04-26: **반복 모션 / 유체 흐름 task 추가 제외** (`pour, stir, wipe, dust, sweep, rub, mix` 7 keyword). 이유: progress 단조 가정 위배 + critical=fine-align+contact-onset 정의가 cyclic motion에 안 맞음. 살린 경계 케이스: `fold` (20), `unfold` (36), `straighten` (2) — 단일 grasp 동작.
- 2026-04-26: **Stage 1 schema 단순화 + reach-only 제외**. (1) Critical 정의 = "policy가 자주 실패하는 어려운 subtask" (난이도 기준), contact 직전 fine-align + onset 한 구간. (2) JSON 4-field: `reasoning + tasks + critical_tasks + tasks_time`. (3) Frame sampling = 5 FPS uniform (cap 없음). (4) Retry up to N=3, raw/{traj_id}/attempt_{n}.json. (5) 디렉토리 = `annotations/{raw,validated,rejected}/`. (6) reach-only 5 task 121 traj 제외 (berkeley_mvp, tokyo_u_lsmo, utokyo_xarm, nyu_rot의 "Reach …" 류).
- 2026-04-22: **Critical 시간 단위 segment(start_sec, end_sec)로 재정의** (frame index → 초). Stage 2 generator는 frame_timestamps에서 piecewise-linear evaluate, 32-frame이든 64-frame이든 같은 시간점에서 같은 progress.
- 2026-04-22: **Critical을 단일 frame이 아닌 frame 구간으로 재정의**. progress jump가 instantaneous spike가 아니라 critical 구간 전체에 걸쳐 linear하게 분배 (구간 내 기울기로 critical 표현).
- 2026-04-22: **데이터셋 RBM-1M → RoboReward**. (1) raw video parquet 포맷 → frame sampling 자유. (2) per-episode 1-5 score → success-only 필터 깔끔. (3) RoboRewardBench cross-dataset eval 자동 확보. Critical subtask 개념 도입.
- 2026-04-21: 최종 결정 — base Qwen3-VL-4B primary, LIBERO+Franka eval.
- 2026-04-20: 초안 — dataset survey, scope 정의, label design 결정 (jump magnitude, equal magnitude, failure 제외).
