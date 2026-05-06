# RoboCasa ZMQ Parallel Eval Handoff

Date: 2026-04-29

## 목적

기존 RoboCasa eval은 환경 step 시간이 커서 GPU utilization이 안정적으로 100% 유지되지 않았다.
이번 작업의 목표는 GR00T-1.6의 RoboCasa 병렬 eval 방식을 참고해서, 여러 env가 동시에 rollout을 진행하고 policy query는 batch 단위로 묶어 보내는 ZMQ 기반 eval 경로를 만드는 것이었다.

추가로 기존 영상 저장 경로는 policy obs 128 이미지를 512로 단순 upscale하는 구조였기 때문에 실제 정보량이 128에 머물렀다.
이를 개선해서 env camera는 256으로 만들고, 같은 obs를 policy 입력 직전에 128로 resize하며, video는 256 obs를 그대로 저장하는 구조로 바꿨다.
즉 128/256을 따로 render하지 않고 256 obs 하나만 생성한다.

## 병합 상태

실험용 worktree:

- `/home/junhyeong/Value/Isaac-GR00T-eval-zmq`

병합 대상 repo:

- `/home/junhyeong/Value/Isaac-GR00T`

병합 완료된 파일:

- `gr00t/eval/robot.py`
- `gr00t/eval/service.py`
- `gr00t/eval/wrappers/robocasa_n15_wrapper.py`
- `scripts/robocasa_n15_zmq_eval.py`
- `scripts/robocasa_n15_zmq_parallel_eval.py`
- `scripts/run_groot_robocasa_zmq_server.sh`
- `scripts/run_robocasa_n15_zmq_eval_container.sh`
- `scripts/run_robocasa_n15_zmq_parallel_eval_container.sh`

원본 repo에 기존부터 있던 다른 수정사항은 건드리지 않았다.
예를 들어 `gr00t/eval/http_server.py`, `scripts/robocasa_n15_http_eval.py`, `scripts/run_robocasa_n15_eval_container.sh`, handoff 문서 등은 이번 병합 범위 밖이다.

## 구현 요약

### ZMQ server/client

`gr00t/eval/robot.py`와 `gr00t/eval/service.py`를 수정해서 RoboCasa eval client가 가볍게 ZMQ policy server를 호출할 수 있게 했다.

핵심 변경:

- `RobotInferenceClient`가 eval client에서 무거운 policy/data import를 강제하지 않도록 정리했다.
- ZMQ request timeout, socket linger, socket 재생성 처리를 추가했다.
- seeded action endpoint를 추가했다.
  - endpoint 이름: `get_action_seeded`
  - 요청 payload에 `observation`과 `action_seed`를 넣는다.
  - server 쪽에서 `random`, `numpy`, `torch`, `torch.cuda` seed를 설정한 뒤 policy action을 계산한다.

주의:

- ZMQ REP server는 요청을 순차 처리하므로 HTTP 기반 global RNG race 문제는 줄었다.
- 다만 batch 내부 sample별로 완전히 독립적인 torch generator를 쓰는 것은 아니다.
  현재 구현은 batch composition과 call order가 deterministic할 때 batch 단위 deterministic을 기대하는 구조다.

### RoboCasa obs resize

`gr00t/eval/wrappers/robocasa_n15_wrapper.py`에 image resize 옵션을 추가했다.

목표 구조:

- env camera: 256x256
- video 저장: 256x256 camera obs를 3개 concat해서 768x256 composite
- policy 입력: 같은 256 obs를 policy 직전에 128x128로 resize
- 별도 128 camera render 없음
- 별도 `env.sim.render()` 영상 render 없음, `VIDEO_SOURCE=obs`

### ZMQ single eval

`scripts/robocasa_n15_zmq_eval.py`는 단일 env용 ZMQ eval script다.
기존 HTTP eval과 유사하게 schedule JSON을 읽고, episode별 결과 JSON과 summary를 저장한다.

기본값:

- `camera_width=256`
- `camera_height=256`
- `policy_image_size=128`
- `video_render_size=256`
- `video_source=obs`

### ZMQ parallel eval

`scripts/robocasa_n15_zmq_parallel_eval.py`는 여러 env worker를 띄우고 main process가 policy batch를 구성하는 parallel eval script다.

구조:

- worker process마다 RoboCasa env 1개를 가진다.
- worker는 reset/step/video write를 담당한다.
- main process는 worker가 보낸 policy observation들을 모아서 ZMQ server에 batch request를 보낸다.
- `policy_batch_mode=lockstep` 기본값을 사용하면 active env들이 모두 action을 요청할 때까지 기다렸다가 batch로 한 번에 policy를 호출한다.
- `coalesce` mode도 있으나 smoke에서 batch size가 1로 흐르는 경우가 있어, 현재 목표에는 `lockstep`이 더 맞다.

재현성 의도:

- scene은 schedule JSON의 `seed`와 `ep_meta`로 고정한다.
- weight별 동일 환경 비교는 같은 schedule JSON을 사용하면 된다.
- action sampling도 `get_action_seeded`로 batch call seed를 통제한다.
- 정확한 per-episode action noise 독립성을 더 엄밀히 보장하려면 추후 action head 내부 generator 단위 제어가 필요할 수 있다.

## Launcher 기본값

새 launcher들은 기본적으로 GPU 3을 사용하도록 설정했다.
사용자가 GPU 0을 쓰지 말라고 했기 때문이다.

기본값:

- `GPU_DEVICE=3`
- `CAMERA_WIDTH=256`
- `CAMERA_HEIGHT=256`
- `POLICY_IMAGE_SIZE=128`
- `VIDEO_RENDER_SIZE=256`
- `VIDEO_SOURCE=obs`
- `WRITE_VIDEO=1`
- `STREAM_VIDEO=1`

관련 launcher:

- `scripts/run_groot_robocasa_zmq_server.sh`
- `scripts/run_robocasa_n15_zmq_eval_container.sh`
- `scripts/run_robocasa_n15_zmq_parallel_eval_container.sh`

모두 env var override 가능하다.

## Smoke Test 결과

실험 worktree에서 pm16 alpha50 checkpoint로 microwave smoke를 돌렸다.

Server:

- GPU: 3
- port: 8056
- container: `isaac-gr00t-robocasa-zmq-smoke-pm16-ckpt20000-gpu3`
- checkpoint: `/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/awr_pm16_alpha50_20k/checkpoint-20000`

Parallel eval:

- env: `PnPMicrowaveToCounter`
- n_envs: 5
- n_episodes: 5
- policy batch mode: `lockstep`
- camera: 256
- policy image: 128
- video: obs source, 256
- container: `isaac-gr00t-robocasa-bench-zmq-lockstep5-obs256-policy128-pm16-microwave-gpu3`

Output:

- `/home/junhyeong/Value/Isaac-GR00T-eval-zmq/local_outputs/robocasa_benchmark/smoke_lockstep5_obs256_policy128_pm16_microwave_gpu3/pm16_alpha50/ckpt20000/PnPMicrowaveToCounter`

결과:

- 5 episodes 완료
- success: 0/5
- total env steps: 4000
- total policy calls: 250
- log에서 `policy_batch size=5`가 50회 확인됨
- episode별 800 step까지 진행 후 종료

영상:

- 최종 mp4 5개 생성됨
- 예시: `ep000_seed1_composite_outcome0.mp4`
- `ffprobe` 확인:
  - width: 768
  - height: 256
  - duration: 10.05s
  - frames: 201
- 3 camera composite라서 256x256 카메라 3개가 가로로 붙은 768x256이 맞다.

## 검증

원본 repo `/home/junhyeong/Value/Isaac-GR00T`에서 다음 검증을 완료했다.

- Python compile 통과:
  - `gr00t/eval/robot.py`
  - `gr00t/eval/service.py`
  - `gr00t/eval/wrappers/robocasa_n15_wrapper.py`
  - `scripts/robocasa_n15_zmq_eval.py`
  - `scripts/robocasa_n15_zmq_parallel_eval.py`
- shell syntax check 통과:
  - `scripts/run_groot_robocasa_zmq_server.sh`
  - `scripts/run_robocasa_n15_zmq_eval_container.sh`
  - `scripts/run_robocasa_n15_zmq_parallel_eval_container.sh`
- 실행 스크립트 권한 확인 및 보정:
  - 세 launcher 모두 executable

## 현재 남은 일

기능 smoke는 통과했지만, 정식 eval 전에는 아래를 확인하면 좋다.

1. `N_ENVS=5` 또는 더 큰 값으로 50 episode 전체 eval이 안정적으로 끝나는지 확인.
2. 같은 schedule JSON으로 재실행했을 때 scene replay와 summary ordering이 기대대로 유지되는지 확인.
3. weight별 비교를 할 때 같은 schedule JSON과 같은 `N_ENVS`, 같은 `policy_batch_mode`를 유지한다.
4. action noise까지 episode 단위로 완전히 독립 재현해야 한다면, action head 내부 sampling generator 제어를 추가로 검토한다.

## 사용 예시

ZMQ server 실행 예시:

```bash
cd /home/junhyeong/Value/Isaac-GR00T
GPU_DEVICE=3 PORT=8056 CHECKPOINT_DIR=/path/to/checkpoint REPLACE=1 \
  scripts/run_groot_robocasa_zmq_server.sh
```

Parallel eval 실행 예시:

```bash
cd /home/junhyeong/Value/Isaac-GR00T
GPU_DEVICE=3 PORT=8056 ENV_NAME=PnPMicrowaveToCounter N_ENVS=5 N_EPISODES=50 \
  scripts/run_robocasa_n15_zmq_parallel_eval_container.sh
```

기본값이 이미 256 camera, 128 policy resize, 256 obs video이므로 특별히 override하지 않아도 된다.
