---
title: "Handoff: chẩn đoán fail bước prucot/weight 4b (log shard0 rỗng), audit enable-thinking + thinking-mode của Pru-CoT gốc, push nhánh temp"
date: 2026-08-29
scope: "Đọc-hiểu code + docs để trả lời câu hỏi của user; KHÔNG sửa code src/scripts. Có 1 thao tác git: tạo & push nhánh `temp` lên origin (voluyen/Spectral-guided-learning)."
method: "Log trực tiếp của phiên: đọc code (src/, scripts/, Pru-CoT/), đọc docs (handoff cũ, server-runbook, system-architecture, pru-cot.pdf), verify trên transformers 5.5.3 đã cài, đối chiếu 2 nhánh qua git. Không phải audit toàn diện."
---

# Handoff: phiên làm việc 2026-08-29 (sáng)

Mục đích: cho phiên sau nắm ngay kết luận, không phải dò lại. Phiên này **thuần điều tra/giải thích** cho user (tiếng Việt) + 1 thao tác git cuối. Không đụng code trong `src/` hay `scripts/`.

## 0. Tóm tắt nhanh

User hỏi (lượt trước, chạy pipeline trên B200 bằng deployment `new_nothing`): vì sao bước `prucot/weight` track 4b fail, log `logs/qwen3-4b-instruct-prucot-weight-shard0.log` **rỗng (chỉ có đúng 1 dòng là tên file)**, và vì sao 2 model khác nhau ở cờ `enable-thinking`. Rồi yêu cầu kiểm tra code gốc `Pru-CoT/` xem có dùng thinking mode không. Cuối cùng yêu cầu push tất cả file vào nhánh mới `temp` ở repo Spectral (origin).

4 kết luận chính:
1. **Log shard0 rỗng** ≠ bằng chứng chắc chắn OOM. Cơ chế: shard được commit `8711691` tự bung 1/GPU, stdout Python block-buffered bị mất khi process bị SIGKILL (đặc trưng OOM-killer) trước khi flush. Cần `dmesg` để chốt.
2. **Bug thật làm dễ OOM**: `prucot_weight.py` gọi `gradient_checkpointing_enable()` rồi `model.eval()` → transformers chỉ chạy checkpointing khi `self.training==True` → checkpointing bị tắt âm thầm. Fix `9e8faea` (eval→train) **chỉ có trên nhánh `spectral-guided-learning`**, **CHƯA có trên `sang/training-scripts`**.
3. **enable-thinking khác nhau** là do 2 loại model khác nhau (1.7B hybrid-thinking vs 4B-Instruct-2507 non-thinking), chỉ ảnh hưởng prompt, **không** ảnh hưởng response/step-scoring → **không** phải nguyên nhân fail.
4. **Pru-CoT gốc LUÔN dùng thinking mode** (verify trong `Pru-CoT/cot_weight.py`). Track 1.7b (`--enable-thinking`) trung thành; track 4b (`--no-enable-thinking`) lệch, do bám model của paper Spectral.

## 1. Bối cảnh nhánh (đọc kỹ — dễ nhầm)

Có **2 deployment tách biệt** (xem handoff `260828-1121` §3):

| | Nhánh user đang mở | Deployment chạy B200 |
|---|---|---|
| Remote | `origin` = `github.com/voluyen/Spectral-guided-learning` | `target` = `github.com/aiskylimit/new_nothing` |
| Nhánh | `sang/training-scripts` (tip `f0ef176`) | `spectral-guided-learning` (tip `1c1ccb9`) |
| Layout | phẳng, file ở root | monorepo, project trong subfolder `spectral-guided-learning/` |
| venv/path | `.venv` local, `_models/spectral-guided-learning` | `/mnt/local/uvenvs/...`, `_models/aiskylimit_new_nothing` |
| Có fix `9e8faea`? | **KHÔNG** | **CÓ** |

Cả 2 remote đều đã có sẵn trong local `.git/config` (origin + target, kèm PAT `PhuocSang07`). Local repo path: `/mnt/hungpv/Spectral-guided-learning`.

Các commit fix của deployment B200 (đều trên nhánh `spectral-guided-learning`, tác giả phongdq):
- `8711691` "Actually use every GPU in the capture/weight/prune phases" — `capture_*.sh`/`weight_*.sh` tự bung **1 shard/GPU** qua `--num-shards/--shard-index` (mỗi shard ghi log riêng `...-shard{i}.log`); `prune_*.sh` set `TENSOR_PARALLEL_SIZE=${#GPUS[@]}`.
- `441a278` — cap prune `max_model_len` = 32768.
- `9e8faea` — `prucot_weight.py`: `model.eval()` → `model.train()` khi bật checkpointing.

## 2. Vì sao `...-prucot-weight-shard0.log` rỗng (chỉ 1 dòng = tên file)

- **Không script nào trong repo `sang/training-scripts` sinh tên `-shard0.log`.** Ở nhánh này `weight_qwen3-4b-instruct.sh:37` ghi `logs/qwen3-4b-instruct-prucot-weight.log` (1 process, không shard). Tên `-shard0.log` là của **nhánh B200** sau commit `8711691` (tự bung 1 shard/GPU, mỗi shard `python ... --shard-index i > logs/...-shard$i.log 2>&1`). User xác nhận **chỉ chạy `project_commands.sh`**, không gõ tay, không xem runbook → khớp: script tự shard.
- **1 dòng duy nhất = header shell tự `echo "$LOG"`**, KHÔNG phải output Python. `prucot_weight.py` không bao giờ in tên log của chính nó; dòng in đầu tiên khi sharded là `shard X/N: K records assigned` (`src/prucot_weight.py:209`) — dòng này **không có** ⇒ Python chưa in được gì.
- **0 output Python** khớp với: stdout khi redirect là **block-buffered (~4-8KB)**; nếu process bị **SIGKILL** (OOM-killer giết cứng, không traceback) trước khi buffer flush → mất sạch → log rỗng. NHƯNG cũng khớp "đang treo/chưa flush" hoặc "chưa launch". ⇒ **log rỗng KHÔNG chứng minh chắc chắn OOM**, chỉ tương thích.
- **Chốt nguyên nhân** cần (trên B200): `dmesg -T | grep -iE "killed process|out of memory|oom"`; `tail logs/qwen3-4b-instruct-prucot-weight.log`; đếm `.npz` trong `data/qwen3-4b-instruct/prucot/`. Lần sau chạy nên thêm `PYTHONUNBUFFERED=1`/`python -u` để log không mất khi bị kill.

## 3. Bug checkpointing (nguyên nhân dễ OOM) — verify xong

`src/prucot_weight.py:194-196` (nhánh `sang/training-scripts`, hiện tại):
```python
if config["gradient_checkpointing"]:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.eval()          # <-- vẫn còn ở nhánh này
```
Verify trên `transformers 5.5.3` đã cài:
- `Qwen3DecoderLayer(GradientCheckpointingLayer)` — `.venv/.../models/qwen3/modeling_qwen3.py:294`.
- `GradientCheckpointingLayer.__call__`: `if self.gradient_checkpointing and self.training:` — `.venv/.../transformers/modeling_layers.py:60`.
⇒ Ở `eval()` → `self.training==False` → checkpointing **bị vô hiệu âm thầm** → giữ activation cả model 4B ở 32k token → dễ OOM. Fix `9e8faea` đổi sang `train()` (params đã freeze + Qwen dropout=0 nên numerically identical, chỉ bật lại checkpointing).

**Điểm cần nhớ:** fix này CHỈ nằm trên nhánh `spectral-guided-learning`. Nhánh `sang/training-scripts` (và snapshot `temp` vừa push) **vẫn còn bug**. Handoff `260828-1121` §3 cố ý giữ 2 repo tách biệt **vì lý do path/venv**, nhưng `9e8faea` là **bug fix thật, không liên quan path** → nên back-port. **Chưa làm** (chờ user quyết — đã hỏi cuối phiên, user chưa trả lời).

## 4. Vì sao 2 model khác `enable-thinking`

| Track | Model | Bản chất | Cờ (data script) |
|---|---|---|---|
| 1.7b | `Qwen3-1.7B` | Qwen3 gốc, **hybrid thinking** (template đọc `enable_thinking`, default True) | `--enable-thinking` (`scripts/qwen3/data/data_qwen3-1.7b.sh:30`) |
| 4b | `Qwen3-4B-Instruct-2507` | bản 2507 tách dòng Instruct = **non-thinking** (Qwen KHÔNG ra `1.7B-Instruct-2507`) | `--no-enable-thinking` (`scripts/qwen3/data/data_qwen3-4b-instruct.sh:30`) |

- Cờ đặt để **khớp chat-template từng model**, không tùy tiện. Với model non-thinking, template bỏ qua kwarg (docstring `src/data_prep.py:28-30` tự ghi).
- Cờ **chỉ ảnh hưởng PROMPT**, không đụng response. Response luôn chứa `<think>...</think>` vì đến thẳng từ dataset `s1K-1.1-DeepSeek-R1-Distill-Qwen-32B` (`src/data_prep.py:100-108`). Điểm cắt CoT/solution dựa `</think>` **trong response** (`src/segmentation.py:48-51`) → độc lập cờ.
- ⇒ **KHÔNG phải nguyên nhân bước weight 4b fail.** Mỗi track nhất quán nội bộ (data→capture→weight→prune→sft→eval cùng format) → vẫn apples-to-apples TRONG từng track.
- Docs KHÔNG giải thích cờ này (grep `docs/`+`plans/` trống). Nhưng model là do bám paper Spectral: `plans/reports/review-260805-plan-code-vs-paper.md:31` cho thấy paper Spectral dùng `Qwen3-4B-Instruct-2507`. Repo thu nhỏ xuống `Qwen3-1.7B` cho track nhẹ.

## 5. Pru-CoT gốc CÓ dùng thinking mode không → CÓ (verify trong code)

Đọc `Pru-CoT/cot_weight.py` + `Pru-CoT/LLM_prune_threshold.py`:
- **Ép mở `<think>` mọi sample:** prompt = `apply_chat_template(..., add_generation_prompt=True) + '<think>\n\n'` (`cot_weight.py:98-108, 156-166`). KHÔNG dùng `enable_thinking` — tự nối `<think>\n\n`.
- **Method chạy TRÊN khối think:** data schema `think_components.assistant_think.sections` (step cần chấm điểm nằm **trong** `<think>`), `think_components.solution` là phần sau `</think>` (`cot_weight.py:110-112`). Response = `<think>\n\n{steps}\n\n</think>\n\n\n\n{solution}<｜end▁of▁sentence｜>`. Chỉ solution được supervise (`cot_weight.py:84-87`, đúng Eq.2).
- **Model gốc = reasoning-native:** EOS `<｜end▁of▁sentence｜>` = DeepSeek → base là **DeepSeek-R1-Distill-Qwen** (paper: 1.5B/7B). Không có nhánh non-thinking.
- **Prune cũng giữ think:** `LLM_prune_threshold.py:209-211` luôn tái dựng `<think>\n\n...\n\n</think>\n\n\n\n{solution}`.
- **Bonus OOM:** `Pru-CoT/config/default_config.yaml` bật `fsdp_activation_checkpointing: true` → bản gốc DỰA vào activation checkpointing để đủ bộ nhớ ⇒ đúng thứ bug `model.eval()` vô hiệu hoá.

**Hệ quả cho repo:** Pru-CoT gốc = thinking LUÔN bật. Track 1.7b (`--enable-thinking`) **trung thành**. Track 4b (`--no-enable-thinking`) **lệch** — probe chạy trên model non-thinking, prompt không mồi `<think>` (response vẫn có `<think>` từ s1K nên cơ chế vẫn chạy, nhưng model freeze không được prime → tín hiệu importance kém khớp ý đồ Pru-CoT). Lý do 4b non-thinking: bám model của paper Spectral (method chính), Pru-CoT chỉ là baseline chạy trên cùng model cho công bằng.

## 6. Docs đã đọc

- `pru-cot.pdf` (13 trang) = paper Pru-CoT (Findings of ACL 2026, Han Liu et al., DLUT). Method khớp code repo: Eq.1 `ẽ = w·e + (1−w)·n` (n=embedding filler `"."`); Eq.2 freeze student, tối ưu w/sample, minimize CE loss **chỉ trên solution**; giữ trọng số epoch loss thấp nhất; SGD 3 epoch lr 10; rồi threshold τ + LLM-guided pruning. **Divergence có chủ đích:** paper dùng budget 8K token + lọc >8192; repo dùng `MAX_TOKENS=32768`.
- `ThoughtFold.pdf` = **hỏng** (12KB, 0 ký tự text, không font, renderer báo "corrupted or invalid"). Không đọc được — cần bản khác nếu cần.
- Ngoài ra đã đọc: `plans/reports/handoff-260828-1121-...md`, `docs/server-runbook.md` (§6 tự ghi `prucot/weight` là bước nặng nhất = footprint full-FT, remedy: hạ EPOCHS hoặc shard), `docs/system-architecture.md` (mô tả bản gốc cũ: 1.7B-Base/cutoff 16k/2k sample — **doc drift** so với script hiện tại: chat-template/32k/1050 sample).

## 7. Thao tác git đã thực hiện: push nhánh `temp` lên origin

User chốt (qua AskUserQuestion): repo = **origin** (voluyen), phạm vi = **file git track được** (không ép file gitignore).

Đã làm:
- `git checkout -b temp` (từ HEAD = `sang/training-scripts` tip `f0ef176`).
- `git add -A` (tôn trọng `.gitignore` — verify không có path nào trong `data/checkpoints/results/logs/.venv*/Pru-CoT/ThoughtFold/P-ALIGN`).
- Commit `d3d3a2d` "snapshot: WIP training-config tweaks + reference PDFs and PR#3 handoff" — **11 file** (8 sửa + 3 thêm):
  - Sửa: `scripts/qwen3/{prucot,sft,spectral}/*_qwen3-{1.7b,4b-instruct}.sh` (6 file), `src/train_sft.py`, `configs/deepspeed/ds_config_zero2_offload.json`.
  - Thêm: `pru-cot.pdf`, `ThoughtFold.pdf`, `plans/reports/handoff-260828-1121-...md`.
- `git push -u origin temp` → tạo `origin/temp` thành công. (Các thay đổi này là WIP đang-dở của user, chưa rõ hết ý đồ — chỉ snapshot theo yêu cầu.)

**Lưu ý:** commit KHÔNG kèm trailer `Co-Authored-By: Claude` (theo preference của user).

## 8. Trạng thái cuối phiên

- Local working tree: **đang ở nhánh `temp`** (đã checkout sang), sạch (mọi WIP đã vào commit `d3d3a2d`).
- `origin/temp` = `f0ef176` + `d3d3a2d`. Link PR (nếu cần): https://github.com/voluyen/Spectral-guided-learning/pull/new/temp
- `sang/training-scripts` vẫn ở `f0ef176` (chưa có WIP). Nếu user muốn về `sang/training-scripts` mà GIỮ các sửa đổi như cũ ở đó → cần xử lý riêng (đã offer, chưa làm).
- File handoff này (`plans/reports/handoff-260829-0719-...md`) mới tạo, **chưa được add/commit** vào nhánh nào (working tree hiện ở `temp`).

## 9. Việc còn mở (chờ user quyết)

1. **Back-port fix `9e8faea` (eval→train)** sang `sang/training-scripts` (và/hoặc `temp`)? — Đây là bug fix thật, nhánh này chưa có. Đã hỏi, chưa trả lời.
2. **Chỉnh track 4b về thinking** để trung thành Pru-CoT hơn? — Đánh đổi với việc bám model paper Spectral. Cần user cân nhắc mục tiêu (reproduce Spectral vs reproduce Pru-CoT).
3. **Chốt OOM thật** trên B200: chạy `dmesg | grep -i oom` + `tail logs/qwen3-4b-instruct-prucot-weight.log`; nếu đúng OOM sau khi có fix checkpointing thì mới tính tiếp (hạ EPOCHS / shard / giảm activation).
4. Các việc mở từ handoff `260828-1121` vẫn còn: `torchaudio==2.11.0`/`torch==2.13.0` pin cần verify mạng; duplication lớn (shard/resume/merge, config-merge boilerplate, ~30 script bash gần giống nhau); track qwen25-7b/qwen3-8b trên `new_nothing` chưa wire offline-mirror.
