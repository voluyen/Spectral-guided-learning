---
title: "Handoff: SmolLM2 trial run, chuyển dataset sang s1K-1.1, dọn yaml train/eval"
date: 2026-08-12
scope: "configs/, scripts/, src/data_prep.py, src/train_sft.py, docs/server-runbook.md, data/, checkpoints/ — phiên làm việc chạy thử pipeline trên server GPU chia sẻ"
method: "Log trực tiếp của phiên làm việc (lệnh đã chạy, log output, quyết định người dùng chốt qua AskUserQuestion). Không phải audit code — xem review-260805-plan-code-vs-paper.md cho việc đó."
---

# Handoff: phiên làm việc 2026-08-12

Mục đích file này: cho phiên Claude Code sau đọc để hiểu ngay context, không phải dò lại từ đầu.

## 0. Bối cảnh máy chủ (quan trọng, đọc trước)

- Máy là **server GPU dùng chung** (2× A100-PCIE-40GB), nhiều user/job khác cùng chạy (thấy được qua `tmux ls`: `bench_l1`, `qlora14b_smoke32k`, `poc7b`, `vllm`, v.v. — không phải của project này, **không đụng vào**).
- `/home/hungpv` là **symlink trỏ vào `/mnt/hungpv`** — nghĩa là `~/.cache/huggingface` và `data/` của project này nằm trên **cùng một ổ đĩa** (`/mnt`, ext4, 1.5TB).
- Ổ `/mnt` đang **đầy 98% (chỉ ~39GB trống)**, nhưng phần chiếm dụng lớn là model của **người khác** (`Qwen3.6-27B-FP8` 29GB, 2× `Qwen2.5-7B-Instruct` 15GB, `Llama-3.1-8B-Instruct` 15GB, `qwen2.5-14b-instruct-unsloth-bnb-4bit` 11GB...). Project này chỉ chiếm ~1.2GB (Qwen3-0.6B-Base) + vài trăm MB.
- Có một job khác tên **`finetun14b`** (QLoRA fine-tune model 14B, ở `/home/hungpv/anvt/finetun14b*`) chạy `torchrun --nproc_per_node=2` chiếm **cả 2 GPU cùng lúc** (~17-20GB/GPU). Job này đã tự thoát một lần trong phiên (thấy log `=== TRAIN EXIT 1 ===` trong shell của họ — **không phải do mình**), rồi sau đó **tự resume lại** (`RESUME=auto`) và chiếm GPU trở lại. Đây là nguyên nhân trực tiếp gây OOM cho gradient capture Qwen3 ở cuối phiên (xem §4).
- **Quy tắc đã thống nhất với user:** không bao giờ kill/can thiệp process của người khác; nếu cần dùng GPU đang được share, phải chủ động giảm footprint bộ nhớ của job mình (ví dụ `gpu-memory-utilization` cho vLLM), không giả định GPU trống.

## 1. Câu hỏi mở đầu: multi-GPU support

User hỏi code hiện tại đã hỗ trợ multi-GPU chưa. Kết luận (agent Explore đã audit):

- **Hạ tầng đã sẵn sàng**: `scripts/train-vanilla.sh`/`train-spectral.sh` launch qua `torchrun` với `--nproc_per_node ${#GPUS[@]}`; `src/train_sft.py` dùng HF `Trainer` (tự wrap DDP qua `accelerate` khi `torchrun` set `WORLD_SIZE`/`LOCAL_RANK`). Model load không có `.to("cuda")` cứng hay `device_map` — tương thích DDP.
- **Đang bị giới hạn 1 GPU** vì `GPUS=(0)` hard-code ở đầu mỗi script (`scripts/train-vanilla.sh:5`, `scripts/train-spectral.sh:5`).
- Chỉ là **DDP thuần** (không DeepSpeed/FSDP) — mỗi GPU giữ bản copy đầy đủ model, không shard. Đủ cho Qwen3-0.6B nhưng không giải quyết vấn đề model không fit 1 GPU.
- Để bật multi-GPU: sửa `GPUS=(0)` → ví dụ `GPUS=(0 1)`, và giảm `GRAD_ACC` tương ứng nếu muốn giữ global batch không đổi (global batch = `per_device_batch_size × world_size × gradient_accumulation_steps`).

## 2. SmolLM2-135M trial run (AceReason data) — ĐÃ XONG

User muốn "chạy thử với model 120M tham số" để test pipeline end-to-end trước khi đụng tới Qwen3 (model chính, tốn GPU-hours nhiều hơn). Chọn **`HuggingFaceTB/SmolLM2-135M`** (135M, Llama arch, `max_position_embeddings=8192`) — GPT-2 (124M) bị loại vì context chỉ 1024, quá ngắn cho reasoning CoT dài.

Đã tạo bộ config/script **song song**, không đụng setup Qwen3 gốc:
- `configs/data-config-smollm2.yaml`, `configs/capture-config-smollm2.yaml`
- `scripts/train-vanilla-smollm2.sh`, `scripts/train-spectral-smollm2.sh`, `scripts/eval-smollm2.sh`

Pipeline đã chạy full (data AceReason, `max_tokens: 8192`):
1. `data_prep.py`: 2000/3543 scanned mẫu giữ được (1542 bị loại vì > 8192 token).
2. `gradient_capture.py --verify`: autograd verify khớp (`max|diff|=1.19e-07`), `k*/T=0.0418`, runtime 543s.
3. `build_masks.py --vanilla`: spectral p=0.95 drop 22.5% token / 16.3% step corpus-level.
4. Train (EPOCHS **đã giảm từ 6 → 3** theo yêu cầu user):
   - **vanilla**: `train_loss=1.2758`, runtime 1904s, 8,954,417 supervised tokens.
   - **spectral**: `train_loss=1.4608`, runtime 1172s (nhanh hơn hẳn — vì lúc train spectral, job `finetun14b` đã tạm thoát, GPU 0 không bị share nữa), 6,918,980 supervised tokens.
5. Eval `vanilla-smollm2` (vLLM, GPU 1): chạy được `aime24` (0% acc, 93.3% truncated) và `aime25` (0% acc, 96.7% truncated) — **0% là kỳ vọng bình thường** cho model 135M train 3 epoch trên bài toán AIME khó nhất. User yêu cầu dừng eval giữa chừng (đã dừng sạch, GPU 1 giải phóng về 0MB) vì không cần điểm benchmark đầy đủ, chỉ cần xác nhận pipeline chạy ổn.

**Lưu ý kỹ thuật đã học được, áp dụng lại nếu cần eval trên GPU share:**
`gpu_memory_utilization` của vLLM tính theo % **TỔNG** VRAM GPU (không phải % phần trống). Nếu GPU đã bị process khác chiếm ~50%, set `gpu_memory_utilization=0.3` sẽ ra ngân sách âm ngay từ đầu → lỗi `"No available memory for the cache blocks"`. Phải set đủ cao để `fraction × total − used_by_others > 0` với margin cho weights+KV cache.

**Checkpoint kết quả**: `checkpoints/vanilla-smollm2-acereason/`, `checkpoints/spectral-smollm2-acereason/` (đã đổi tên thêm suffix `-acereason` khi pivot sang dataset s1K-1.1, xem §3, để không mất kết quả).

## 3. Pivot dataset: AceReason → s1K-1.1

User muốn đổi dataset train sang **`simplescaling/s1K-1.1`** (1000 dòng, distillation reasoning trace từ DeepSeek-R1/Gemini).

**Khác biệt schema quan trọng** so với `nvidia/AceReason-1.1-SFT`:
- Field câu hỏi: `question` (khớp fallback có sẵn trong `data_prep.py`).
- Field category: `cot_type` (giá trị `"math"` hoặc `"science"`, 877/1000 dòng là math) — **khác** với `category`/`source` mà `iter_samples()` gốc check.
- Field `solution` **không** phải long-CoT — chỉ là đáp án số ngắn (ví dụ `"128"`). Long-CoT thật nằm ở `deepseek_thinking_trajectory` (~21k ký tự) + `deepseek_attempt` (~1-2k ký tự, bản viết lại gọn).

**Đã sửa `src/data_prep.py` (`iter_samples()`)**: thêm `cot_type` vào fallback category field; khi có `deepseek_thinking_trajectory`+`deepseek_attempt`, ghép thành `<think>\n{trajectory}\n</think>\n\n{attempt}` làm response (thay vì dùng `solution`). Không phá logic cũ (AceReason vẫn dùng `solution`/`response`/`output` như trước vì không có field `deepseek_*`).

Đã chạy `data_prep.py` cho SmolLM2+s1K-1.1 (context 8192): **292/877** mẫu math giữ được (66% bị loại vì response quá dài — s1K-1.1 vốn có reasoning trace rất dài, trung bình ~29k ký tự).

## 4. Pivot model: bỏ SmolLM2, tập trung Qwen3 (model chính) — ĐANG DỞ

User: "không cần chạy cho model SmolLM2 đâu, về sau tôi sử dụng các model chính là Qwen3 nên cần processed data cho Qwen3 thôi."

Đã tạo (song song với `data-config.yaml`/`capture-config.yaml` gốc dùng AceReason — **không đụng**):
- `configs/data-config-s1k.yaml`: Qwen3 tokenizer, `dataset_name: simplescaling/s1K-1.1`, `category_filter: math`, `max_tokens: 16384`, `output_path: data/s1k/train-s1k-segmented.jsonl`.
- `configs/capture-config-s1k.yaml`: `model_name: Qwen/Qwen3-0.6B-Base`, `chunk_size: 256` (**đã giảm từ 1024** — xem lý do OOM dưới đây; có thể trả về 1024 khi GPU rảnh hẳn).

**Trạng thái:**
1. ✅ `data_prep.py --config configs/data-config-s1k.yaml`: **778/877** mẫu giữ được (chỉ 11% loại, vì context 16384 rộng hơn nhiều so với SmolLM2's 8192). File: `data/s1k/train-s1k-segmented.jsonl`.
2. ❌ `gradient_capture.py --config configs/capture-config-s1k.yaml --verify`: **OOM 2 lần liên tiếp** (cả ở `chunk_size=1024` và `chunk_size=256` — giảm chunk_size không giúp vì lần 2 crash ngay ở forward pass gốc của Qwen3, không phải phần chunk unembedding). Nguyên nhân: job `finetun14b` đang chiếm ~18GB mỗi GPU (cả 2 GPU), chỉ còn ~21GB trống, mà forward pass của Qwen3-0.6B trên sample ~16k token cần hơn số đó.
3. User quyết định: **"để đó, chạy sau"** — không tiếp tục ngay bây giờ.

**Đã setup để chạy lại dễ dàng:** tmux session **`sgl-s1k-capture`** (idle ở bash prompt, lệnh đã chạy xong với lỗi, không tốn tài nguyên khi idle). Để chạy lại:
```bash
tmux attach -t sgl-s1k-capture
source .venv/bin/activate && export PYTHONPATH=$(pwd)/src
python src/gradient_capture.py --config configs/capture-config-s1k.yaml --verify 2>&1 | tee logs/gradient-capture-s1k-qwen3.log
```
Kiểm tra `nvidia-smi` trước — nếu `finetun14b` đã xong (2 process, mỗi GPU release hết VRAM), có thể trả `chunk_size` về `1024` trong `configs/capture-config-s1k.yaml` trước khi chạy.

**Việc tiếp theo sau khi gradient_capture xong:** `build_masks.py --config configs/data-config-s1k.yaml --vanilla` → sinh `data/s1k/train-vanilla.jsonl` + `data/s1k/train-spectral.jsonl`, sẵn sàng cho train Qwen3 (dùng `scripts/train-vanilla.sh`/`train-spectral.sh` nhưng phải sửa `DATA_PATH`/`OUTPUT_DIR` trong 2 script này trỏ vào `data/s1k/` và ví dụ `checkpoints/vanilla-s1k`/`checkpoints/spectral-s1k` — **chưa làm**, vì user chưa yêu cầu train Qwen3 thật (10-20h GPU/run theo comment cũ trong `run-pipeline.sh`), cần hỏi trước khi launch).

## 5. Dọn dẹp: bỏ yaml cho train/eval

User: script `.sh` (`train-vanilla.sh`, `train-spectral.sh`, `eval.sh`) đã tự chứa toàn bộ setting inline và **không đọc yaml** (chỉ pass CLI flags) → các file yaml tương ứng là tài liệu trùng lặp, không cần giữ.

**Đã xoá:**
- `configs/train-vanilla.yaml`, `configs/train-spectral.yaml`, `configs/eval-config.yaml` (bản gốc Qwen3)
- `configs/train-vanilla-smollm2.yaml`, `configs/train-spectral-smollm2.yaml` (bản smollm2 vừa tạo trong phiên, đã xoá luôn)

**Đã sửa các chỗ tham chiếu:**
- `scripts/run-pipeline.sh`: case `smoke`/`train` chuyển từ `python src/train_sft.py --config configs/train-*.yaml` sang gọi `./scripts/train-vanilla.sh`/`train-spectral.sh` (case `smoke` dùng CLI flags trực tiếp).
- `src/train_sft.py`: sửa docstring đầu file, không còn ví dụ `--config configs/train-vanilla.yaml`.
- `docs/server-runbook.md`: sửa hướng dẫn đổi `attn_implementation` sang trỏ vào `capture-config.yaml` + biến `ATTN=` trong 2 script train.

**Giữ nguyên** (không thuộc phạm vi yêu cầu, và không có `.sh` wrapper trùng lặp): `configs/data-config.yaml`, `configs/capture-config.yaml` và 2 bản `-smollm2`/`-s1k` — các phase data prep/gradient capture chỉ được gọi trực tiếp bằng `python ... --config ...`, không có script self-contained tương đương.

**Chưa đụng** (tài liệu lịch sử, không sửa cho khớp thực tế mới): các file trong `plans/` — đây là record quá trình, không phải doc sống.

## 6. Snapshot dataset vào `data/raw/`

User muốn giữ bản copy dataset train + eval ngay trong `data/` để sau này không phải tải lại — dù đã lưu ý là HF cache (`~/.cache/huggingface`, thực chất nằm trên `/mnt` — xem §0) vốn đã persist giữa các lần chạy rồi.

**Đã snapshot** (dùng `datasets.load_dataset(...).to_json(...)`, không streaming — snapshot toàn bộ, không lọc):
- `data/raw/s1k-1.1.jsonl` — toàn bộ 1000 dòng, 50.9MB.
- `data/raw/eval/aime24.jsonl`, `aime25.jsonl`, `math500.jsonl`, `olympiadbench.jsonl` — 4 benchmark eval không bị gate, tổng ~2MB.

**Chưa snapshot:**
- `gpqa` (`Idavidrein/gpqa`, config `gpqa_diamond`) — dataset **gated**, cần `HF_TOKEN` (biến môi trường chưa set trong session này — xem `docs/server-runbook.md` mục biến môi trường). Cần user cung cấp token để tải.
- **AceReason** (`nvidia/AceReason-1.1-SFT`) — **cố ý không** snapshot toàn bộ vì đây là corpus rất lớn, `data_prep.py` chỉ stream một phần nhỏ (2000-3543 dòng scan) qua `streaming=True`; tải hết sẽ tốn dung lượng lớn trên ổ đã gần đầy (§0). Bản đã trích xuất/tokenize sẵn (`data/smollm2/train-acereason-segmented.jsonl`, 85MB) đã là snapshot cục bộ đủ dùng cho SmolLM2 track.

## 7. File inventory (mới/sửa/xoá trong phiên này)

**Mới:**
- `configs/data-config-smollm2.yaml`, `configs/capture-config-smollm2.yaml`
- `configs/data-config-s1k.yaml`, `configs/capture-config-s1k.yaml`
- `scripts/train-vanilla-smollm2.sh`, `scripts/train-spectral-smollm2.sh`, `scripts/eval-smollm2.sh`
- `data/raw/s1k-1.1.jsonl`, `data/raw/eval/{aime24,aime25,math500,olympiadbench}.jsonl`
- `data/smollm2/train-s1k-segmented.jsonl` (292 mẫu, SmolLM2 tokenizer)
- `data/s1k/train-s1k-segmented.jsonl` (778 mẫu, Qwen3 tokenizer)
- `checkpoints/vanilla-smollm2-acereason/`, `checkpoints/spectral-smollm2-acereason/`

**Sửa:**
- `src/data_prep.py` (`iter_samples()`: hỗ trợ `cot_type` + ghép `deepseek_thinking_trajectory`+`deepseek_attempt`)
- `scripts/run-pipeline.sh`, `src/train_sft.py` (docstring), `docs/server-runbook.md`

**Xoá:**
- `configs/train-vanilla.yaml`, `configs/train-spectral.yaml`, `configs/eval-config.yaml`

**Đổi tên (backup, tránh mất kết quả AceReason khi pivot sang s1K-1.1):**
- `data/smollm2/train-2k-segmented.jsonl` → `train-acereason-segmented.jsonl`
- `data/smollm2/train-vanilla.jsonl` → `train-vanilla-acereason.jsonl`
- `data/smollm2/train-spectral.jsonl` → `train-spectral-acereason.jsonl`
- `data/smollm2/spectral/` → `spectral-acereason/`
- `data/smollm2/spectral-strengths.parquet` → `spectral-strengths-acereason.parquet`
- `data/smollm2/selection-stats.json` → `selection-stats-acereason.json`

## 8. Việc còn tồn (next steps)

1. **Gradient capture Qwen3+s1K-1.1** đang chờ GPU rảnh (§4) — resume qua tmux `sgl-s1k-capture`.
2. Sau khi có `data/s1k/train-vanilla.jsonl`/`train-spectral.jsonl`, cần **hỏi user trước khi launch train Qwen3 thật** (ước tính 10-20h GPU/run theo comment cũ, tốn tài nguyên đáng kể trên server đang share).
3. `gpqa` eval benchmark chưa snapshot — cần `HF_TOKEN` từ user.
4. Chưa rõ user còn muốn giữ track SmolLM2 (đã train xong, backup ở `*-acereason`) để so sánh gì thêm, hay track này coi như đã "xong nhiệm vụ" (chỉ để validate pipeline).
5. Skill `ak:handoff` mà user gọi ở cuối phiên **không tồn tại** trong danh sách skill hiện có (đã tra `ListSkills`/`SearchSkills`, không thấy) — file handoff này được soạn tay thay thế. Nếu `ak:handoff` là skill từ plugin riêng của user, cần họ cài/enable lại.
