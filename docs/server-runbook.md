# Server Runbook

Hướng dẫn chạy pipeline trên GPU server. Máy local (không GPU) chỉ để sửa code và chạy test.

Yêu cầu: 1× GPU 40-48GB, disk ≥150GB trống, Python 3.12 (`pyproject.toml` ghim `>=3.12,<3.13`; `uv sync`
tự cài Python 3.12 nếu máy chưa có, miễn có mạng). Có mạng ra HuggingFace Hub để `pip`/`uv`
cài package và tải model/dataset trực tiếp; server không mạng thì xem `download.txt`
(model + dataset cần tải sẵn) và set `LOCAL_MODELS_ROOT`/`LOCAL_DATA_ROOT`/`BENCH_DATA_ROOT` để mọi
script đọc từ local thay vì Hub.

---

## 1. Cài đặt

```bash
git clone https://github.com/voluyen/Spectral-guided-learning.git
cd Spectral-guided-learning

nvidia-smi   # xem "CUDA Version" góc trên phải trước — đó là trần driver hỗ trợ, không phải lựa chọn tự do
./scripts/setup.sh   # uv sync từ pyproject.toml/uv.lock vào 1 venv (.venv) duy nhất, dùng cho mọi stage
```

`pyproject.toml` ghim torch/torchvision/torchaudio/vllm theo 1 CUDA build cụ thể (`[tool.uv.index]`
"pytorch", hiện là `cu130` cho GPU Blackwell/B200) — không có biến môi trường nào đổi được nữa.
Đổi cho GPU đời khác (vd A100 driver cũ hơn) nghĩa là đổi URL đó trong `pyproject.toml` rồi chạy
lại `uv lock` để resolve lại `uv.lock` — kiểm tra trước bằng `nvidia-smi` xem driver có hỗ trợ
CUDA build đó không (driver quá cũ sẽ cài xong nhưng `torch.cuda.is_available()` vẫn `False`).

Sửa `INSTALL_FLASH_ATTN=true` đầu `scripts/setup.sh` để cài `flash-attn` (optional, build lâu).

**Nên dùng venv riêng, đừng dùng env dùng chung của platform** (ví dụ `cloudspace` của Lightning
Studio). Env dùng chung thường có `scipy`/`scikit-learn` build cho numpy 1.x, xung đột ABI với
`pandas` build cho numpy 2.x — xem mục Troubleshooting.

`flash-attn` build hỏng thì bỏ qua: scripts mặc định `sdpa`, chạy được, chỉ chậm hơn. Build được
thì sửa `sdpa` → `flash_attention_2` trong `ATTN=` (`scripts/<family>/{sft,spectral}/`) hoặc
`--attn-implementation` (`scripts/<family>/data/capture_<model>.sh`).

## 2. Biến môi trường

```bash
export HF_TOKEN=hf_...                # bắt buộc: GPQA-Diamond là dataset gated
export HF_HOME=/duong/dan/disk-lon    # ổ ≥100GB: model + dataset cache
```

Thêm vào `~/.bashrc` để không mất khi reconnect SSH.

## 3. Kiểm tra trước khi tốn GPU-hours

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -m pytest -q                                    # pytest tự in số test đã chạy, ~2 giây
PYTHONPATH=src python scripts/smoke_test_pipeline.py   # end-to-end với model tí hon, CPU
```

`pytest` là cách rẻ nhất để lòi hết lỗi môi trường. Pass hết mới chạy tiếp.

---

## 4. Chạy pipeline

Repo giữ đúng 4 track model (Qwen3-1.7B, Qwen3-4B-Instruct-2507, Qwen3-8B, Qwen2.5-7B-Instruct),
tổ chức theo family trước / phase sau (giống layout của distillm), mỗi script tự chứa toàn bộ
tham số của đúng 1 run cụ thể (không yaml, không case/function rẽ nhánh trong 1 file):

```
scripts/qwen3/data/{data,capture,masks}_<model>.sh      # phase 2-4, 1 file/phase/model
scripts/qwen3/sft/sft_<model>.sh                        # phase 5 vanilla (chỉ 1.7b/4b-instruct)
scripts/qwen3/spectral/spectral_<model>.sh              # phase 5 spectral (cả 3 model qwen3)
scripts/qwen3/prucot/{weight,prune}_<model>.sh          # phase 3b/4b Pru-CoT baseline (cả 3 model qwen3)
scripts/qwen3/prucot/prucot_<model>.sh                  # phase 5 Pru-CoT (data khác, hyperparam giống spectral)
scripts/qwen3/eval/eval_<model>.sh                      # phase 6 (arg 1/2 = model path/tag để đè, dùng chung cho vanilla/spectral/prucot)
scripts/qwen25/{data,spectral,prucot,eval}/..._qwen25-7b.sh # tương tự, chỉ 1 model
```

Pru-CoT là baseline thứ 3 (bên cạnh vanilla/spectral), theo `pru-cot.pdf` (`Pru-CoT/` chứa code gốc của
paper): thay vì chỉ mask loss như spectral, nó thật sự **xoá** các bước CoT bị đánh giá dư thừa
rồi train full-supervision trên bản CoT đã rút gọn -- `weight_<model>.sh` (phase 3b) tính trọng
số quan trọng từng bước bằng global optimization (soft-mask + SGD trên embedding, `src/prucot_weight.py`),
`prune_<model>.sh` (phase 4b) dùng 1 LLM pruning agent (vLLM) để quyết định xoá bước nào trong
số các bước dưới ngưỡng (`src/prucot_prune.py`), ghi ra `train-prucot.jsonl` cùng format
`{id, input_ids, loss_mask}` như `build_masks.py` nên `train_sft.py` không cần sửa gì.

Dùng `tmux` — các stage chạy hàng giờ, mất SSH là mất luôn tiến trình. Ví dụ track qwen3-1.7b:

```bash
tmux new -s spectral
./scripts/qwen3/data/data_qwen3-1.7b.sh          # phase 2: network-bound, CPU
./scripts/qwen3/data/capture_qwen3-1.7b.sh       # phase 3: ~2-4h GPU (--verify)
./scripts/qwen3/data/masks_qwen3-1.7b.sh         # phase 4: vài giây, CPU
./scripts/qwen3/prucot/weight_qwen3-1.7b.sh      # phase 3b: backprop qua cả model, chậm hơn capture
./scripts/qwen3/prucot/prune_qwen3-1.7b.sh       # phase 4b: vLLM pruning agent
./scripts/qwen3/sft/sft_qwen3-1.7b.sh            # phase 5 vanilla (~10-20h GPU)
./scripts/qwen3/spectral/spectral_qwen3-1.7b.sh  # phase 5 spectral (~10-20h GPU)
./scripts/qwen3/prucot/prucot_qwen3-1.7b.sh      # phase 5 prucot (~10-20h GPU, cùng hyperparam)
./scripts/qwen3/eval/eval_qwen3-1.7b.sh                                                    # phase 6, spectral (default)
./scripts/qwen3/eval/eval_qwen3-1.7b.sh checkpoints/vanilla-qwen3-1.7b vanilla-qwen3-1.7b   # phase 6, vanilla
./scripts/qwen3/eval/eval_qwen3-1.7b.sh checkpoints/prucot-qwen3-1.7b prucot-qwen3-1.7b     # phase 6, prucot
python src/compare_results.py                    # in bảng so sánh
```

Detach: `Ctrl-b d`. Attach lại: `tmux attach -t spectral`.

`qwen3-1.7b`/`qwen3-4b-instruct` có cả `sft/` (vanilla) lẫn `spectral/` (không có số công bố bên
ngoài để so sánh, cần tự train baseline). `qwen3-8b`/`qwen25-7b` chỉ có `spectral/` (so với số
vanilla đã công bố của P-ALIGN, không cần tự train vanilla).

Script tự activate `.venv` (nếu có) và tạo `logs/`. Mỗi stage ghi ra đĩa, stage sau đọc file đó
nên dừng giữa chừng rồi chạy tiếp được — trừ train, xem mục Hạn chế.

### Thứ tự và output từng stage (ví dụ track qwen3-1.7b — các track khác thay `qwen3-1.7b` bằng tên track)

| Stage | Đọc | Ghi |
|---|---|---|
| `data` | VoCuc/s1K-1.1-DeepSeek-R1-Distill-Qwen-32B (Hub) | `data/qwen3-1.7b/train-s1k-segmented.jsonl` |
| `capture` | jsonl trên | `data/qwen3-1.7b/spectral-strengths.parquet`, `data/qwen3-1.7b/spectral/{id}.npz` |
| `masks` | jsonl + parquet | `data/qwen3-1.7b/train-vanilla.jsonl`, `train-spectral.jsonl`, `selection-stats.json` |
| `prucot/weight` | jsonl (Phase 2) | `data/qwen3-1.7b/prucot-weights.parquet`, `data/qwen3-1.7b/prucot/{id}.npz` |
| `prucot/prune` | jsonl + weights parquet | `data/qwen3-1.7b/train-prucot.jsonl`, `prucot-selection-stats.json` |
| `sft`/`spectral`/`prucot` | 1 file train | `checkpoints/{vanilla,spectral,prucot}-qwen3-1.7b/`, `logs/{sft,spectral,prucot}-*.log` |
| `eval` | 1 checkpoint | `results/<tag>/raw/*.jsonl`, `results/<tag>/summary.json`, `results/comparison-table.md` |

---

## 5. Hai chỗ BẮT BUỘC dừng lại xem output

### Sau `capture`

Đọc mấy dòng cuối log:

- `k*/T mean ratio` — phải ≪ 1. Nếu không, cấu trúc low-rank không tồn tại ở scale 1.7B, tức
  tiền đề của paper không đứng vững ở đây. Dừng lại đánh giá trước khi train.
- `strength spread (p90/p10)` — nếu ~1x (phổ phẳng) thì bước chọn step sẽ không loại được gì.

### Sau `masks`

Đọc bảng drop:

```
variant             steps dropped   median  tokens dropped   median  keeps final
spectral p=0.95            22.8%   23.3%          19.4%   20.3%       66.7%
```

Nếu **cả hai** tỉ lệ đều ~0% thì `train-spectral.jsonl` trùng `train-vanilla.jsonl` — chạy 2 run
là vô nghĩa. `p=0.95` **không phải giá trị công bố của paper** (Eq. 8 không có số cụ thể nào cho
`p` — xem `docs/system-architecture.md`), mà là lựa chọn kỹ thuật, khá "bao trùm", nên đây là
rủi ro thật chứ không phải lo xa. Trước khi chạy `masks` chính thức, cân nhắc
`python src/build_masks.py --data-path data/qwen3-1.7b/train-s1k-segmented.jsonl --sweep 0.7,0.8,0.9,0.95`
(vài giây, không train lại, không ghi file dataset) để chọn `p` bằng số liệu drop-ratio thay vì đoán.

### Sau `prucot/prune`

`prune_<model>.sh` in ra bảng step/token drop giống hệt format của `masks` (cả hai gọi chung
`summarize()` trong `build_masks.py`) -- cùng cách đọc: nếu drop ~0% thì `train-prucot.jsonl`
gần như trùng bản gốc, chạy thêm run `prucot` không cho thêm tín hiệu gì. Kiểm tra thêm
`data/<model>/prucot-failed-log.json` (nếu tồn tại): những sample mà pruning agent trả JSON
không parse được sẽ giữ nguyên toàn bộ CoT (fallback an toàn) thay vì bị loại khỏi tập train.

---

## 6. Hạn chế đã biết

**Disk khi train.** Mặc định `SAVE_STRATEGY=epoch` + `EPOCHS=3` + `SAVE_TOTAL_LIMIT=6` (đặt ở
đầu mỗi script trong `scripts/{qwen3,qwen25}/{sft,spectral}/` — `SAVE_TOTAL_LIMIT` không bao giờ
kích hoạt vì 3 epoch < 6), mà HF Trainer lưu cả optimizer state: mỗi checkpoint ≈ 3.4GB (model
bf16) + ~14GB (Adam states) ≈ **17GB**. 3 epoch × 2 run ≈ **102GB**. Không đủ chỗ thì hạ
`SAVE_TOTAL_LIMIT`, hoặc `SAVE_STRATEGY=no` để chỉ giữ model final. Đổi `SAVE_STRATEGY=steps` +
`SAVE_STEPS=N` để lưu dày hơn theo step (tốn thêm disk tương ứng).

**Train không resume, capture thì có.** `train_sft.py` chưa truyền `resume_from_checkpoint`, crash
giữa chừng là chạy lại từ đầu — với run 15h thì nên bổ sung trước khi bắt đầu. `gradient_capture.py`
ngược lại tự resume: mỗi sample ghi `.npz` riêng ngay khi xong, và lần chạy sau bỏ qua mọi id đã
có `.npz` — crash giữa `capture` chỉ mất phần chưa ghi.

**OOM ở seq dài.** `train_sft.py` hỗ trợ `--deepspeed-config configs/deepspeed/ds_config_zero2_offload.json`
(ZeRO-2 + optimizer state offload sang CPU RAM) và `--max-seq-len N` (bỏ hẳn sample dài hơn N
token khỏi tập train, log số lượng bị bỏ). Đã bật sẵn cho track `qwen3-1.7b`
(`scripts/qwen3/{sft,spectral}/*_qwen3-1.7b.sh`, `MAX_SEQ_LEN=32768` — khớp "Cutoff Length" của
paper Spectral, Table 3; trên data hiện tại (max ~27k token) coi như không lọc gì, target server
là H200/B200 nên đủ VRAM). Lưu ý: **chỉ riêng ZeRO-2 offload không đủ** trên GPU nhỏ VRAM (đã xác
nhận thật trên A100 40GB) — offload chỉ giảm bộ nhớ optimizer state, không giảm activation của 1
sample dài, batch size 1 vẫn OOM ở step 9 khi gặp sample >16k token dù đã bật offload. Đo trực
tiếp trên A100 40GB (Qwen3-1.7B, LoRA r16, gradient checkpointing, không DDP): ~31GB peak ở 12k
token, ~40GB (hết margin) ở 16k, OOM ở 20k. Nếu chạy trên GPU VRAM hạn chế (không phải H200/B200):
hạ `MAX_SEQ_LEN` xuống lại quanh mức đo được cho GPU đó (đo bằng cách tăng dần `seq_len` giả qua 1
forward+backward trước khi chốt), hoặc hạ `MAX_TOKENS` trong `scripts/<family>/data/data_<model>.sh`
(phải chạy lại từ stage `data`), hoặc `CHUNK_SIZE` trong `capture_<model>.sh` (không cần chạy lại `data`).

**`prucot/weight` nặng hơn `capture` nhiều.** `gradient_capture.py` chỉ cần 1 forward `no_grad`
+ SVD; `prucot_weight.py` phải backprop qua *toàn bộ* model mỗi epoch (3 epoch/sample mặc định)
để tối ưu trọng số từng bước -- footprint activation-memory tương đương 1 bước full fine-tuning,
không phải capture. OOM thì hạ `EPOCHS` trong `weight_<model>.sh`, hoặc dùng
`--num-shards`/`--shard-index` (giống `gradient_capture.py`) để chia corpus ra chạy tuần tự
từng phần thay vì 1 tiến trình ôm cả model 2 lần (train + eval reference).

---

## 7. Troubleshooting

### `ValueError: numpy.dtype size changed, Expected 96 from C header, got 88`

`pandas` build cho numpy 2.x nhưng env đang cài numpy 1.x.

```bash
pip install -U "numpy>=2"
```

### `ImportError: numpy.core.multiarray failed to import` (từ scipy/sklearn)

Ngược lại: numpy đã lên 2.x nhưng `scipy`/`scikit-learn` vẫn là bản build cho numpy 1.x.
`transformers` import `sklearn` → `scipy` → nổ.

```bash
pip install -U scipy scikit-learn
```

Còn lòi thêm package khác cùng kiểu ABI thì đừng vá lẻ nữa — tạo venv riêng (mục 1).

### `deepspeed.ops.op_builder.builder.CUDAMismatchException` khi chạy `--deepspeed-config`

DeepSpeed JIT-compile `cpu_adam` (optimizer CPU offload) bằng `nvcc` hệ thống (`/usr/local/cuda`),
không phải bản CUDA mà `torch` được build sẵn (`torch.version.cuda`) — hai bản lệch nhau (vd. nvcc
12.3 nhưng torch build cu118) thì DeepSpeed chặn compile để tránh lỗi ABI. Nếu chỉ lệch minor/major
version (không phải kiến trúc GPU khác hẳn), bật cờ bỏ qua check này (đã verify chạy đúng trên A100
với nvcc 12.3 + torch cu118):

```bash
DS_SKIP_CUDA_CHECK=1 ./scripts/qwen3/sft/sft_qwen3-1.7b.sh
```

### `data` đứng im ở 0 sample

Bình thường. `datasets` phải tải đủ 10.000 dòng để lấp shuffle buffer trước khi yield dòng đầu.
Log có in cảnh báo `buffering 10,000 rows...` trước khi bar xuất hiện.

### GPQA load lỗi 401/403

Thiếu `HF_TOKEN`, hoặc token chưa được duyệt quyền truy cập dataset. Cách khác: bỏ GPQA
(`--benchmarks aime24,aime25,math500,olympiadbench`) và ghi rõ trong báo cáo.

### Muốn chấm điểm lại mà không generate lại

Raw generations được lưu ở `results/<tag>/raw/` (mỗi tag/track một thư mục riêng, không dùng
chung một `results/raw/` phẳng nữa):

```bash
python src/evaluate.py --rescore results/spectral/raw/math500.jsonl
```

Chạy trên CPU, không cần GPU.

---

## 8. Đồng bộ code

Sửa code ở local → commit → push. Trên server:

```bash
git pull && python -m pytest -q
```

`data/`, `checkpoints/`, `results/**/raw/`, `logs/` đều nằm trong `.gitignore` — chỉ tồn tại trên
server, không đẩy về repo.
