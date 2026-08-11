# Server Runbook

Hướng dẫn chạy pipeline trên GPU server. Máy local (không GPU) chỉ để sửa code và chạy test.

Yêu cầu: 1× GPU 40-48GB, disk ≥150GB trống, Python ≥3.10, có mạng ra HuggingFace Hub.

---

## 1. Cài đặt

```bash
git clone https://github.com/voluyen/Spectral-guided-learning.git
cd Spectral-guided-learning

python -m venv .venv && source .venv/bin/activate
nvidia-smi                                    # xem CUDA version + VRAM trống
pip install torch --index-url https://download.pytorch.org/whl/cu121   # đổi cu121 cho khớp
pip install -r requirements.txt
pip install vllm                              # chỉ cần cho stage eval
pip install flash-attn --no-build-isolation   # optional
```

**Nên dùng venv riêng, đừng dùng env dùng chung của platform** (ví dụ `cloudspace` của Lightning
Studio). Env dùng chung thường có `scipy`/`scikit-learn` build cho numpy 1.x, xung đột ABI với
`pandas` build cho numpy 2.x — xem mục Troubleshooting.

`flash-attn` build hỏng thì bỏ qua: configs mặc định `attn_implementation: sdpa`, chạy được, chỉ
chậm hơn. Build được thì sửa `sdpa` → `flash_attention_2` trong `configs/capture-config.yaml`,
`configs/train-vanilla.yaml`, `configs/train-spectral.yaml`.

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

Dùng `tmux` — các stage chạy hàng giờ, mất SSH là mất luôn tiến trình.

```bash
tmux new -s spectral
./scripts/run-pipeline.sh data       # phase 2: network-bound, CPU
./scripts/run-pipeline.sh capture    # phase 3: ~2-4h GPU
./scripts/run-pipeline.sh masks      # phase 4: vài giây, CPU
./scripts/run-pipeline.sh smoke      # 8 samples, 1 epoch — thử VRAM thật
./scripts/run-pipeline.sh train      # phase 5: ~10-20h GPU × 2 run
./scripts/run-pipeline.sh eval       # phase 6: ~4-6h GPU × 2 model
```

Detach: `Ctrl-b d`. Attach lại: `tmux attach -t spectral`.

Script tự activate `.venv` (nếu có) và tạo `logs/`. Mỗi stage ghi ra đĩa, stage sau đọc file đó
nên dừng giữa chừng rồi chạy tiếp được — trừ `train`, xem mục Hạn chế.

### 4b. Máy trống — bộ script setup / train / eval / push

Cho server chưa cài gì sẵn. Mỗi việc một script nhỏ, self-contained (param nằm ngay đầu mỗi
script, sửa trực tiếp trong đó). Không đụng tới `data`/`capture`/`masks`: các stage đó vẫn chạy
riêng bằng `run-pipeline.sh` vì có gate bắt buộc dừng lại xem output (mục 5).

```bash
export HF_TOKEN=hf_...                       # bắt buộc cho push (và data nếu gọi run-pipeline.sh)
./scripts/setup.sh                           # venv + torch cu121 + requirements.txt, check GPU
./scripts/train-vanilla.sh                   # train nhánh vanilla (torchrun theo OPTS, tee ra logs/)
./scripts/train-spectral.sh                  # train nhánh spectral
./scripts/eval.sh checkpoints/vanilla vanilla   # vLLM generate + score 1 checkpoint
./scripts/eval.sh checkpoints/spectral spectral
python src/compare_results.py                # in bảng so sánh 2 kết quả
./scripts/push.sh                            # push checkpoint (bỏ checkpoint-*/ trung gian) + log lên HF
./scripts/run.sh                             # gọi toàn bộ chuỗi trên thành 1 pipeline
```

Hai nhánh có file train riêng (`train-vanilla.sh` / `train-spectral.sh`) — chỉnh hyperparameter
từng nhánh độc lập; chúng chỉ khác nhau ở data file + thư mục checkpoint. `eval.sh` không biết
variant: nó eval đúng một checkpoint theo `MODEL`/`TAG` đặt sẵn trong script (arg 1 = model path,
arg 2 = tag để đè); mọi tham số generate cũng nằm trong `eval.sh`. Muốn eval một epoch cụ thể thì
trỏ tới `checkpoints/<variant>/checkpoint-*`. `push.sh` vẫn nhận tên variant làm arg.

Push lên `HF_NAMESPACE/HF_REPO_PREFIX-<variant>` (mặc định `spectral-guided-learning-<variant>`,
private). Set `HF_NAMESPACE` ở đầu `scripts/push.sh` trước khi push.

### Thứ tự và output từng stage

| Stage | Đọc | Ghi |
|---|---|---|
| `data` | AceReason-1.1-SFT (Hub) | `data/train-2k-segmented.jsonl` |
| `capture` | jsonl trên | `data/spectral-strengths.parquet`, `data/spectral/{id}.npz` |
| `masks` | jsonl + parquet | `data/train-vanilla.jsonl`, `data/train-spectral.jsonl`, `data/selection-stats.json` |
| `train` | 2 file train | `checkpoints/{vanilla,spectral}/`, `logs/train-*.log` |
| `eval` | 2 checkpoint | `results/raw/*.jsonl`, `results/*-summary.json`, `results/comparison-table.md` |

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
`python src/build_masks.py --config configs/data-config.yaml --sweep 0.7,0.8,0.9,0.95` (vài giây,
không train lại, không ghi file dataset) để chọn `p` bằng số liệu drop-ratio thay vì đoán.

---

## 6. Hạn chế đã biết

**Disk khi train.** Mặc định `SAVE_STRATEGY=epoch` + `SAVE_TOTAL_LIMIT=6` (đặt ở đầu mỗi
`train-*.sh`), mà HF Trainer lưu cả optimizer state: mỗi checkpoint ≈ 3.4GB (model bf16) + ~14GB
(Adam states) ≈ **17GB**. 6 epoch × 2 run ≈ **200GB**. Không đủ chỗ thì hạ `SAVE_TOTAL_LIMIT`, hoặc
`SAVE_STRATEGY=no` để chỉ giữ model final. Đổi `SAVE_STRATEGY=steps` + `SAVE_STEPS=N` để lưu dày
hơn theo step (tốn thêm disk tương ứng).

**Train không resume, capture thì có.** `train_sft.py` chưa truyền `resume_from_checkpoint`, crash
giữa chừng là chạy lại từ đầu — với run 15h thì nên bổ sung trước khi bắt đầu. `gradient_capture.py`
ngược lại tự resume: mỗi sample ghi `.npz` riêng ngay khi xong, và lần chạy sau bỏ qua mọi id đã
có `.npz` — crash giữa `capture` chỉ mất phần chưa ghi.

**OOM ở seq 16k.** Thứ tự xử lý: đổi optimizer sang `adamw_bnb_8bit` (tiết kiệm ~10GB) → ZeRO-2
CPU offload → hạ cutoff xuống 12k trong `configs/data-config.yaml` (phải chạy lại từ stage `data`).

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

### `data` đứng im ở 0 sample

Bình thường. `datasets` phải tải đủ 10.000 dòng để lấp shuffle buffer trước khi yield dòng đầu.
Log có in cảnh báo `buffering 10,000 rows...` trước khi bar xuất hiện.

### GPQA load lỗi 401/403

Thiếu `HF_TOKEN`, hoặc token chưa được duyệt quyền truy cập dataset. Cách khác: bỏ GPQA
(`--benchmarks aime24,aime25,math500,olympiadbench`) và ghi rõ trong báo cáo.

### Muốn chấm điểm lại mà không generate lại

Raw generations được lưu ở `results/raw/`:

```bash
python src/evaluate.py --rescore results/raw/spectral-math500.jsonl
```

Chạy trên CPU, không cần GPU.

---

## 8. Đồng bộ code

Sửa code ở local → commit → push. Trên server:

```bash
git pull && python -m pytest -q
```

`data/`, `checkpoints/`, `results/raw/`, `logs/` đều nằm trong `.gitignore` — chỉ tồn tại trên
server, không đẩy về repo.
