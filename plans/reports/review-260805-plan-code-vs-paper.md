---
title: "Review: Plan & Code vs. Paper — Spectral-guided Learning Reproduction"
date: 2026-08-05
scope: "plans/260805-0836-spectral-guided-learning-reproduction/ + src/ + configs/ vs Uncovering_the_Gradient.pdf"
method: "Full-text read of the 14-page PDF (pdftotext, cross-checked page by page), full read of all 7 phase files + plan.md + brainstorm report + docs, full read of all 13 src/ modules + 6 test files + configs, verification against transformers 5.14.1 source and live HF config/dataset schemas."
---

# Review: Spectral-guided Learning — Plan & Code vs. Paper

## 0. Nguồn và phương pháp kiểm chứng

- Đọc toàn văn `Uncovering_the_Gradient.pdf` (14 trang, kể cả Appendix A–D) qua `pdftotext -layout`, không dựa vào tóm tắt.
- Đọc toàn bộ `plan.md`, 7 phase file, brainstorm report, `docs/system-architecture.md`, `docs/project-changelog.md`.
- Đọc toàn bộ 13 module trong `src/`, 6 file test, 5 file config, `scripts/run-pipeline.sh`.
- Kiểm chứng chéo qua web: venue ICML 2026 (Seoul, 6–11/7/2026) hợp lệ; `config.json` thật của `Qwen/Qwen3-1.7B-Base` trên HuggingFace; schema thật của `nvidia/AceReason-1.1-SFT`; hành vi thật của `transformers` Trainer (đọc source `trainer.py` phiên bản 5.14.1 cài trong sandbox) cho phần chuẩn hóa loss.
- Không có GPU/torch trong sandbox nên **không** chạy lại được pytest suite hay xác nhận số liệu GPU (3.7e-9, 48 tests xanh...) — phần này được đánh giá bằng đọc code tĩnh + suy luận toán học tay, không phải chạy lại thực nghiệm.

## 1. Bài báo nói gì (phần kỹ thuật cần để đối chiếu)

**"Uncovering the Gradient Geometry of Long CoT: A Spectral-guided Approach to Reasoning Distillation"**, Fan et al., ICML 2026 (PMLR 306, Seoul). Không tìm thấy bản arXiv được index — có thể do quá mới hoặc chỉ công bố dạng proceedings; venue/định dạng trích dẫn (43rd ICML, Seoul, PMLR 306) khớp với lịch ICML 2026 thật (6–11/7/2026), nhưng danh tính tác giả/venue chỉ được xác nhận từ chính PDF, chưa đối chiếu độc lập được qua nguồn thứ hai.

**Cơ chế cốt lõi (Loss Subspace Attribution):**
- `g_t = ∇_{h_t} ℓ_t ∈ R^d` (Eq.1): gradient của loss token t theo **hidden state cuối cùng** h_t (không phải gradient tham số) — đây là lý do phương pháp rẻ để tính.
- `G = [g_1,...,g_T]^T ∈ R^{T×d}` (Eq.2), SVD `G = UΣV^T` (Eq.3), năng lượng tích lũy `E(k) = Σσ_i²/Σσ_j²` (Eq.4).
- Consensus subspace = span của k* vector kỳ dị phải đầu tiên; residual = phần bù trực giao (Eq.5–6).
- Spectral strength của một step s: `S(s) = mean_{t∈s} ||(U_{1:k*})_t||²` (Eq.7) — **leverage score thống kê dựa trên U, không nhân với Σ**.
- Dynamic truncation (Eq.8): sắp bước theo S(s) giảm dần, chọn tập tối thiểu đạt `ΣS(selected)/ΣS(all) ≥ p`.
- Masked loss (Eq.9): `L = -(1/Z)Σ_t M_t log P(y_t|y_<t)`, `Z = Σ_t M_t`, `M_t∈{0,1}`.
- Mọi bước trên (SVD, spectral strength, lựa chọn) được làm **theo từng sample riêng lẻ** — xác nhận rõ ở §3.2: "Within each sample, we rank all reasoning steps..." — không phải SVD toàn corpus.
- Appendix D chứng minh hình thức: chiếu lên consensus subspace giảm variance đúng hệ số `k*/d` (giả định noise isotropic, signal nằm gọn trong subspace).
- Setup thật: 4 model (Qwen3-4B-Base/8B-Base/4B-Instruct-2507, Qwen2.5-7B-Instruct), 10k sample từ AceReason-1.1-SFT (chỉ lấy **problem**, CoT được **sinh lại** bằng DeepSeek-R1-0528), cutoff 32k, LLaMA-Factory, lr 5e-5/cosine_with_min_lr(1e-5)/warmup 0.1/6 epoch/batch 32, eval temp 0.6/top-p 0.95/n=4/max 32768 token, 5 benchmark (AIME24/25, MATH500, OlympiadBench, GPQA-Diamond), baseline TDST/EDSP (+ appendix: ESSP/HTST/Random/Residual/Static Ratio).

**Điều bài báo KHÔNG công bố** (đã rà toàn văn, kể cả 3 lần grep "threshold"/"0.95"/"95%"): không có giá trị số cụ thể nào cho `p` (ngưỡng chọn bước, Eq.8) hay cho ngưỡng năng lượng tích lũy dùng để chọn `k*` trong pipeline thật (§3.1). Con số "95%" duy nhất xuất hiện trong toàn bài chỉ nằm ở **chú thích Hình 2** — mô tả một thí nghiệm minh họa khác (so sánh Reasoning-CoT vs Non-Reasoning gradient, không phải bước chọn dữ liệu huấn luyện).

## 2. Đánh giá Plan triển khai — có chính xác không?

### 2.1 Phát hiện quan trọng nhất: `p = 0.95` bị gán sai là "giá trị của bài báo"

Đây là sai lệch nghiêm trọng nhất và lặp lại **có hệ thống** xuyên suốt tài liệu và code:

| Vị trí | Trích dẫn | Vấn đề |
|---|---|---|
| `plans/.../plan.md:63,78` | "p is the paper's constant, not a gate" / "p=0.95 (paper value)" | Sai |
| `phase-04-step-selection-masks.md:13,30` | "p = 0.95 (the paper's value — fixed, not tuned; no sweep)" / "No decision gate on p — it is the paper's constant" | Sai |
| `docs/system-architecture.md:71` | bảng "Deviations from the paper": `Energy threshold p \| 0.95 \| 0.95 (paper value) \| —` | Sai, và cột "Reason" bị để trống dù đây chính là chỗ cần giải thích |
| `docs/project-changelog.md:10` | "Energy threshold p = 0.95, **the paper's published value** — the sweep and the decision gate are gone" | Sai, và ghi nhận rõ ràng việc **loại bỏ** một quyết định đúng trước đó |
| `configs/data-config.yaml:13` | `energy_threshold_p: 0.95 # paper value (Eq. 8) — fixed, not a tuned parameter` | Sai |
| `src/build_masks.py:6-8,94` | "p is the paper's published value... not a tuned hyperparameter — there is no sweep" / `print(f"p={threshold} (paper value)")` | Sai |
| `scripts/run-pipeline.sh:33` | "p=0.95 is the paper's value (not tuned)" | Sai |

Tôi đã đọc lại toàn văn bài báo (kể cả Appendix) và xác nhận: **không có chỗ nào bài báo công bố `p = 0.95`** cho Eq.8. Điều này quan trọng vì nó ảnh hưởng dây chuyền:
- Success criteria bị đặt sai: "no decision gate on p" nghĩa là dừng kiểm tra độ nhạy — trong khi đây chính xác là siêu tham số tự do duy nhất của phương pháp, không có gì để đối chiếu "đúng/sai" với bài báo.
- Che mất một quyết định *đúng và trung thực hơn* đã có sẵn trong brainstorm report (`plans/reports/brainstorm-...md:29,78`): *"Paper does NOT publish p → cheap sweep p ∈ {0.7, 0.8, 0.9}... Unresolved reproduction gap, documented."* — Plan hiện tại đã **âm thầm loại bỏ sweep này** và thay bằng một khẳng định sai, thay vì giữ cách làm minh bạch ban đầu.
- Về mặt khoa học: nếu p=0.95 gần 1.0 và phổ spectral strength tương đối phẳng (plan tự ghi nhận rủi ro này ở "Known Risks"), lựa chọn gần như giữ lại toàn bộ token → Spectral ≈ Vanilla, khiến thí nghiệm A/B mất khả năng phân biệt. Đây chính là rủi ro mà brainstorm report gốc muốn phòng bằng sweep.

**Ghi chú công bằng:** ngưỡng năng lượng dùng cho `k*` (chọn rank, không phải chọn bước) được gán nhãn cẩn thận hơn — `configs/capture-config.yaml:6` ghi "cumulative energy threshold for k* (**paper Fig 2 dashed line**)", tức quy về đúng nguồn (hình minh họa, không khẳng định là hằng số chính thức của phương pháp). Chỉ riêng `p` (Eq.8) bị thổi phồng thành "giá trị công bố".

**Khuyến nghị:** sửa toàn bộ 7 vị trí trên thành "engineering assumption, không có trong bài báo" và cân nhắc khôi phục sweep `p ∈ {0.7, 0.8, 0.9, 0.95}` như brainstorm đã đề xuất — việc này rẻ (chỉ cần thống kê step/token-drop, không cần train lại).

### 2.2 Các giả định khác — đã disclose trung thực

Những điểm sau đây bài báo không nêu chi tiết, và plan đã **nêu rõ là giả định** (không giả vờ là sự thật đã kiểm chứng) — đây là điểm tốt cần ghi nhận:
- **Segmentation**: bài báo chỉ nói "standardized step-wise segmentation" không có công thức. Plan dùng regex `r'([.?!\}\]])([\s\n]+)([A-Z])'`, tự nhận "not specified in the paper" trong `docs/system-architecture.md:72`.
- **Nguồn CoT huấn luyện**: bài báo lấy problem từ AceReason-1.1-SFT rồi **sinh lại** CoT bằng DeepSeek-R1-0528 (đã xác minh qua trang HF của dataset: câu trả lời gốc trong AceReason-1.1-SFT do **DeepSeek-R1** — không phải bản `-0528` — sinh ra). Plan/brainstorm **tái dùng trực tiếp cột `output`** có sẵn thay vì gọi lại R1-0528, và đã tự ghi nhận đánh đổi này ở brainstorm ("Regenerate via R1-0528 API — Rejected — cost; open R1-distilled data ≈ same distribution"). Đây là một xấp xỉ hợp lý về chi phí, nhưng đáng lưu ý thêm trong báo cáo kết quả (phase 7): vì phương pháp của bài báo đặt trọng tâm vào việc phân biệt "bước lý luận thật" và "bước dư thừa" trong CoT, một checkpoint teacher khác (R1 gốc thay vì R1-0528) có thể cho phân phối độ dài/độ dư thừa bước hơi khác, ảnh hưởng đến baseline drop-ratio quan sát được.
- **Quy mô**: 1 GPU thay 8×L20, Qwen3-1.7B thay 4 model 4B–8B, 2k thay 10k mẫu, cutoff 16k thay 32k, Custom Trainer thay LLaMA-Factory — tất cả đã liệt kê tường minh trong bảng "Deviations from the paper".

### 2.3 Điểm khớp tốt với bài báo

- Per-sample SVD (không phải SVD toàn corpus) — khớp đúng cách đọc §3.2 ("within each sample").
- Công thức spectral strength dùng leverage score thuần từ U (không nhân Σ) — khớp chính xác Eq.7 và đoạn văn "this energy is intrinsically encoded in the row norms of... U".
- Hyperparameter huấn luyện (`train-vanilla.yaml`, `train-spectral.yaml`) khớp đúng Table 3 của bài báo (lr 5e-5, min_lr 1e-5, warmup 0.1, 6 epoch, `cosine_with_min_lr`).
- Eval setup (`eval-config.yaml`) khớp đúng §4.1 (temp 0.6, top-p 0.95, n=4, max 32768 token).
- `d = 2048` cho Qwen3-1.7B đã xác minh đúng qua `config.json` thật trên HuggingFace (`hidden_size: 2048`, `tie_word_embeddings: true` — khớp ghi chú trong phase-03 về việc dùng `get_output_embeddings().weight`).
- Schema AceReason-1.1-SFT (`category`/`source`/`input`/`output`) đã xác minh đúng qua HF dataset viewer thật.

## 3. Đánh giá Code — có chính xác/khoa học không?

### 3.1 Kiểm tra công thức, module theo module

| Module | Đối chiếu công thức | Kết luận |
|---|---|---|
| `gradient_utils.py` | `g = W_U^T(softmax(W_Uh) - onehot(y))` = d(CE)/dh đúng theo Eq.1; shift nhân quả `h[j-1]` dự đoán token `j` đúng | **Đúng**, có test `test_hidden_states_are_post_norm_so_recomputed_logits_match` xác nhận `last_hidden_state @ W_out^T == logits` trên model thật (kiến trúc Qwen3) |
| `spectral_utils.py` | `cumulative_energy`, `effective_rank` (k* = min k: E(k)≥cutoff), `token_leverage_scores`, `step_spectral_strengths` | **Đúng**, khớp Eq.4 và Eq.7 chính xác từng bước |
| `step_selection.py` | `select_steps_by_energy` = arg min |S'| s.t. tổng dồn ≥ p·tổng, sort giảm dần, tie-break theo index | **Đúng**, khớp Eq.8 |
| `masked_loss.py` | `sum(CE)/Z`, Z mặc định = số token chưa mask, cho phép truyền Z ngoài | **Đúng**, khớp Eq.9 |
| `data_collator.py`, `train_sft.py` (`MaskedSFTTrainer`) | Z gộp qua toàn bộ optimizer step (mọi microbatch grad-accum) qua `num_items_in_batch` | **Đúng về mặt thiết kế**, xem 3.2 để biết rủi ro phụ thuộc phiên bản |
| `segmentation.py`, `data_prep.py` | Heuristic tách câu + tokenize incremental để giữ đúng token offset | Hợp lý, tự nhận là giả định (không phải công thức bài báo) |
| `benchmarks.py`, `answer_scoring.py`, `evaluate.py`, `compare_results.py` | Loader 5 benchmark, trích `\boxed{}`, so khớp math-verify, MCQ shuffle cố định seed | Hợp lý, không có claim sai lệch với bài báo |

### 3.2 Rủi ro kỹ thuật đã xác minh được (không phải suy đoán)

**Chuẩn hóa loss phụ thuộc hành vi nội bộ của `transformers.Trainer` chưa được ghim phiên bản chặt.** Tôi đã đọc trực tiếp mã nguồn `trainer.py` (bản 5.14.1) để kiểm chứng thiết kế "một `sum/Z` cho mỗi optimizer step":
- `Trainer._get_num_items_in_batch` chỉ đếm `labels[..., 1:]` (dịch nhãn đi 1 vị trí để không đếm dư vị trí 0) khi `self._loss_shifts_labels` là `True`. Thuộc tính này được `Trainer.__init__` tự suy ra từ `model.loss_type == "ForCausalLMLoss"`, và `MaskedSFTTrainer` gán đè lại `= True` một cách tường minh — hợp lý và đúng cho Qwen3.
- `Trainer.training_step` **bỏ qua** việc chia thêm cho `gradient_accumulation_steps` đúng khi `model_accepts_loss_kwargs=True` và `num_items_in_batch is not None` — khớp với thiết kế "không double-normalize" mà `train_sft.py` giả định.
- Cả hai cơ chế trên **đã xác nhận tồn tại và đúng như mô tả trong docstring** ở bản 5.14.1. Nhưng `requirements.txt` chỉ ghim sàn `transformers>=4.56`, không có trần, và không có bằng chứng các cơ chế này (đặc biệt `_loss_shifts_labels` dịch nhãn khi đếm) đã tồn tại từ đúng bản 4.56 — đây là tính năng tinh vi, khả năng cao được thêm ở bản mới hơn. Nếu server cài đúng bản sàn 4.56 mà thiếu cơ chế dịch nhãn này, `num_items_in_batch` sẽ đếm dư đúng 1 vị trí/sequence trong mẫu số — sai số nhỏ (không đáng kể trên chuỗi 16k token) nhưng phá vỡ khẳng định "khớp chính xác Eq.9".
- **Khuyến nghị cụ thể**: thêm một assertion khởi động trong `train_sft.py` (`assert trainer._loss_shifts_labels is True`) hoặc ghim phiên bản `transformers` chính xác đã test, thay vì chỉ ghim sàn `>=4.56`. Việc "48 unit test xanh" không bắt được lỗi này vì test chỉ gọi `masked_cross_entropy` trực tiếp, không đi qua `Trainer._get_num_items_in_batch` thật.

**Các điểm nhỏ, không phải lỗi chức năng:**
- `effective_rank`: nếu `energy_cutoff = 1.0` chính xác và giá trị năng lượng tích lũy cuối cùng lệch dưới 1.0 do làm tròn dấu phẩy động, `torch.searchsorted` trả về chỉ số ngoài phạm vi (`r`), khiến `k_star = r+1`; việc cắt lát `U[:, :k_star]` trong PyTorch tự động giới hạn về `r` cột nên không crash, nhưng giá trị `k_star` ghi log có thể sai 1 đơn vị. Không ảnh hưởng ở cutoff mặc định 0.95.
- Comment `requirements.txt`: "`transformers>=4.56` # Qwen3 support" — Qwen3 thực ra đã được hỗ trợ từ 4.51 (xác nhận qua `config.json` gợi ý `transformers_version: 4.51.0`); lý do thật của việc ghim ≥4.56 nhiều khả năng là cú pháp `dtype=` thay `torch_dtype=` (đã xác minh việc deprecate này là có thật), không phải hỗ trợ Qwen3. Chỉ là comment sai, không ảnh hưởng chức năng.

### 3.3 Chất lượng test

Bộ test (`test_spectral_utils.py`, `test_step_selection.py`, `test_masked_loss.py`, `test_gradient_utils.py`, `test_segmentation.py`, `test_answer_scoring.py`) được thiết kế tốt: test cả tính chất toán học (leverage scores cộng dồn đúng bằng k*, cumulative energy đơn điệu tăng), cả tính bất biến quan trọng (mask toàn 1 ⇒ loss = CE chuẩn; gradient giải tích khớp autograd trên model Qwen3 thật dù nhỏ). Đây là các test đúng chỗ, không phải test hình thức. Tôi không chạy lại được (sandbox không có GPU/torch, và cài `torch` qua pip bị chặn proxy tới `download.pytorch.org`; bản CPU đầy đủ từ PyPI mặc định kéo theo các gói `nvidia-*` vượt quá dung lượng đĩa còn trống của sandbox) — đánh giá "đúng" ở đây dựa trên việc đọc code và tự tính tay, không phải chạy lại thực nghiệm.

## 4. Kết luận

Về **kiến trúc pipeline và các công thức cốt lõi** (Eq.1, 2, 4, 7, 8, 9), code triển khai chính xác so với bài báo, có test hợp lý bảo vệ các bất biến quan trọng, và các quyết định thiết kế mở (segmentation, nguồn CoT, quy mô model/data) đều được disclose rõ ràng là giả định thay vì giả vờ là sự thật. Đây là một reproduction plan được làm cẩn thận ở phần lõi.

Vấn đề nghiêm trọng nhất không nằm ở toán học hay code, mà ở **tài liệu hóa sai một giả định thành sự thật đã được nghiệm chứng**: `p = 0.95` bị gọi là "giá trị công bố của bài báo" tại 7 vị trí (plan, phase spec, 2 file docs, 1 config, 1 module code, 1 script) trong khi bài báo không công bố giá trị này ở bất kỳ đâu — và một hướng xử lý đúng hơn (sweep p, đã có sẵn trong brainstorm) đã bị âm thầm loại bỏ. Vì đây là siêu tham số then chốt duy nhất quyết định sự khác biệt giữa hai lần train A/B, sai lệch tài liệu này có nguy cơ dẫn đến kết luận khoa học sai (Spectral ≈ Vanilla do p quá cao, bị hiểu nhầm là "method không có hiệu quả" thay vì "p chưa được chọn phù hợp").

## 5. Khuyến nghị hành động

1. Sửa 7 vị trí liệt kê ở §2.1 để mô tả đúng: `p=0.95` là lựa chọn kỹ thuật của người triển khai, không phải giá trị công bố.
2. Khôi phục kế hoạch sweep `p ∈ {0.7, 0.8, 0.9, 0.95}` từ brainstorm report (chỉ cần thống kê drop-ratio, không cần train lại) trước khi chạy 2 lần train tốn GPU.
3. Thêm assertion/test xác nhận hành vi `_loss_shifts_labels`/`num_items_in_batch` của `transformers.Trainer` đúng như giả định trước khi tin tưởng "một sum/Z mỗi optimizer step" trên server thật, hoặc ghim đúng phiên bản đã kiểm thử thay vì chỉ ghim sàn.
4. Khi viết Phase 7 report, ghi rõ thêm rằng CoT huấn luyện dùng lại từ DeepSeek-R1 (gốc) thay vì sinh lại bằng R1-0528 như bài báo, như một giới hạn về độ trung thực tái lập.
