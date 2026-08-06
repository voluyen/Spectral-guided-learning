# Method Fidelity Audit — code vs paper

Đối chiếu từng module `src/` với `Uncovering_the_Gradient.pdf` (Eq. 1-9, §3, §4.1, Appendix A).
Ngày: 2026-08-06. Trạng thái code: commit `0cf0a7b`, 56 tests pass.

## Kết luận ngắn

Logic method **khớp paper**. Eq. 1-9 implement đúng, không có lỗi công thức. Vấn đề nằm ở
**tham số và nguồn dữ liệu**, không phải ở thuật toán — trong đó có 1 điểm cần bạn quyết định
(mục F1).

---

## A. Khớp paper (đã verify)

| Paper | Code | Ghi chú |
|---|---|---|
| Eq. 1 `g_t = ∇_h ℓ_t = W_U^T(softmax(W_U h) − onehot(y))` | `gradient_utils.analytic_hidden_gradients` | verify vs autograd: max diff 1.2e-7 (float32) |
| Eq. 2 `G = [g_1..g_T]^T ∈ R^{T×d}` | `capture_sequence_gradients` | chỉ stack token của response (supervised positions) — đúng |
| Eq. 3 SVD `G = UΣV^T` | `spectral_utils.analyze_gradient_matrix` | float32, `full_matrices=False` |
| Eq. 4 `E(k) = Σσ²/Σσ²`, k\* tại ngưỡng | `cumulative_energy` + `effective_rank` | cutoff 0.95 = đường 95% ở Figure 2 ✓ |
| Eq. 7 `S(s) = (1/\|s\|) Σ ‖(U_{1:k*})_t‖²` | `token_leverage_scores` + `step_spectral_strengths` | mean leverage trên token của step ✓ |
| Eq. 8 minimal set, cumulative ≥ p | `step_selection.select_steps_by_energy` | sort desc, tie theo index, luôn giữ ≥1 step |
| Eq. 9 `L = −(1/Z)ΣM_t log P` | `masked_loss.masked_cross_entropy` | xem F3 về Z |
| §3.1 "using the base model" | `capture-config.model_name = Qwen3-1.7B-Base` | gradient capture bằng trọng số khởi tạo của student ✓ |
| §3.3 "low-strength steps vẫn tham gia forward" | mask nằm ở `labels=-100`, không cắt input | context nguyên vẹn ✓ |
| Table 3 HP | `configs/train-*.yaml` | lr 5e-5, cosine_with_min_lr, min 1e-5, warmup 0.1, 6 epochs, global batch 32 — khớp từng dòng |
| §4.1 eval | `configs/eval-config.yaml` | temp 0.6, top-p 0.95, n=4, max 32768 — khớp; đủ 5 benchmark |

Alignment token cũng nhất quán giữa 2 pha: capture dùng `hidden[i−1]` để dự đoán token `i`;
training shift `logits[:, :−1]` vs `labels[:, 1:]`. Cùng một cặp (state, target).

---

## B. Deviation có chủ đích (đã ghi trong `docs/system-architecture.md`)

Model 1.7B (paper: 4B-8B) · data 2k (10k) · cutoff 16k (32k) · HF Trainer (Llama-Factory) ·
segmentation bằng regex (paper chỉ nói "standardized step-wise segmentation", không mô tả).

---

## C. Findings

### F1. `p = 0.95` KHÔNG phải giá trị của paper — cần quyết định lại

Paper **không công bố p** ở bất kỳ đâu: Table 3 (Appendix A) liệt kê đủ 7 hyperparameter, không
có p; §3.2 chỉ viết "a preset energy threshold p".

Con số 0.95 xuất hiện trong paper là **energy cutoff để chọn k\*** (Figure 2 "95% energy",
§3.1 "Truncate at Threshold") — đây là ngưỡng của Eq. 4, **khác** với p của Eq. 8:

- `capture-config.yaml: energy_cutoff: 0.95` → k\*, đúng là giá trị của paper ✓
- `data-config.yaml: energy_threshold_p: 0.95` → p của Eq. 8, **là lựa chọn của chúng ta**

Hai ngưỡng này độc lập về mặt toán học. Comment hiện tại trong `data-config.yaml`
("paper value (Eq. 8) — fixed, not a tuned parameter") **sai sự thật** và cần sửa.

Hệ quả thực tế: p=0.95 rất bao trùm → giữ gần hết step → mask spectral ≈ mask vanilla → thí
nghiệm A/B mất ý nghĩa. Bằng chứng gián tiếp từ paper: Table 2 cho thấy "Static Ratio Selection"
(giữ tỉ lệ cố định) vẫn thắng Vanilla 1.2%, và dynamic thắng 2.4% — tức lượng step bị loại là
đáng kể, không phải vài %.

### F2. Nguồn CoT khác paper

Paper (§4.1 Training Data): lấy 10.000 **problem** từ AceReason-1.1-SFT rồi **dùng
DeepSeek-R1-0528 sinh lại** CoT trajectories.

Code: dùng thẳng field `output` có sẵn của AceReason-1.1-SFT. Không sai method, nhưng trajectory
đến từ teacher khác → độ dài, phong cách, mật độ redundancy khác. Rẻ hơn nhiều (không tốn
inference R1). Nên ghi vào bảng deviation thay vì im lặng.

### F3. Eq. 9 normalization: code dùng Z theo batch, paper viết theo sequence

Eq. 9 viết cho **một sequence**: `Z = Σ_{t=1..T} M_t`. Code chuẩn hoá theo Z gộp toàn bộ
microbatch của một optimizer step (`num_items_in_batch`).

Đây là quyết định trong phase-05 plan và trùng với hành vi mặc định của Llama-Factory/HF
(token-level mean toàn batch) — tức nhiều khả năng giống paper *thực tế chạy* hơn là công thức
viết trong paper. Giữ nguyên, nhưng nên ghi rõ là diễn giải, không phải bám chữ.

### F4. Eq. 7 (leverage) vs §2.2 (projection energy) — mâu thuẫn nội tại của paper

§2.2 nói spectral strength đo `‖g_t^∥‖²`, nhưng Eq. 7 định nghĩa bằng row norm của U
**không có trọng số σ**. Hai đại lượng khác nhau: `‖g_t^∥‖² = Σ_{i≤k} σ_i² U_{t,i}²` vs
`leverage = Σ_{i≤k} U_{t,i}²`.

Code theo **Eq. 7** (định nghĩa formal) — lựa chọn đúng. Kiểm tra số trên ma trận ngẫu nhiên:
tương quan Pearson 0.98, top-4 token trùng khớp → khác biệt nhỏ trong thực tế. Không cần sửa.

### F5. Chưa có ablation "Residual Selection"

Table 2/4 của paper dùng 3 baseline: Random, Static Ratio, Residual. Plan cố ý bỏ (scope = method
only). Nhưng **Residual Selection** (train trên các step spectral strength *thấp*) chỉ cần đảo
điều kiện chọn (~10 dòng) và là bằng chứng rẻ nhất cho thấy metric có ý nghĩa: paper báo residual
tệ hơn cả Vanilla (−3.6%). Nếu ở scale của mình residual ≈ spectral thì kết luận A/B vô giá trị.

### F6. Minor — `effective_rank` có thể trả k\* = r+1

`torch.searchsorted(energy, cutoff) + 1`: nếu sai số làm `energy[-1] < cutoff`, k\* vượt rank thật.
Ở cutoff 0.95 không xảy ra (energy[-1] = 1.0). Chỉ thành vấn đề nếu ai đó set cutoff ≈ 1.0.
Clamp `min(k*, len(singular_values))` là đủ.

---

## Việc nên làm, theo thứ tự

1. Sửa comment sai trong `data-config.yaml` (F1) — 1 dòng, bắt buộc vì đang ghi sai sự thật.
2. Quyết định p (F1). Không có giá trị paper để bám → hoặc sweep, hoặc chọn theo drop ratio mục tiêu.
3. Ghi F2 + F3 vào bảng deviation của `docs/system-architecture.md`.
4. Cân nhắc F5 trước khi chạy train (thêm run thứ 3, cùng data, chỉ đảo mask).
5. F6 clamp — vặt.

## Câu hỏi chưa giải quyết

- Chọn p bao nhiêu? Paper không cho. Đề xuất: chạy `capture` trên 2k thật, xem phân phối spectral
  strength, rồi chọn p sao cho token-drop rơi vào 30-50% (khoảng mà "fewer tokens yet better" của
  paper có ý nghĩa). Hoặc sweep {0.7, 0.8, 0.9} như plan ban đầu.
- Có sinh lại CoT bằng R1 không (F2)? Tốn inference nhưng sát paper hơn.
- Có chạy thêm run Residual (F5) không? +10-20h GPU.
