---
title: "Phân tích sâu 3 idea nghiên cứu từ họp 08-02 (Spectral-guided multi-response distillation)"
date: 2026-08-06
inputs: "meeting/08-02.txt + Uncovering_the_Gradient.pdf (Eq.1-9, Appendix D) + pipeline repo hiện có"
scope: "Đánh giá cơ chế / novelty / khả thi / rủi ro của Idea 1 (multi-response + ranking), Idea 2 (reward từ learnability của student), Idea 3 (đo đa dạng) + đề xuất tổng hợp thành 1 paper."
related: "[[meeting-analysis-260806-1202-spectral-learning-reqs-and-next-ideas-report]]"
---

# Phân tích sâu 3 idea

## 0. Nền chung — primitive bắt buộc phải xây trước

Cả 3 idea đều dựa trên **cùng một mở rộng**: paper gốc tính spectral strength ở **mức step, trong 1 response** (1 ma trận `G ∈ R^{T×d}` / chuỗi, SVD per-sample). Cả 3 idea đều cần **so sánh/chọn GIỮA nhiều response** của cùng 1 bài toán → cần một **"response-level spectral signature"** và cách **chuẩn hóa** (vì mỗi response có `T`, `k*` khác nhau).

→ **Đây là khối enabling chung. Xây nó trước, cả 3 idea mới chạy được.** (Chính là câu hỏi mở #6 ở báo cáo họp.)

Ứng viên cho response-level signature (cần thực nghiệm chọn):
- `S_resp = mean/median` leverage score các step **được chọn** của response (đơn giản, tái dùng `step_strengths` đã có).
- **Tổng năng lượng consensus** response chiếm được: `Σ_{i≤k*} σ_i² / Σ σ_j²` — nhưng cái này gần như bằng đúng ngưỡng `p` nên **ít phân biệt** → cân nhắc bỏ.
- **Chữ ký gradient** của response: vector đặc trưng (vd trung bình các `g_∥` đã chiếu vào consensus) → dùng cho đo tương đồng chéo response (Idea 3).

Repo hiện có: `gradient_capture.py` đã sinh `step_strengths`, `k_star`, `singular_values` per-sample. Mở rộng = (a) stage sinh **G response từ teacher** (giống `evaluate.py` gọi vLLM, nhưng trên tập train), (b) gom `step_strengths`→`S_resp`. **Đường đi cụ thể, không phải làm lại từ đầu.**

---

## 1. Idea 1 — Multi-response + spectral selection + ranking (DPO)

**Ý tưởng:** teacher sinh **G response** / bài (kiểu rollout GRPO nhưng để **distill**). Dùng spectral strength để (a) chọn step quan trọng **trong** mỗi response (y hệt paper), (b) **rank các response**; thêm **DPO ranking loss** (đúng đáp án = winning, sai = losing).

### Cơ chế đề xuất cụ thể
1. Teacher gen G response; **lọc theo đáp án đúng** trước (correct → tập positive để SFT).
2. Trong mỗi response giữ lại: áp **masked spectral SFT** (Eq.9) như paper.
3. Cross-response: tính `S_resp` → **weighting** loss theo response (distillation có trọng số) **thay cho** DPO cứng.
4. (Tùy chọn) DPO giữa correct (winning) vs incorrect (losing) như loss phụ.

### Novelty
- Trục mới so paper: **1 teacher CoT → G teacher CoT**. Nhưng "gen G rồi distill cái tốt" **trùng ý** với **Rejection-sampling FT / Best-of-N distillation** (đã biết) và **preference distillation**.
- ⇒ Novelty **phải nằm ở**: tín hiệu **spectral/geometric** chọn tốt hơn lọc-theo-đúng-sai hoặc perplexity; **không phải** ở chỗ "gen nhiều".

### Rủi ro / thách thức
- **DPO là bolt-on** — chính thầy nói "ranking chỉ bổ sung, không cốt lõi". Reviewer dễ nhìn ra. DPO trên long-CoT **bất ổn** (length bias, KL blow-up).
- **Teacher mạnh ⇒ đa số chuỗi đúng** ⇒ **thiếu negative** ⇒ DPO signal yếu.
- **Chi phí ×G**: nhân toàn bộ pipeline gradient+SVD (vốn đã đắt) với G. Phải chặn `G` nhỏ (4–8).
- Chuẩn hóa `S_resp` giữa response khác độ dài — chưa có công thức chuẩn.

### Đánh giá
**Khả thi cao nhất, novelty mong manh nhất.** Nên dùng làm **bộ khung** nhưng phải chứng minh rõ spectral > correctness-only. Ưu tiên **weighted distillation** (đơn giản, ổn định) hơn DPO; DPO để phần phụ.

---

## 2. Idea 2 — Reward từ "learnability" của student

**Ý tưởng:** tìm đoạn reasoning quan trọng (spectral) → **đưa làm gợi ý cho student** → nếu student **tự suy ra đúng đáp án** với gợi ý đó → đoạn đó **reward cao** (dựng reward model / cặp positive-negative DPO).

### Điểm cốt lõi (vì sao khác Idea 1)
- Spectral strength đo trên **student-lúc-khởi-tạo** = tính chất **hình học tĩnh** (geometry), **không** phải kết quả hành vi.
- Idea 2 đo **learnability thực**: đoạn có giá trị nếu **giúp student ra đáp án đúng** — tín hiệu **outcome-based**, khác chất.
- ⇒ **Đóng vòng** giữa tín hiệu geometric (rẻ, offline) và outcome (student giải được). Một bài "spectral strength có **dự đoán được** learnability của student không?" là **insight-driven** — đúng gu thầy muốn (không black-box).

### Liên hệ literature
- **Learnability / reducible loss** (RHO-LOSS), **influence functions**, **on-policy distillation**, hint-based curriculum. Reward-từ-đúng-sai mang màu **RL**.

### Rủi ro / thách thức
- **Rất đắt**: cần **student rollout** cho mỗi ứng viên gợi ý, ×n mẫu (reward nhiễu do student stochastic).
- **Credit assignment**: response có nhiều step — step nào gây ra thành công? Đây là **bài toán nghiên cứu khó**.
- Nếu làm per-segment → bùng nổ chi phí.

### Cơ chế đề xuất (giảm chi phí)
1. **Thí nghiệm kiểm chứng trước** (rẻ, gần như đã là claim của paper): student học **spectral-selected steps** có giải đúng nhiều hơn học **random / PPL-selected** không? A/B ở **mức mask**, không per-segment.
2. Nếu đúng: dùng **student pass@k** (với tập step chọn làm prefix/hint) làm reward để **re-rank / re-weight** lựa chọn spectral — tầng tinh chỉnh thứ 2.
3. Framing: **spectral = proxy rẻ; student-success = gold đắt** → dùng gold để hiệu chỉnh/validate proxy, hoặc học predictor rẻ.

### Đánh giá
**Novelty/insight cao nhất, nhưng đắt & mơ hồ nhất.** Dùng làm **phần validation + insight** của paper hơn là toàn bộ method. Credit assignment là chỗ dễ sa lầy — giữ ở mức **mask-level A/B**, đừng per-token.

---

## 3. Idea 3 — Đo đa dạng (diversity) của G response

**Ý tưởng:** đo độ đa dạng giữa G response (qua gradient / cosine giữa các `G`); chọn tập con đa dạng để dạy student; tránh bias/trùng lặp.

### Phản biện của thầy (đúng) — và cách hóa giải
- thầy: "**đo đa dạng để làm gì** còn chưa rõ" — đo suông = chỉ ra 1 cái plot, vô dụng.
- Hóa giải: **biến diversity thành ENGINE chọn tập con**, không phải metric quan sát. → **DPP (Determinantal Point Process)** hoặc **facility-location submodular** trên chữ ký gradient response: chọn `k` response **tối đa hóa (chất lượng × phủ đa dạng)**. Đây là công cụ chuẩn cho "chọn tập con vừa tốt vừa đa dạng", **có nền lý thuyết** (định thức = thể tích = đa dạng) — đúng gu insight/theory.

### Tái dùng chính machinery của paper
- Paper **đã** tính cosine chéo-sample của gradient (Fig.4: consensus dương, residual ~0). Idea 3 = **áp đúng Fig.4 nhưng giữa G response của CÙNG 1 bài**, thay vì giữa các bài khác nhau. → **Tái sử dụng gọn**, không phát minh mới.
- Diversity metric = `1 − mean pairwise cosine` của chữ ký gradient (đã chiếu consensus) giữa G response.

### Rủi ro
- **Đa dạng ≠ chất lượng**: chuỗi đa dạng-nhưng-sai gây hại → **phải kết hợp correctness + spectral quality**, không đa dạng thuần.
- Đo bằng full gradient+SVD trên G response = **đắt**; bản rẻ = cosine trên embedding đáp án/chiến lược (nhưng mất tính "geometric").

### Đánh giá
**Yếu nếu đứng riêng, mạnh nếu là COMPONENT.** Chính là "engine chọn tập con" cho Idea 1 → biến critique của thầy thành đóng góp.

---

## 4. So sánh & tổng hợp

| Tiêu chí | Idea 1 (multi-resp + DPO) | Idea 2 (learnability reward) | Idea 3 (diversity) |
|---|---|---|---|
| Novelty | Thấp–TB (trùng RFT/best-of-N) | **Cao** (insight) | Thấp riêng lẻ / **TB** khi là engine |
| Khả thi | **Cao** | Thấp (đắt, credit assignment) | TB |
| Chi phí | ×G pipeline | ×G ×rollout ×n (đắt nhất) | ×G gradient |
| Rủi ro chính | DPO bolt-on, thiếu negative | reward nhiễu, credit assignment | đa dạng ≠ chất lượng |
| Nền lý thuyết sẵn | Appendix D (variance ↓ k*/d) | learnability/influence | DPP (định thức), Fig.4 |
| Vai trò đề xuất | **Bộ khung** | **Validation + insight** | **Engine chọn tập con** |

### Đề xuất: ghép thành 1 paper mạch lạc — "Diverse Spectral Distillation"
Một khung thống nhất, mỗi idea giữ 1 vai:
1. **Khung (Idea 1):** teacher gen G response → spectral masking trong từng response → distill có trọng số.
2. **Engine chọn tập con (Idea 3):** DPP trên chữ ký gradient (consensus-projected) chọn tập response **đa dạng + chất lượng cao** → giải quyết critique "đo đa dạng để làm gì".
3. **Insight/validation (Idea 2):** chứng minh spectral selection **dự đoán learnability** của student (A/B mask-level); dùng student-success để hiệu chỉnh trọng số.

**Hook lý thuyết** (thầy muốn insight): gộp consensus subspace qua G response → ước lượng consensus **robust hơn**, variance giảm thêm ~`1/G` (nối tiếp Appendix D); DPP cho nền đa dạng bằng định thức. → paper có "insight bên dưới" chứ không black-box.

---

## 5. Gating khả thi (phải làm trước khi cam kết scale)

1. **Đo chi phí SVD/gradient** (R3 của báo cáo họp) — pipeline ×G có tractable không. Nếu đắt: dùng **truncated/randomized SVD** (`torch.svd_lowrank`) vì chỉ cần top-`k*` (lưu ý: cumulative energy Eq.4 cần **toàn bộ** σ ở mẫu số → randomized SVD phải ước lượng mẫu số, cần kiểm định sai số).
2. **Gap teacher–student đủ lớn** (lo ngại lặp lại của thầy): teacher Qwen lớn (32B/2.5) → student nhỏ (0.6–7B); chọn **bộ toán khó** nơi gap rõ (~10% chênh phương pháp). Gap nhỏ ⇒ multi-response distillation lợi ích biên.
3. **Chặn G nhỏ** (4–8): sinh quá nhiều → đáp án trùng, chi phí bùng, lợi ích bão hòa (thầy: "sinh nhiều quá cũng không tốt").
4. **Chuẩn hóa `S_resp`** giữa response khác độ dài — cần thực nghiệm chốt trước khi rank/weight.

---

## 6. Câu hỏi mở / cần quyết định

1. **Response-level signature** dùng công thức nào (mean-leverage vs chữ ký gradient)? — quyết định primitive chung §0.
2. **Weighted distillation vs DPO**: chọn cái nào làm chính? (khuyến nghị: weighted làm chính, DPO phụ.)
3. **Idea 2 credit assignment** giữ ở mask-level hay cố làm per-step? (khuyến nghị: mask-level để không sa lầy.)
4. **DPP có quá nặng** cho chọn tập con G nhỏ không? Với G=4–8 có thể greedy submodular thay cho DPP đầy đủ.
5. **Teacher/student/dataset cụ thể** — chờ Phong khảo sát; quyết định gap.
6. **Randomized SVD** có giữ đúng Eq.4 (mẫu số toàn phổ) trong sai số chấp nhận được không? — cần benchmark nhỏ.
7. Phạm vi paper: làm **cả 3 vai** (tham vọng, rủi ro thời gian 2 tuần) hay **chỉ Idea 1 + validation Idea 2**, để Idea 3/DPP cho vòng sau?
