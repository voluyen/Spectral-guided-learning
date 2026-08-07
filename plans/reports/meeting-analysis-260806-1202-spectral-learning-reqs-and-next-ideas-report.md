---
title: "Báo cáo cuộc họp 08-02 — Spectral-guided Learning: chốt hiểu paper, yêu cầu tái lập & định hướng đề tài"
date: 2026-08-06
source: meeting/08-02.txt (transcript ASR tiếng Việt, ~66 phút, 758 dòng)
participants: "4 người — Thầy (chủ trì) + Sang, Luyện, Phong"
scope: "Trích xuất yêu cầu chi tiết + các trao đổi trong họp, đối chiếu với paper Uncovering_the_Gradient.pdf và pipeline tái lập hiện có trong repo"
note: "Đã chuẩn hóa lỗi ASR về đúng thuật ngữ: cô T→CoT, DeepSeed R1→DeepSeek-R1, publicity/PurpleXCT/Purple Lack City→perplexity, special strength/screen→spectral strength, loa run/low run→low-rank, level score→leverage score, GIPO/GAPO→GRPO, điếp xích→DeepSeek, tiêu đình→student lúc khởi tạo, thiêu thờ→bài GRPO thầy gửi."
---

# Cuộc họp 08-02 — Spectral-guided Learning: yêu cầu & trao đổi

## 0. Bối cảnh cuộc họp

Buổi seminar nhóm **4 người: Thầy, Sang, Luyện, Phong**. Sang trình bày paper **"Uncovering the Gradient Geometry of Long CoT: A Spectral-guided Approach to Reasoning Distillation"** (ICML 2026). Thầy hỏi sâu để làm rõ cơ chế, đánh giá độ khả thi/chi phí, rồi đề ra định hướng phát triển đề tài mới dựa trên paper này (ghép thêm 1–2 paper khác). Cuối họp giao việc cho nhóm. Có nhắc buổi trước Luyện trình bày 1 paper safety-alignment (Geometric/hidden-state editing) nhưng thầy đánh giá hướng Spectral tiềm năng hơn → tập trung vào đây.

Ngoài paper chính, thầy nhắc **3 paper cần ghép/tham khảo**:
- Paper Spectral-guided Learning (paper chính, đang tái lập).
- 1 paper **GRPO sinh nhiều chuỗi + tìm token/đoạn quan trọng để rút ngắn reasoning** ("bài thầy gửi", áp dụng cho GRPO) — thầy đã gửi lên nhóm.
- 1 paper **Long-Context / rút ngắn chuỗi CoT từ teacher** (nhóm khác từng trình bày) — cân nhắc kết hợp.

---

## 1. Chốt hiểu về paper (các điểm được làm rõ trong họp)

Đây là phần Q&A giữa thầy và nhóm, đã thống nhất cách hiểu:

| Câu hỏi trong họp | Kết luận đã chốt |
|---|---|
| Gradient là gradient **của cái gì**? | Gradient của **loss theo hidden representation `h_t`** (không phải gradient tham số), tính bằng **student lúc mới khởi tạo** (chưa fine-tune) — tương tự ý "chỗ nào model chưa chắc chắn thì là chỗ cần học". |
| `h` là biểu diễn ở **tầng nào**? | **Paper KHÔNG nói rõ** ("trong bài không thấy đề cập rõ") → **câu hỏi mở**, cần kiểm tra khi tái lập. |
| SVD làm **chung toàn corpus** hay **từng sample**? | **Từng sample**: mỗi chuỗi có 1 ma trận `G ∈ R^{T×d}` (T token × d chiều), SVD riêng từng `G`. Không phải SVD gộp. |
| Cách **tách step**? | Dựa **biểu thức chính quy trên dấu câu** (theo OpenReview): 1 câu ≈ 1 step. Tới dấu chấm thì cắt. |
| Chọn `k*` thế nào? | `k*` = số chiều nhỏ nhất để **cumulative energy E(k) đạt 95%**. Trên Fig.2: reasoning-CoT đạt 95% với `k*` rất nhỏ (low-rank), non-reasoning thì cần nhiều chiều hơn. |
| `spectral strength` là gì? | Thực chất là **leverage score** (Eq.7), nhưng chỉ trên `k*` cột đầu của U. Cảm hứng từ leverage score trong hồi quy (đo mẫu nào ảnh hưởng mạnh tới dự đoán). |
| Chọn step (Eq.8) hoạt động ra sao? | Rank các step theo spectral strength **giảm dần**, chọn tập con tối thiểu `S'` sao cho tổng strength ≥ `p`·tổng. Cho phép chọn **rời rạc** (step 1,3,8 quan trọng; 2,4,5 bỏ) — không cần liên tục. |
| Step bị loại có bị bỏ khỏi input? | **Không.** Forward vẫn đưa vào (giữ ngữ cảnh/ngữ nghĩa); chỉ **backward** mới chặn — gradient các token đó bị đặt = 0 (masked loss Eq.9). |
| Đây có phải distillation? | **Có — black-box distillation.** Từ chuỗi CoT của teacher, chọn step nào đưa cho student học. |
| Bài có **theory** không? | Rất ít. Appendix chỉ chứng minh vài tính chất (consensus subspace giảm variance ~k*/d). Idea thiên về **novelty/insight**, không nặng lý thuyết. |

### Đánh giá chi phí (mối lo lặp lại trong họp)
- Pipeline **tốn kém**: phải **tính gradient trên từng step** rồi **SVD trên từng sample** → nặng hơn nhiều so với perplexity (perplexity chỉ cần 1 forward, không cần gradient/SVD).
- Paper **công bố ~20h** cho SFT/training, nhưng **KHÔNG công bố thời gian chạy SVD** (điểm cần tự đo khi tái lập).
- Quy mô paper: **4 model Qwen** (4B-Base, 8B-Base, 4B-Instruct, 7B) trên **8×L20 GPU**; teacher = **DeepSeek-R1**; ~1000–2000 sample cho phần phân tích spectral (200/1000 sample cho các thí nghiệm phụ).
- **Dữ liệu & code KHÔNG public** — nhưng OpenReview đánh giá cao. OpenReview còn có thực nghiệm thêm **GPT-OSS-120B** (reviewer đòi đa dạng kiến trúc).
- Đánh giá chỉ so với **các phương pháp SFT khác** (PPL/entropy filtering, TDST…), **không so trực tiếp với teacher**, không đo gap teacher–student.

---

## 2. Yêu cầu chi tiết được giao (phần "cần làm")

### 2.1 Tái lập & kiểm chứng paper (ưu tiên trước — giao Sang & Luyện làm code)
Mục tiêu: xác nhận **spectral strength có thật sự tìm ra đoạn quan trọng không** và **có tốn kém tới mức nào**.

- **R1 — Cài đặt & tính spectral strength** trên dữ liệu có CoT sẵn.
- **R2 — Visualize** kết quả spectral strength sau khi tính (giống Fig.3 của paper: vùng strength cao vs thấp; đối chiếu với perplexity).
- **R3 — Đo thời gian**: xác nhận chi phí tính gradient + SVD trên từng step/sample (paper giấu con số này) → biết pipeline có "quốc" (khả thi) không.
- **R4 — Kiểm định tính đúng**: các step được chọn có đúng là đoạn reasoning quan trọng, và các đoạn bị loại có đúng là câu dẫn/lặp/thừa không (đối chiếu định tính như Fig.3).

> Liên hệ repo hiện tại: pipeline tái lập đã có (`data_prep → gradient_capture → build_masks → train_sft → evaluate`), đã dùng **Qwen3-1.7B-Base**, per-sample SVD, spectral strength = leverage score đúng Eq.7. R2/R3 chính là phần còn thiếu: **script visualize** spectral strength và **log thời gian SVD/gradient** (một phần đã có: `gradient_capture.py` in `k*`, strength quantiles, s/sample; còn thiếu biểu đồ trực quan).

### 2.2 Chuẩn bị dữ liệu cho hướng mới (giao Phong + cả nhóm)
- **Sinh nhiều lời giải (G responses)** từ **teacher Qwen lớn** trên vài bộ toán (thay vì teacher chỉ gen 1 chuỗi như paper).
- Chọn **bộ dữ liệu toán** mà teacher **rõ ràng mạnh hơn** student (gap đủ lớn → distillation mới có ý nghĩa).
- Tham khảo các bộ paper dùng (toán, code); ưu tiên bộ **đa dạng độ khó**; chênh lệch phương pháp quanh **~10%**.
- **Phong lấy data xong sẽ chuyển cho nhóm.** Hai bạn chuẩn bị **code** sẵn.

### 2.3 Setup distillation cho hướng mới
- **Black-box distillation**: teacher Qwen lớn (vd Qwen-2.5 ~32B, hoặc con ~27–35B chạy được trên A40) → distill xuống **Qwen/Llama nhỏ** (0.6B / 4B / 7B).
- Lý do chọn black-box: **DeepSeek khó host** (model 120B+ không host nổi); white-box distillation cân nhắc sau nếu có model host được.
- Giai đoạn **làm data vất vả**, nhưng **train nhẹ** (chỉ quanh 0.x–7B).

---

## 3. Định hướng đề tài mới (ý tưởng thầy đề xuất)

Điểm chung: **thay vì teacher chỉ sinh 1 CoT như paper gốc, sinh nhiều chuỗi (G responses)**, rồi tận dụng công cụ "tìm đoạn quan trọng" của paper để khai thác.

### Idea 1 — Multi-response + ranking loss (kết hợp bài GRPO thầy gửi)
- Teacher sinh **G responses** (kiểu GRPO nhưng dùng để **distill**, không phải RL).
- Dùng **spectral strength** (thay cho cách tìm token quan trọng của bài GRPO kia) để tìm đoạn/response quan trọng → **nhanh hơn?** (cần đo).
- **Ranking** các response (quan trọng hơn / kém hơn) → thêm **loss kiểu DPO** để dạy student:
  - chuỗi teacher **đúng đáp án** → winning; **sai đáp án** → losing.
- Kiến trúc 2 loss: giữ SFT/GRPO-style + **DPO ranking loss** (bổ sung để bài "thú vị/insight" hơn — thầy lưu ý ranking chỉ là **bổ sung**, không phải cốt lõi).
- Cốt lõi giá trị: **tận dụng nhiều response của teacher** (khai thác sức mạnh teacher mạnh hơn) + **rút ngắn** chuỗi reasoning (giữ đoạn quan trọng, bỏ đoạn thừa).

### Idea 2 — Reward từ khả năng reasoning của student
- Sinh nhiều chuỗi → tìm **đoạn reasoning quan trọng** → **đưa gợi ý/đoạn quan trọng (hoặc bản tóm tắt) cho student**.
- Nếu student **reasoning ra đúng đáp án** → chuỗi/đoạn đó **reward cao** (dạng như reward model / positive-negative cho DPO).
- Biến thể online: cho student tự gen với gợi ý; đúng → positive, sai → negative; **baseline = chính paper này**.
- Liên hệ hướng của Phong: đoạn reasoning nào đưa vào giúp student ra kết quả → đánh reward cao.

### Ý tưởng phụ — Đo tính đa dạng (diversity) của G responses
- Đo **độ đa dạng** giữa các response (bằng gradient / độ tương đồng giữa các ma trận `G`) → chọn tập response **đa dạng** để dạy student, **tránh bias**, tránh trùng lặp (sinh nhiều nhưng đáp án hay trùng → không tốt).
- **Lưu ý phản biện của thầy**: đo đa dạng **để làm gì** còn chưa rõ mục đích — nếu chỉ đo mà không dùng để điều khiển gen thêm lời giải khác thì **chưa thuyết phục**. → cần chốt mục đích trước khi làm.
- Có thể ra **công thức trọng số** cho sample (weighting) hoặc cho token/đoạn → dùng để ranking/khai thác G responses.

### Ý tưởng bị cân nhắc rồi gác lại
- **Student tự tóm tắt lại CoT rồi học chính bản tóm tắt**: thầy đánh giá **có thể work nhưng khó viết paper** — idea "simple but effective" khó được chấp nhận; cần **insight/phân tích chiều sâu** (như paper này làm) mới dễ qua review.
- **Chuyển sang domain safety/online alignment** (có deadline abstract ~14/8, ~21/8): thầy thấy **lan man, không phù hợp** với hướng chính → tạm gác, "nghĩ thêm".

---

## 4. Ràng buộc & rủi ro (nêu trong họp)

- **Gap teacher–student phải đủ lớn.** Lo ngại lặp lại: nếu teacher ≈ student thì distillation gần như vô nghĩa; phải chọn dataset toán khó nơi teacher rõ ràng mạnh hơn.
- **Chi phí SVD/gradient chưa rõ** (paper giấu) → phải tự đo trước khi cam kết scale.
- **Không host được DeepSeek / model 120B** → chốt hướng **black-box distillation** với teacher Qwen lớn host được (A40).
- **Áp lực viết paper**: idea phải có insight/novelty, không chỉ "đơn giản mà hiệu quả".
- **Thời gian**: thầy muốn **làm nhanh** để kịp (có nhắc mốc 2 tuần và deadline online/safety 14–21/8, nhưng hướng safety đã gác).
- **Dữ liệu/code paper không public** → phải tự dựng lại toàn bộ.

---

## 5. Action items (theo người)

| Người | Việc |
|---|---|
| **Sang & Luyện** ("hai bạn chuẩn bị code") | Tái lập & kiểm chứng paper: cài đặt spectral strength (R1), **visualize** (R2), **đo thời gian** SVD/gradient (R3), kiểm định step chọn có đúng đoạn quan trọng (R4). Chuẩn bị **code** cho hướng multi-response. Đọc paper GRPO (thầy gửi) + paper Long-Context để tìm cách ghép. |
| **Sang** (riêng) | Đã trình bày xong paper chính; đang nắm phần "đại số/toán" của paper (mò cả tuần); tiếp tục làm rõ các câu hỏi mở §6. |
| **Luyện** (riêng) | Paper safety-alignment (hidden-state editing) trình bày buổi này **tạm gác**; chuyển sang tập trung hướng Spectral chung. |
| **Phong** | Lấy/sinh **data** (teacher Qwen lớn sinh G responses trên bộ toán), chọn bộ có gap teacher–student rõ, **chuyển data cho nhóm**. Liên hệ hướng reward từ reasoning của student ("giống Phong đang làm"). |

---

## 6. Câu hỏi mở / cần chốt trước khi triển khai

1. **`h` là hidden state tầng nào?** Paper không nói rõ. Repo hiện dùng **last hidden state (post-norm, trước lm_head)** — cần xác nhận đây là lựa chọn đúng hay thử vài tầng.
2. **Chi phí SVD thực tế** trên quy mô của nhóm (Qwen lớn, chuỗi dài) — chưa có số; R3 phải trả lời.
3. **Mục đích của "đo đa dạng"** — dùng để điều khiển sinh thêm lời giải, để lọc, hay để weighting? Chưa chốt → thầy đã phản biện.
4. **Chọn teacher/student cụ thể** (Qwen 32B/2.5 → 0.6B/4B/7B?) và **bộ dataset toán** nào (gap đủ lớn, ~10% chênh) — chờ Phong khảo sát + lấy data.
5. **Ranking/DPO loss** thiết kế cụ thể ra sao (winning/losing theo đúng đáp án?) — mới ở mức ý tưởng.
6. **Định nghĩa "đoạn quan trọng ở mức response"** (paper làm ở mức step trong 1 chuỗi; hướng mới cần so **giữa nhiều response**) — cần mở rộng công thức spectral strength từ step→response.
