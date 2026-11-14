# Báo cáo Day 7 — Nguyễn Công Quân (2A202603675)

## Dữ liệu và benchmark

- Nguồn: *Python Tutorial, Release 3.13.7*, Python Software Foundation License Version 2. File PDF có 167 trang vật lý; theo quy tắc lọc của Day 6 có 101 trang nội dung hợp lệ. Snapshot có SHA-256 `7e934cd7d3acbe31d5590391148231b2d185a92d0b843d4daee82913ec1907ef`.
- Corpus dùng strategy B đã chọn ở Day 6: trích xuất text block theo trang, nhóm theo paragraph, tối đa 1.000 ký tự và không tạo chunk qua trang. Tổng cộng 308 chunk có ID ổn định, nguồn, trang và metadata.
- Dense index Day 6: Chroma 1.5.9, cosine similarity; embedding `sentence-transformers/all-MiniLM-L6-v2`, 384 chiều, vector được normalize. Index cục bộ không commit; có thể tạo lại từ `data/chunks.json` bằng cùng model.
- Benchmark gồm 24 câu. Dev có 15 câu có đáp án và 2 câu thiếu evidence; test có 5 câu có đáp án và 2 câu thiếu evidence. Gold chỉ phục vụ đánh giá, không được đưa vào agent hoặc retrieval query.
- Câu có đáp án được lấy từ benchmark Day 6 và đối chiếu để evidence nằm nguyên vẹn trong đúng chunk, `source_id` và trang. Câu thiếu evidence hỏi thông tin gần chủ đề nhưng nằm ngoài phạm vi Python Tutorial; agent phải tìm rồi từ chối vì không đủ bằng chứng.
- Lệnh kiểm tra CP1: `python scripts/validate_cp1.py`. Kết quả: 308 chunk, đủ tỷ lệ 15+2 dev và 5+2 test, toàn bộ 20 evidence hợp lệ.

## Retrieval và thử nghiệm

Kết quả TF-IDF chạy thử trên **dev** sau CP1: Hit@1 = 9/15, Evidence Hit@3 = 13/15; đây chưa phải kết quả test.

Checkpoint 2 nối dense retrieval thật bằng Chroma cosine và cùng embedding model Day 6. Agent dùng câu hỏi gốc cộng query do model local tạo, lấy top-3 và đóng gói tối đa 3.500 ký tự. Ngưỡng dense `0.40` được thử trên dev để tránh trả lời khi evidence yếu. Persistent index nằm ở `artifacts/chroma/` và có thể tái tạo từ `data/chunks.json`.

| Cấu hình | Phương pháp / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Median retrieval |
|---|---|---|---:|---:|---:|---:|
| Baseline | TF-IDF, top-3, 3.500 chars | dev | 9/15 | 13/15 | 13/15 | 0.622 ms |
| Dense | Chroma cosine, top-3, 3.500 chars | dev | 8/15 | 13/15 | 13/15 | 7.214 ms |
| Hybrid | Dense + TF-IDF, RRF, top-3 | dev | 9/15 | 13/15 | 13/15 | 8.728 ms |
| Cải thiện 1 | Dense top-5 + filter, 5.500 chars | dev | 8/15 | 13/15 | 14/15 | 7.560 ms |
| **Cải thiện 2 / đã chọn** | **Hybrid RRF top-5 + filter, 5.500 chars** | **dev** | **9/15** | **13/15** | **14/15** | **8.177 ms** |
| Baseline | TF-IDF, top-3, 3.500 chars | test | 3/5 | 3/5 | 3/5 | 0.699 ms |
| **Cấu hình đã chọn** | **Hybrid RRF top-5 + filter, 5.500 chars** | **test** | **4/5** | **5/5** | **5/5** | **9.947 ms** |

Tiêu chí đã công bố trước test: tối đa Evidence Hit@3, sau đó Hit@1, Context Evidence Rate, cuối cùng mới giảm latency. Theo tiêu chí này, `hybrid_rrf_top5_filter` được khóa cho checkpoint 4. Hai cải thiện tăng context evidence 13/15 lên 14/15 nhưng không tăng Hit@3; chi phí dịch vụ vẫn 0 USD, đổi lại latency cao hơn TF-IDF.

Phân tích lỗi: q06 và q09 được hybrid phục hồi vào top-3; q10 bị RRF đẩy gold ra ngoài top-5; q14 chuyển gold từ rank 2 xuống rank 4 nên mất Hit@3 nhưng context vẫn đủ. Chi tiết ở `results/cp3/error_analysis.md`.

## Agent và đánh giá

Model thật `Qwen/Qwen2.5-0.5B-Instruct` chạy local trên CPU, không dùng API và chi phí dịch vụ bằng 0. Model tạo query tìm kiếm và sinh câu trả lời từ context; hệ thống chỉ chấp nhận citation thuộc context. Nếu model bỏ alias citation, policy chọn chunk trong context có độ phủ từ vựng cao nhất với câu trả lời. Nếu top dense score dưới ngưỡng hoặc model nói thiếu bằng chứng thì agent từ chối.

Trace CP2 gồm q01, q07 và q21. q07 đúng đủ và citation hợp lệ; q21 từ chối đúng. q01 đúng ý chính nhưng thêm mô tả chưa chính xác về `-c/-m`, nên chấm 1/2 và citation không hỗ trợ toàn bộ câu. Đây là lỗi dev cần xử lý ở checkpoint 3 bằng prompt ngắn gọn/grounding tốt hơn.

Trên toàn bộ dev với cấu hình đã chọn, agent gọi retrieval cho 17/17 câu, trả lời 15 câu có đáp án và từ chối đúng 2/2 câu thiếu evidence. Focused context giảm nhiễu so với đưa cả top-5 vào model, nhưng Qwen 0.5B vẫn sai ở các câu phủ định/định nghĩa gần nhau (q06, q08, q13, q14). Nhãn thủ công và summary nằm trong `results/cp3/agent_dev_reviews.json` và `agent_dev_summary.json`; kết quả thấp được giữ nguyên thay vì sửa bằng gold.

### Đánh giá test sau khi khóa cấu hình

| Cấu hình | Tool use | Grounded answer | Citation hợp lệ | Câu trả lời có citation | Abstention đúng | Từ chối nhầm |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF baseline | 7/7 | 1.8/2 | 5/7 | 7/7 | 0/2 | 0/5 |
| Hybrid đã chọn | 7/7 | 1.8/2 | 5/6 | 6/6 | 1/2 | 0/5 |

Đọc thủ công context của 5 câu test có đáp án: baseline có mức 2 ở 3 câu và mức 0 ở 2 câu (trung bình 1.2/2); hybrid có mức 2 ở cả 5 câu (2.0/2). Hybrid phục hồi evidence q17 và q19, nhưng model vẫn chưa khai thác đầy đủ q17 nên câu trả lời chỉ đạt 1/2.

Ví dụ thành công q16: agent phân biệt đúng instance/class variables và citation page 84 hỗ trợ đầy đủ. Ví dụ thiếu evidence q24: cấu hình chọn tìm kiếm rồi từ chối đúng. Lỗi chính q23: đoạn về binary floating point có dense score đủ cao khiến agent bịa `float64` cho NumPy và gắn citation không hỗ trợ. Vì đây là test, không thay đổi threshold sau khi quan sát lỗi.

Trace và nhãn: `results/cp4/*_traces.json`, `*_reviews.json`, `context_reviews.json`. Model chạy CPU local, một lượt retrieval mỗi câu, không có lỗi provider và chi phí dịch vụ 0 USD. Đánh đổi chính là hybrid chậm hơn TF-IDF khoảng 14 lần ở retrieval nhưng tăng Evidence Hit@3 từ 3/5 lên 5/5.

### Tái lập và giới hạn

Chạy `python scripts/validate_cp1.py`, sau đó các script CP3 và CP4 theo README. Lần đầu cần tải hai model trong `requirements.txt`; các lần sau có thể bật Hugging Face offline mode. Persistent Chroma index được tái tạo tự động và không commit. Hạn chế lớn nhất là model 0.5B có năng lực suy luận/tuân thủ grounding thấp, lexical tokenizer không tối ưu đa ngôn ngữ, benchmark nhỏ và chỉ dùng một tài liệu tiếng Anh.

- Trace: `results/cp2_agent_traces.json`
- Nhãn thủ công: `results/cp2_manual_review.json`

## Thử thách nâng cao

Chưa thực hiện trong checkpoint 1.
