# Báo cáo Day 7 — Nguyễn Công Quân (2A202603675)

## Dữ liệu và benchmark

- Nguồn: *Python Tutorial, Release 3.13.7*, Python Software Foundation License Version 2. File PDF có 167 trang vật lý; theo quy tắc lọc của Day 6 có 101 trang nội dung hợp lệ. Snapshot có SHA-256 `7e934cd7d3acbe31d5590391148231b2d185a92d0b843d4daee82913ec1907ef`.
- Corpus dùng strategy B đã chọn ở Day 6: trích xuất text block theo trang, nhóm theo paragraph, tối đa 1.000 ký tự và không tạo chunk qua trang. Tổng cộng 308 chunk có ID ổn định, nguồn, trang và metadata.
- Dense index Day 6: Chroma 1.5.9, cosine similarity; embedding `sentence-transformers/all-MiniLM-L6-v2`, 384 chiều, vector được normalize. Index cục bộ không commit; có thể tạo lại từ `data/chunks.json` bằng cùng model.
- Benchmark gồm 24 câu. Dev có 15 câu có đáp án và 2 câu thiếu evidence; test có 5 câu có đáp án và 2 câu thiếu evidence. Gold chỉ phục vụ đánh giá, không được đưa vào agent hoặc retrieval query.
- Câu có đáp án được lấy từ benchmark Day 6 và đối chiếu để evidence nằm nguyên vẹn trong đúng chunk, `source_id` và trang. Câu thiếu evidence hỏi thông tin gần chủ đề nhưng nằm ngoài phạm vi Python Tutorial; agent phải tìm rồi từ chối vì không đủ bằng chứng.
- Lệnh kiểm tra CP1: `python scripts/validate_cp1.py`. Kết quả: 308 chunk, đủ tỷ lệ 15+2 dev và 5+2 test, toàn bộ 20 evidence hợp lệ.

## Retrieval và thử nghiệm

Sẽ hoàn thiện ở checkpoint 2–3. Kết quả TF-IDF chạy thử trên **dev** sau CP1: Hit@1 = 9/15, Evidence Hit@3 = 13/15; đây chưa phải cấu hình agent hoặc kết quả test.

## Agent và đánh giá

Sẽ hoàn thiện ở checkpoint 2–4 sau khi xây agent, lưu trace và chốt cấu hình trên dev.

## Thử thách nâng cao

Chưa thực hiện trong checkpoint 1.
