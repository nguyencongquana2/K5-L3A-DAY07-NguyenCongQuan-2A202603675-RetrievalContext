# Định dạng dữ liệu

Các file `example_*.json` chỉ minh họa định dạng, không đủ để nộp bài. Dữ liệu thật là JSON array UTF-8; nếu dùng định dạng khác, ghi cách chuyển trong README.

- **Chunk:** `id`, `text`, `source_id`, `source`, `page`, `metadata`. ID phải duy nhất; giữ phiên bản hoặc ngày nếu có.
- **Câu có đáp án:** `id`, `split` (`dev` hoặc `test`), `question`, `expected_answer`, `gold`: danh sách `{source_id, page, evidence}`. Với schema mẫu, mỗi evidence phải nằm trọn trong một chunk đúng nguồn và trang; các phần tử `gold` là evidence thay thế. Với câu multi-hop cần nhiều đoạn, bạn có thể mở rộng schema, ghi rõ nhóm evidence bắt buộc và cách chấm. Có thể thêm `filter` do người dùng cung cấp trước.
- **Câu thiếu evidence:** `id`, `split`, `question`, `answerable: false`, `expected_behavior: "search_and_abstain"`, `absence_reason`. Xem `agent_cases_example.json`.

Không đưa `gold` hoặc nhãn đáp án vào model và retrieval tool. Ghi nguồn dữ liệu và quyền sử dụng. Cách chuyển một số định dạng Day 6: [starter/README.md](../starter/README.md).

## Dữ liệu CP1 của bài làm này

- `source/tutorial.pdf`: Python Tutorial 3.13.7, bản PDF có text layer.
- `corpus_manifest.json`: nguồn, giấy phép, checksum, cấu hình chunk/embedding/index và số lượng benchmark.
- `chunks.json`: 308 chunk được export từ collection `day6_b` của Chroma Day 6.
- `questions.json`: 20 câu có evidence và 4 câu thiếu evidence, chia dev/test.

Chạy `python scripts/validate_cp1.py` từ thư mục gốc để kiểm tra schema, tỷ lệ
split, checksum và việc mỗi gold evidence nằm nguyên vẹn trong đúng chunk,
`source_id` và trang. Persistent Chroma index không được commit; `chunks.json`
là đầu vào để tái tạo index bằng đúng model ghi trong manifest.
