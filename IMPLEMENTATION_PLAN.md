# Kế hoạch triển khai E0–E4

## Hiện trạng và phạm vi

- Branch `main`; origin: https://github.com/naoh-pt/K4-Track-4-Day10-VisionLanguageModel.git; working tree sạch trước lượt khảo sát.
- Không tìm thấy `AGENTS.md` trong repo và các thư mục cha đã kiểm tra. Đã đọc README, GUIDE, RUBRIC, REPORT, requirements, notebook, check_submission và giao diện prepare_data/score.
- Không có `configs/e0e4.yaml`; cấu hình đề nằm riêng tại `configs/e0.yaml` … `configs/e4.yaml`. Dùng trực tiếp các file này, không tự thay cấu hình hay phiên bản đã ghim.
- Chưa có `src/`; notebook chứa các ô YOUR CODE HERE và chưa có output; REPORT còn mẫu, evidence chỉ có `.gitkeep`. Còn thiếu toàn bộ pipeline nhãn, dữ liệu SFT, setup model/LoRA, train, checkpoint, inference và phân tích kết quả.
- Chỉ làm E0–E4, chưa làm E5. Không sửa `scripts/` và `tests/` gốc. Notebook gọi module trong `src/` và trình bày kết quả.
- Windows/MX330 chỉ viết code và kiểm tra nhẹ: tuyệt đối không tải/nạp model 4B, train hoặc suy luận model, kể cả CPU. Mọi bước model/GPU do người dùng chạy trên Colab T4; sửa lỗi dựa trên log/kết quả người dùng gửi.

## Checklist theo giai đoạn

- [x] **0 — Khảo sát, lập kế hoạch (local):** chỉ tạo file này; chưa cài thư viện, chạy test, tải model, train, commit hoặc push.
- [ ] **1 — Nhãn và giao diện dữ liệu (local):** dự kiến `src/__init__.py`, `src/config.py`, `src/data.py`, `src/coordinates.py`, `src/prompts.py`. Đọc JSONL chuẩn, ghép `image_path` tương đối với `data/`, đổi YOLO sang số nguyên 0–1000, làm tròn/kẹp và sắp theo `(y1,x1)`. Xác nhận bằng mẫu tổng hợp: box biên, ảnh không có object, chuyển ngược pixel; không cần model. Chỉ đọc/chạy test gốc liên quan lúc triển khai, kiểm tra bổ sung đặt trong `src/`.
- [ ] **2 — Chuẩn bị notebook Colab và dữ liệu:** sửa `lab_day10.ipynb` để thiết lập thư mục repo, cài đúng requirements, lấy Kaggle Secrets, mount Drive và gọi script chuẩn. Xác nhận trên Colab: số ảnh/box, session không giao nhau, `dataset_version`, 6 ảnh nhãn. Giữ nguyên split do prepare_data sinh; không tải dữ liệu trong lượt khảo sát.
- [ ] **3 — Model và E0 (Colab):** dự kiến `src/modeling.py`, `src/inference.py`. Nạp NF4, double quant, fp16; bỏ lượng tử hoá `visual` và `lm_head`; chặn đường chạy model ngoài Colab GPU. Suy luận chung dùng left padding, greedy, cắt phần prompt, ghi JSONL `{image_id,raw}`. Xác nhận trước trên ít ảnh val, xem output thô; khi pipeline ổn mới chạy đủ test E0 và chấm bằng score gốc.
- [ ] **4 — SFT và setup E1–E3:** dự kiến `src/sft.py`, `src/setups.py`; notebook gọi chung pipeline. Mask prompt/token ảnh/padding bằng `-100`, kiểm tra prompt là tiền tố, giữ token kết thúc; pad các trường chuỗi và nối pixel/grid. Local kiểm tra logic bằng dữ liệu giả nhỏ; Colab kiểm tra processor thật, 220 token ảnh 640×360 và decode phần labels. Colab xác nhận 252 module LoRA LLM ở E1/E2, 348 ở E3; chỉ đúng khối được train, projector E2/E3 fp32 và chênh khoảng 109M tham số so E1. E1, E2, E3 đều khởi tạo độc lập từ model gốc.
- [ ] **5 — Train và checkpoint dùng chung E1–E4 (Colab):** dự kiến `src/train.py`, `src/checkpoints.py`, `src/run.py`. Theo YAML và GUIDE: AdamW, warmup 5%/cosine, AMP/GradScaler, tích luỹ gradient, clip 1.0, checkpointing non-reentrant, 300 bước tối ưu; dừng khi loss không hữu hạn. Lưu adapter cùng toàn bộ projector đã học ở E2/E3 (và E4 nếu mở projector), cấu hình và run.json; sao lưu Drive mỗi 100 bước. Xác nhận bằng smoke test vài bước/ít ảnh trước mỗi train đầy đủ; kiểm tra gradient hữu hạn, tham số thay đổi đúng, lưu/nạp lại adapter + projector khớp. Local chỉ kiểm tra checkpoint bằng đối tượng giả nhỏ nếu môi trường sẵn có; checkpoint model thật chỉ trên Colab. Ghi trainable_params, peak_vram_gb, train_minutes thực đo. Nếu OOM dùng phương án GUIDE batch 1/accum 8 và ghi nhận thay đổi, không âm thầm đổi setup.
- [ ] **6 — Đánh giá E1–E3 (Colab):** dùng lại `src/inference.py`, nạp model gốc + adapter + projector tương ứng; gỡ lỗi trên val (ví dụ 40 ảnh), không chọn cấu hình bằng test. Khi ổn, chạy đủ test một lần/setup, chấm script gốc với meta thực đo. Xác nhận e0–e3 có score_version=1, split=test, missing_predictions=0, cùng dataset_version khớp meta.json; theo dõi E1 parse_valid_rate ≥ 0,95 và F1 tăng ≥ 0,10 so E0, không tạo số để đạt ngưỡng.
- [ ] **7 — E4 đối chứng (Colab):** dùng pipeline chung, chọn đúng một biến theo e4.yaml trên setup tốt nhất E1–E3; ưu tiên chọn bằng val trước khi chốt test. Dự kiến chọn `data_size` 10%/100% với subset train cố định theo seed; chốt sau khi có log và kết quả thật. Giữ mọi cấu hình khác như đối chứng, khởi tạo từ model gốc cho từng lượt, chạy cả hai giá trị (có thể tái sử dụng đối chứng 100% nếu hoàn toàn trùng). Lưu `e4_<biến>_<giá trị>.json`, ghi nguồn đối chứng và đọc CI95. Nếu chọn rank phải chạy 4/64 với alpha=2r; nếu chọn coord phải đổi đồng bộ nhãn/prompt và --coord khi chấm.
- [ ] **8 — Báo cáo và kiểm tra nộp:** dự kiến `src/visualize.py`, `src/results.py`, notebook, `REPORT.md`, `report/evidence/`, `report/figures/`. Chỉ điền khi có số thật; bảng F1/CI95/FP mỗi ảnh/F1 từng lớp/tài nguyên khớp evidence, ≥6 ảnh dự đoán đúng/sai kèm GT và chú thích, trả lời 10 câu và kiểm chứng sai thang toạ độ từ prediction có thật. Chạy toàn bộ test gốc và check_submission trước nộp trong môi trường có đủ dependency; có thể dùng Colab, không cần nạp model cho các lệnh kiểm tra. Kiểm tra Git không chứa data/work/secrets/weights; yêu cầu ≥5 commit thực hiện theo giai đoạn khi được yêu cầu, không commit/push trong lượt này.

## Cách phối hợp

- Mỗi phần chỉ kiểm tra nhỏ có giá trị; không đọc toàn bộ tests trước khi cần, không lặp kiểm tra đã đạt trừ khi có thay đổi liên quan.
- Chuẩn bị cell Colab và hướng dẫn chạy từng chặng; người dùng cung cấp lỗi, log smoke test, log train và evidence để xác nhận rồi chuyển giai đoạn.
- Bước tiếp theo sau bàn giao code: người dùng chạy mục 0–3 notebook trên Colab và gửi log; sau đó chạy smoke E1 trước full train.

## Tiến độ bàn giao code cho Colab

- [x] Các module dùng chung trong `src/`: config/data/coordinates/prompts, model/setup, SFT, train, checkpoint, inference/CLI, hình và bảng kết quả.
- [x] Notebook E0–E4 có upload source ZIP, Secrets/Drive, split chuẩn, smoke trước full, val trước test, E4 data_size và cell backup/kiểm tra cuối bài.
- [x] Hướng dẫn `COLAB_RUN_GUIDE.md` và công cụ `python -m src.package_colab`; không cần commit/push để chuyển code hiện tại lên Colab.
- [x] Local: 6 kiểm tra CPU nhỏ đạt (tọa độ, mask/padding, target LoRA, projector checkpoint giả, config/subset/chặn local, CLI nối scorer gốc và meta); 36 test gốc liên quan score/config/docs đạt. Cú pháp 22 code cell notebook và source, CLI Unicode Windows, nội dung ZIP đã kiểm tra; không nạp processor/model.
- [ ] Người dùng xác nhận smoke model thật/T4, train, đánh giá và E4 bằng log Colab. Các checkbox thí nghiệm ở trên chưa đánh dấu hoàn thành vì chưa có kết quả thật.
- [ ] Báo cáo, ảnh chọn đúng/sai có chú thích, toàn bộ test/checker và Git trước nộp. Đã phát hiện test mẫu REPORT yêu cầu còn `<điền` mâu thuẫn checker; ghi chú trong hướng dẫn, không sửa tests hoặc lách tiêu chí.

## Cập nhật từ notebook tải về ngày 14/10/2026

- [x] Đọc `lab_day10_v1.ipynb`, giữ nguyên bản gốc và output; tạo bản làm việc `lab_day10.ipynb` bỏ cell debug/lỗi lặp, lưu trích log thật trong phụ lục có ghi nguồn.
- [x] Thêm `src/colab_workflow.py`: log trực tiếp, khôi phục Drive, kiểm tra cấu hình đối chứng E4 và kế thừa AMP scale. Notebook dùng tag E3 `e3_scale1024`, full/smoke chung override 1024; không vô tình gọi cell cũ thiếu scale.
- [x] Đã nhận kết quả thật: E0 test F1 0.0509014; E1 val F1 0.2072452; E2 val F1 0.3001715. Không trộn split, không tạo evidence mới từ log thay cho file scorer gốc; REPORT vẫn giữ nguyên.
- [x] E1/E2 train 300 bước có backup; E3 smoke scale1024 đạt 8 bước, adapter/projector round-trip đúng.
- [ ] Full E3 (chờ đủ GPU), val E3, test E1–E3, E4, lấy evidence gốc về local và hoàn thiện báo cáo/nộp. Chưa commit/push.

## Trạng thái cuối trước nộp

- [x] Theo quyết định người dùng, giữ E3 checkpoint200 thay vì train lại300 sau lỗi backup Drive. Val/test E3 đã chạy; báo cáo ghi rõ sai khác.
- [x] E4 data_size 10%/100% tại cùng200 bước, scale1024, LR horizon300; đủ hai evidence thật. 100% tái sử dụng E3, không suy luận lại.
- [x] Đã nhận evidence test E0–E4, validation E1–E3, dataset_meta, phép chấm sai thang và6 ảnh thật có manifest; REPORT điền bằng số thật.
- [x] Sửa giải nén report/report thành đúng report/{analysis,evidence,figures,validation}; bản giải nén gốc lưu work/submission_import_original_20261014, không đưa vào Git.
- [x] Toàn bộ test gốc đã chạy:67 passed,1 skipped,1 failed do test mẫu REPORT yêu cầu marker chưa điền; không sửa tests. Checker UTF-8:0 FAIL,0 WARN. Chi tiết report/analysis/verification.json.
- [x] Đã push commit nộp đầu a40414a và người dùng đã nộp link repo. Không tuyên bố đạt đủ300 bước E3/E4.
- [x] Sau nộp, đối chiếu notebook v2 giữ nguyên SHA256; bổ sung log trích,44 điểm train đã in (gồm smoke), biểu đồ loss/gradient và liên kết README/REPORT. Evidence nộp đầu không thay đổi; không chạy thêm model.

## Gia hạn15/10 — chuẩn bị run300

- [x] State optimizer/scheduler/scaler/RNG và batch cursor, manifest/complete checkpoint; chặn resume từ checkpoint legacy200 hoặc sai config/version.
- [x] Backup ZIP rolling được kiểm checksum trước thay latest của đúng run mới; lỗi Drive không dừng train, giữ local để cứu.
- [x] Kiểm tra CPU nhỏ: batch order giống DataLoader cũ; dropout/optimizer/LR/scaler/tensor cuối khớp khi pause/resume; ZIP round-trip/checksum và rejection; đếm logical Params4bit shape.
- [x] `lab_day10_continue300.ipynb`, `COLAB_CONTINUE300.md`, bundle mới; kết quả mới ở report/extended, chưa đổi evidence/REPORT đã nộp.
- [x] Người dùng chạy smoke pause4→resume8 trên T4; đã ghi architecture và kiểm tra state restore GPU.
- [x] E3 đủ300, val chọn nền E4, E4 v2 đủ300 và test mới; báo cáo đã cập nhật, hình mới còn chờ xuất.

# Cập nhật 16/10 — kết quả sau gia hạn

- [x] E3 và E4 v2 đủ300 bước, test đủ311 ảnh; nhập JSON thật, chọn nền E3 bằng val.
- [x] Giữ báo cáo/evidence200 tại report/history/step0200; cập nhật evidence chính và REPORT.
- [x] Retry AMP có giới hạn, phát lại batch/RNG; kiểm tra fixture CPU, không tải model local.
- [x] Chạy lại toàn bộ test gốc:67 passed,1 skipped,1 failed do marker REPORT; checker0 FAIL0 WARN.
- [x] Colab CPU: xuất sáu hình/kiểm chứng tọa độ từ prediction E3 300; nhập ZIP và xem mẫu đúng/lỗi.
- [ ] Xác minh bản rescue E4 qua phiên/thiết bị khác; kiểm tra lại cấu trúc nộp và commit/push bản cuối.
