# Chạy bài trên Colab T4

## Tiếp tục từ trạng thái ngày 14/10/2026

### Thay đổi sát hạn nộp (ưu tiên hơn hướng dẫn full E3 bên dưới)

Người dùng quyết định dùng E3 `e3_scale1024/step-0200` sau khi backup Drive làm dừng run ở bước 200. Đã đánh giá val và test E1–E3; E3 val F1=0.3223736969. E3 là run **200 bước, không đạt đủ 300 bước của đề**. Dùng tên E3_step0200 và trình bày rõ hạn chế; không gọi là full 300.

E4 data_size dùng nền E3 đã chọn bằng val: 10% so với đối chứng 100% tại cùng 200 bước thực tế, scale1024, batch2/accum4, LR/seed gốc. CLI `--stop-after 200` ghi nhận dừng sớm nhưng vẫn giữ `max_steps=300` để lịch warmup/cosine khớp 200 bước đầu của đối chứng E3 (không đổi thành lịch LR 200 bước). Run E4 và đối chứng đều có sai khác thời lượng so với yêu cầu đề; không bảo đảm đủ điểm. Đối chứng 100% tái chấm prediction E3 test đã có, không suy luận lại. Không chạy mục 7/8 cũ với đường dẫn step-0300 cho E3.

- `lab_day10_v1.ipynb` là bản người dùng tải về, giữ nguyên lịch sử/ảnh/output/lỗi. Dùng **`lab_day10.ipynb` đã chỉnh** cho lượt Colab tiếp theo; các cell mới chưa chạy và không gắn output giả. Phụ lục trích output thật có ghi nguồn cell v1; REPORT chưa sửa và evidence gốc chưa được tải về local.
- Đã xong E0 test; E1/E2 train 300 bước + val. E3 smoke `e3_scale1024_smoke` đạt 8 bước và round-trip adapter/projector; **full E3 chưa chạy**. Test E1–E3, E4 và báo cáo vẫn còn.
- Tạo lại ZIP bằng `python -m src.package_colab` (đã tạo khi bàn giao), upload notebook mới và ZIP mới. Chạy mục 0: setup, kiểm tra GPU/mount Drive, **cell khôi phục sau nghỉ GPU**. Cell này chép các run cần thiết từ Drive, giữ nguyên source mới và kết quả đang có. ZIP không chứa trọng số; trọng số chỉ chép giữa Drive và Colab.
- Chạy mục 1 để tạo lại dữ liệu và xác nhận `d10-21e4070301`. Không chạy lại E0; mục 3 cũng tự giữ evidence đã có. Mục 5 mặc định `EXP="e3"`, tag `e3_scale1024`, batch 2/accum 4, AMP init scale 1024. Mục 6 cell smoke chỉ xác nhận fingerprint và đọc marker cũ. Cell tiếp theo là **full train E3 duy nhất**, cần đủ GPU; E1/E2 đã mất khoảng 45 phút, E3 có thể lâu hơn. Chưa có resume optimizer chính xác, nên không bắt đầu khi chỉ còn 40 phút.
- Hàm `run()` mới hiện tiến độ/traceback trực tiếp và lưu riêng từng log trong `work/logs/`; cell backup sao lưu cả log. `torchao` có sẵn nhưng cũ hơn 0.16 được gỡ ở setup vì đã gây lỗi PEFT; không đổi các thư viện đã ghim. Skip quantization dùng đúng prefix `model.visual`.
- Sau full E3 chạy val ở mục 6, sao lưu rồi gửi log. E4 tự kế thừa **cả AMP scale** và batch/accum từ checkpoint đối chứng; kiểm tra config trước khi chạy để chỉ thay data_size.
- Những lần E3 scale mặc định lỗi gradient và thử batch 1/accum 8 không được tính là thí nghiệm hoàn chỉnh. Scale 1024 mới xác nhận smoke, chưa chứng minh full ổn. Cần ghi nhận khác biệt scale giữa E3 và E1/E2 trong báo cáo khi có kết quả, không bỏ qua giới hạn so sánh.

Các phần dưới là hướng dẫn chung cho quy trình từ đầu; nếu tiếp tục bài hiện tại, ưu tiên thứ tự ở mục trên để không chạy lại phần đã hoàn thành.

## Chuyển code hiện tại lên Colab (không cần commit/push)

1. Trong PowerShell tại repo: `python -m src.package_colab`. Lệnh chỉ tạo `work/colab_bundle.zip` từ source/tài liệu/config/test, không đóng gói data, secrets, trọng số hay `.git`.
2. Mở https://colab.research.google.com → Upload notebook → chọn `lab_day10.ipynb` ở repo local.
3. Runtime → Change runtime type → **T4 GPU**. Chạy mục 0, upload `work/colab_bundle.zip` khi được hỏi. Không clone origin rồi chạy vì thay đổi local chưa được push.
4. Cài requirements đúng phiên bản trong notebook. Nếu Colab yêu cầu restart session, restart rồi chạy lại cell thiết lập workspace (`%cd /content/vlm_lab`); không upload/giải nén lại đè dữ liệu cũ.
5. Tạo Colab Secrets `KAGGLE_USERNAME`, `KAGGLE_KEY`, bật notebook access. Không ghi/hiển thị giá trị khóa. Hoặc dùng zip dataset trong Drive với prepare_data `--zip` thay cho cell Kaggle.

## Thứ tự chạy

- Mục 0–2: cài đặt, mount Drive, chuẩn bị split, thống kê và xem nhãn. Dataset lưu `/content/vlm_lab/data`; đường dẫn ảnh trong JSONL tương đối với thư mục này.
- Mục 3: smoke E0 trên 2 ảnh val. Xem output thô trước; nếu lỗi dừng, gửi log. Chỉ chạy cell E0 test khi pipeline đã ổn.
- Mục 4–6: smoke E1 (2 bước, tối đa 8 ảnh) kiểm tra mask/token ảnh, module LoRA, gradient, tham số thay đổi, checkpoint round-trip và suy luận sau nạp lại. Full train bị chặn nếu chưa có `smoke_pass.json` cùng config/dataset.
- Chạy E1 đầy đủ rồi chấm val. Đổi `EXP` sang `e2`, chạy lại smoke/full/val; sau đó `e3`. Mỗi lệnh là một process riêng và nạp model gốc độc lập; không train nối E1 → E2 → E3.
- Mục 7: khi đã kiểm tra val, chạy đủ test E1–E3 và chấm evidence bằng script gốc. Không dùng test để thử nhiều cấu hình.
- Mục 8: chọn đối chứng có F1 val tốt nhất (hòa ưu tiên setup đơn giản hơn), E4 chỉ đổi `data_size` 10%/100%. Smoke và train 10% từ model gốc, giữ 300 bước và các siêu tham số khác. Đối chứng 100% tái sử dụng đúng run/prediction của setup gốc, tạo evidence E4 bằng score gốc; không train lại không cần thiết.
- Cuối notebook: xem bảng kết quả thật, vẽ GT/dự đoán, lưu report/prediction/evidence vào Drive. REPORT vẫn để nguyên cho đến khi có số thật. Trước nộp mới chạy toàn bộ tests và check_submission; chưa đủ evidence/báo cáo/commit thì checker báo FAIL là dự kiến.

## Log cần gửi lại

Gửi đầu tiên: GPU/phiên bản thư viện, thống kê dataset và toàn bộ output smoke E0/E1 (kể cả traceback). Sau mỗi setup gửi `setup.json`, `run.json`, `train_log.jsonl` và score val/test tương ứng. Không gửi khóa hoặc trọng số. Notebook hiển thị tên checkpoint và đầu ra cần lấy.

## Lưu trữ và chạy lại

- Cập nhật sau lỗi Drive ở E3 bước 200: `src/checkpoints.py` chỉ backup checkpoint mới nhất; lỗi copy Drive được ghi `backup_status.json` và cảnh báo `BACKUP FAILED`, train vẫn tiếp tục vì checkpoint local đã lưu. Phải kiểm tra `BACKUP OK` hoặc cứu file riêng trước khi reset runtime; đây không phải resume optimizer. Run E3 200 bước đã được cứu thành ZIP trên Drive, không tính là kết quả 300 bước.

- `/content` sẽ mất khi Colab ngắt. Checkpoint và log được sao lưu vào `MyDrive/day10_vlm/runs` mỗi 100 bước và khi kết thúc; evidence/prediction dùng cell sao lưu riêng cuối notebook, nên chạy cell này sau mỗi setup.
- Checkpoint hiện hỗ trợ nạp lại để suy luận; chưa lưu optimizer/RNG để resume train chính xác. Nếu train ngắt giữa chừng, giữ log cũ và chạy mới từ model gốc vào thư mục mới; không gọi phần train còn lại là run 300 bước hoàn chỉnh.
- Lệnh không ghi đè thư mục train hoặc prediction/evidence đã có. Khi cần chạy lại, đổi `TAG` (ví dụ `e2_retry1`) và cập nhật đường dẫn checkpoint/score tương ứng; giữ kết quả cũ để đối chiếu. Với mất session, chép run cần dùng từ Drive về `work/runs/` trước.
- Nếu OOM: dừng, gửi traceback/VRAM. GUIDE cho phép batch=1, grad_accum=8; thêm **cả hai** `--batch-size 1 --grad-accum 8` vào smoke và full, chạy smoke mới. Ghi lại thay đổi; E4 giữ đúng cấu hình đối chứng đã chạy. Không tự giảm bước, đổi phiên bản hay dùng GPU local.
- Smoke vài bước chỉ xác nhận đường chạy, không chứng minh F1 tốt hoặc loss sẽ giảm đủ. Chưa xác nhận model thật/T4 cho đến khi bạn cung cấp log Colab.

## Ghi chú test gốc khi điền báo cáo

`tests/test_configs_and_docs.py::test_report_template_has_student_markers_and_ten_questions` hiện yêu cầu REPORT vẫn có `<điền`, trong khi checker yêu cầu bỏ toàn bộ marker sau khi hoàn thành. Khi điền báo cáo thật, chạy test và ghi nhận xung đột này; không thêm marker giả để lách checker và không sửa test gốc. Nếu phát sinh đúng lỗi này, cần hỏi giảng viên về tiêu chí áp dụng.

Tham chiếu API dùng khi triển khai: [Qwen3-VL trong Transformers](https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl), [PEFT quantization](https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/quantization.md). Cấu hình thực nghiệm vẫn lấy từ GUIDE/YAML trong repo; phiên bản runtime vẫn giữ nguyên requirements đã ghim.
