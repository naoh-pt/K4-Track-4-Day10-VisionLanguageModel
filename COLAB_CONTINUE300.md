# Tiếp tục sau gia hạn: E3/E4 đủ300 bước

## File cần mở

- Notebook mới: `lab_day10_continue300.ipynb`.
- Source ZIP mới: `work/colab_resume_bundle.zip`, tạo bằng `python -m src.package_colab --out work/colab_resume_bundle.zip`.
- Giữ nguyên `lab_day10.ipynb`, REPORT.md và report/evidence của bản đã nộp; snapshot Git là commit `9e211e9`. Không đổi nhãn200 thành300.

## Thứ tự và điểm dừng

1. Upload notebook mới lên Colab, chọn T4. Mục0 upload ZIP mới vào `/content/vlm_lab300`, cài đúng requirements, gỡ torchao cũ nếu gặp lại, mount đúng Drive có `day10_vlm`.
2. Mục1 khôi phục E1/E2 đã đủ300 và prediction cũ; sao chép baseline sang `report/extended`. Chuẩn bị lại dataset chuẩn; dataset_version phải là `d10-21e4070301`.
3. Mục2 smoke E3 **8 bước**, cố ý pause tại4, rồi process mới resume4→8. Gửi log và `smoke_pass.json` trước full train. Phải có `resume_smoke_pass=true`, cả adapter/projector round-trip=true. Marker smoke cũ200 bị chặn vì chưa xác nhận resume. Xem `architecture.json` để có số tham số thực đo, không nạp thêm model chỉ để đếm.
4. Mục3 full E3 mới từ model gốc đủ300 (tag `e3_r300_v1`). Không resume checkpoint200 cũ: thiếu optimizer/RNG. Nếu ngắt, restore ZIP của đúng run rồi cell full tiếp tục từ checkpoint đã hoàn tất. `train_minutes` cộng thời gian các session train, không cộng thời gian tắt máy/khôi phục/nạp lại model; peak VRAM là max giữa các session.
5. Mục4 đánh giá val E3 mới, chọn nền bằng F1 val E1/E2/E3; E1/E2 không train lại. Không chọn nền bằng test. Nếu chọn E3 mới, E4 dùng scale1024; nếu chọn E1/E2, E4 kế thừa config của run đó.
6. Mục5 smoke pause/resume và train E4 10% đủ300; 100% là đúng đối chứng đã có. Giữ horizon LR300, không dùng stop-after200. Nếu GPU không đủ, dừng an toàn sau checkpoint rồi tiếp tục buổi khác, không rút số bước.
7. Mục6 suy luận test chỉ cho E3/E4 mới; E0/E1/E2 tái sử dụng kết quả/prediction cũ. Evidence mới ghi trong `report/extended/evidence`, validation mới trong `report/extended/validation`.
8. Mục7 backup artifact nhỏ sau mỗi chặng, Save a copy in Drive cho notebook có output. Cuối lượt tải `work/extended300_results.zip`, giải nén vào repo và gửi log để đối chiếu. Chưa tự thay root evidence/REPORT: sau xác nhận mới cập nhật báo cáo, hình và kiểm tra trước nộp lại.

Với5 giờ GPU, ưu tiên E3 train300 + val trước; sau đó E4. Lượt cũ200 bước mất khoảng39 phút, nhưng checkpoint mới chứa optimizer lớn hơn nên I/O có thể tăng đáng kể. Không bảo đảm hoàn thành mọi phần trong đúng5 giờ; có thể chia session nhờ resume mới. Chưa chạy T4 với code mới cho đến khi nhận log smoke.

## Resume lưu những gì?

Mỗi checkpoint có adapter, projector nếu train, optimizer, scheduler, GradScaler, RNG Python/Torch/NumPy/CUDA, hoán vị batch/cursor/epoch và generator, thứ tự tham số trainable, phiên bản thư viện, config/dataset fingerprint, model revision, metadata thời gian/VRAM. State lưu **sau một bước tối ưu hoàn tất**, không ở giữa4 microbatch. Khi resume, khôi phục RNG sau khi tạo/nạp model; log session mới giữ riêng, không nối các dòng bị lặp sau checkpoint vào log cũ.

BatchStream được kiểm tra giữ đúng thứ tự batch của DataLoader shuffle=True,num_workers=0 cũ qua nhiều epoch. Resume kiểm tra version/config/data/order, không chấp nhận đổi setup hoặc dùng smoke làm full. State PyTorch được đọc với `weights_only=True`. Chỉ dùng checkpoint tự tạo; checksum kiểm tra hỏng file chứ không là chứng nhận nguồn tin cậy.

Kiểm tra CPU dùng mạng giả nhỏ với dropout, AdamW, scheduler và GradScaler CPU: nhánh chạy liên tục và pause/resume cho cùng batch, LR, scaler, optimizer và tensor cuối. Đây không phải phép thử Qwen trên local; GPU có thể không bảo đảm bitwise giữa các kernel/driver, nên vẫn phải smoke trên T4 cùng phiên bản đã ghim.

## Backup mới và dung lượng

- Với checkpoint có state mới, backup là **một ZIP rolling mỗi run**: `MyDrive/day10_vlm/runs/<run_name>__latest.zip`.
- Upload vào file tạm mới, so SHA256 với ZIP local, rồi thay ZIP latest của đúng run. Chỉ bản backup trước của **run mới cùng tên** được thay; archive/checkpoint run cũ đã nộp không bị sửa. Local vẫn giữ các checkpoint100/200/300 và ZIP trong `work/runs/.backup_archives/` để cứu nếu Drive lỗi.
- E3/E4 checkpoint có thể khoảng1.5–2GB vì thêm2 moment AdamW. Cần chỗ cho latest của smoke và full, cộng một ZIP tạm khi upload; kiểm tra UI Drive. Nếu backup mới lỗi, bản latest trước vẫn là điểm phục hồi gần nhất; log cảnh báo và `backup_status.json` cho biết trạng thái. Chưa reset runtime khi chưa có backup an toàn.
- ZIP mới có manifest checksum và complete marker. `restore_run_archive()` kiểm tra đường dẫn ZIP, hash từng file, không ghi đè checkpoint local khác nội dung. ZIP trọng số không vào Git.
- Có thể cứu thủ công (CPU, không nạp model):

```python
from src.checkpoints import make_run_archive
archive = make_run_archive("work/runs/e3_r300_v1")
print(archive)  # Copy lên Drive mới hoặc Download bằng Files, không commit.
```

## Nếu muốn chủ động nghỉ

Trong lệnh full có thể thêm `--pause-at 200`; đây là dừng có state, **target vẫn300**. Sau đó bỏ pause-at và thêm `--resume-from work/runs/<run>/step-0200`, giữ nguyên config, fraction, scale và smoke_dir. Không thay `--stop-after`, vì tham số đó đổi target và fingerprint. Nếu reset runtime, cell notebook khôi phục ZIP latest rồi tự chọn checkpoint. Bước hoàn tất chưa có checkpoint có thể phải chạy lại; không resume giữa accumulation.

## Phần không thay đổi

Không sửa scripts/tests/configs/requirements gốc; không chạy E5; không tải/nạp model4B hoặc train/suy luận model trên local. Chưa sửa số liệu200 đã nộp hoặc tuyên bố có kết quả300. Lỗi test mẫu REPORT vẫn cần xác nhận với giảng viên; không lách bằng marker giả.

Tham chiếu: [PyTorch serialization](https://docs.pytorch.org/docs/2.11/notes/serialization.html), [PEFT adapter state utilities](https://github.com/huggingface/peft/blob/v0.21.2/src/peft/utils/save_and_load.py). Các phiên bản runtime giữ đúng requirements của đề.
