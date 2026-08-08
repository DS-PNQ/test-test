# Sửa lỗi mã ngôn ngữ ASR cho Tiếng Việt

Hiện tại, module ASR (Whisper) đang sử dụng mã token `50264` cho Tiếng Việt, vốn là mã dành cho Tiếng Hàn (`<|ko|>`). Điều này khiến hệ thống nhận diện sai ngôn ngữ. Kế hoạch này sẽ cập nhật mã token chính xác cho Tiếng Việt là `50278` (`<|vi|>`).

## Proposed Changes

### Android Pipeline

#### [MODIFY] [ASRModule.java](file:///D:/StudioProjects/test-test-demo2/android/app/src/main/java/com/omnivoice/onspeak47/pipeline/ASRModule.java)

Thay đổi giá trị token của "vi" trong `LANGUAGE_TOKENS` từ `50264` thành `50278`.

## Verification Plan

### Automated Tests
- Chạy các test case trong `tests_local/test_01_asr.py` (nếu môi trường hỗ trợ) để đảm bảo mô hình Whisper vẫn hoạt động bình thường với mã token "vi".
- Kiểm tra lại các file code khác xem có sử dụng hằng số này không.

### Manual Verification
- Cài đặt lại ứng dụng trên thiết bị Android.
- Chọn ngôn ngữ nguồn là Tiếng Việt.
*   **Kết quả mong đợi:** ASR nhận diện đúng Tiếng Việt, không còn ra các ký tự hoặc từ ngữ Tiếng Hàn.
