# Kết quả sửa MPCAWIN — 07/10/2026

Đã thực hiện kế hoạch A–D tại `/Users/linhkhanh/Downloads/MPCAWIN-main`.
Project có môi trường riêng, chạy được từ ngoài thư mục project và xuất đầy đủ
quyết định trước reset. Cấu hình KSWIN 60/20 phát hiện thay đổi nguồn trên
ImprTotal hiện tại. Baseline theo mã tác giả được giữ riêng và vẫn đo được miss.

## Các file đã sửa

| File | Thay đổi |
| --- | --- |
| `src/mpcawin.py` | Detector riêng từng instance; RNG riêng; reset đúng kích thước; validate dữ liệu/schema/tham số; fit thành công trước khi thay concept; PCA batch riêng; chặn zero variance và cosine không hợp lệ; sửa cửa sổ alert_rate; snapshot trước reset |
| `src/kswin_compat.py` | Giữ thứ tự test-before-append, sampling có replacement và KS gate legacy; RandomState riêng không reseed khi reset; validate/copy dữ liệu; River là adapter riêng |
| `src/mpcawin_run.py`, `run_imprtotal.py` | CLI dùng package đã cài, config rõ ràng, feature selection dùng chung, nguồn/hash/version/interpreter được ghi; đánh số batch 0/1-based; remainder tường minh; ghép sự kiện theo horizon và không vượt sang boundary tiếp theo |
| `src/dataset_manifest.py` | Hash, schema, features có thứ tự, kiểm tra phép nối nguồn, ranh giới lấy từ số dòng nguồn; provenance chưa biết được ghi rõ |
| `test_c_values.py` | Full-stream sweep c và seed; denominator, warmup, reset, miss, sự kiện chưa ghép và delay; không dùng frozen alert replay thay cho full stream |
| `configs/*.json` | Baseline c=2/20/5, sensitivity c=3 và các biến thể; thêm cấu hình chạy riêng `legacy_60_20.json` |
| `diagnostics/*.py` | Dùng API snapshot mới; audit PCA/scaling/replay và kiểm tra hồi quy toàn stream, nhiều seed |
| `pyproject.toml`, requirements, `environment.yml`, `.vscode/settings.json`, `README.md` | Package 0.2.0, Python 3.9.25 tại `.venv`, lock dependencies, console command và cấu hình interpreter mặc định |
| `../hoids/datasets-main/prepare_impr_stratified.py` | CLI; giữ undersample/shuffle từng năm; ordered intersection; seed/quota/chunk/hash và chỉ mục dòng nguồn; kiểm tra nhãn; không ghi đè output |
| `../hoids/datasets-main/create_imprtotal.py` | Kiểm tra schema/count/hash/manifest, nối đúng thứ tự, tạo sidecar cho CSV cũ mà không viết lại CSV |
| `../hoids/datasets-main/split_batches.py`, `compare_columns.py` | CLI, batch/source từ config và manifest; Mixed/partial batch; ordered feature comparison |
| `tests/*.py` | 20 kiểm thử hồi quy và dữ liệu |

`src/mpcawin_original.py` vẫn có SHA256
`19d43f1e80d8318b6ac90834c58cb80b03670922ed33a59de9f70724694e9d6d`.
Bản source/runner/output trước sửa nằm trong `baselines/before_repair/`.
Các CSV dataset gốc được giữ nguyên; chỉ thêm `ImprTotal.manifest.json`.

## Kiểm chứng

- `pip check`: không có dependency bị hỏng.
- `python -m unittest discover -s tests -v`: **20/20 pass**. Bao gồm ownership,
  legacy parity qua reset, input không hợp lệ không phá concept, cửa sổ vừa đầy,
  evidence trước reset, hai drift liên tiếp, outlier đơn lẻ, feature order,
  hash/order nguồn, Mixed/partial batch và tái lập chỉ mục lấy mẫu với cùng seed.
- Console command chạy từ `/private/tmp`: import đúng canonical source, phát
  hiện cùng batch 94 như chạy tại project.
- ImprTotal: **320.000 dòng, 93 cột, 84 features**, nguồn 2017→2018,
  boundary dòng **160.000** = batch ID **80** = batch thứ **81**.
- Hash ImprTotal:
  `aa1e335d45f7f360e033fed31f8658bd302e51e01287deb4ae7109314a53e315`.
- Hồi quy toàn bộ 160 batch c=2 và c=3: angle, threshold, alert, KS statistic,
  p-value và quyết định drift khớp CSV trước sửa (float tolerance 1e-12).
- Seed replay 50 lượt của tầng xác nhận vẫn cho 7 sự kiện đầu sau boundary,
  43 miss; replay chỉ có ý nghĩa trước lần reset đầu tiên.

## Kết quả trên dataset hiện tại

Mọi hàng dùng c=2, seed=20 trừ hàng ghi c=3. Ranh giới nguồn chỉ phục vụ đánh giá,
không được đưa vào detector. Horizon ghép là 20 batch.

| Cấu hình | Batch ID phát hiện (0-based) | Index delay | Số batch mới đã tiêu thụ | Sự kiện chưa ghép |
| --- | --- | --- | --- | --- |
| Baseline legacy 20/5 | Không | — | — | Không |
| Baseline với c=3 | Không | — | — | Không |
| Legacy 60/20 | 94 (batch thứ 95) | 14 | 15 | Không |
| River 60/20 | 95 (batch thứ 96) | 15 | 16 | Không |
| Alert_rate 20/.5 | 82, 139 | 2 | 3 | 139 |
| Chuẩn hóa mỗi batch, KSWIN 20/5 | Không | — | — | Không |
| Không làm tròn, KSWIN 20/5 | Không | — | — | Không |

Baseline có 18/69 alert được đánh giá trước boundary và 80/80 sau boundary.
P-value nhỏ nhất .079365 vẫn lớn hơn alpha .009. Miss này còn tồn tại sau khi
sửa lỗi kỹ thuật; nó không phải lỗi đọc nhãn hoặc không thấy dataset.

Kiểm tra full stream với **11 seed: 0–9 và 20**:

| Biến thể | Miss | Sự kiện trước boundary | Index delay | Sự kiện chưa ghép |
| --- | --- | --- | --- | --- |
| Legacy 60/20 | 0/11 | 0 | 5–15 | 0 |
| River 60/20 | 0/11 | 0 | 7–15 | 0 |
| Alert_rate 20/.5 | 0/11 | 0 | 2 | Batch 139 trong cả 11 lượt |

Sweep c=2..6 với seed 0,1,2,3,20: c=2..5 miss 4/5 lượt mỗi c; c=6 miss 5/5.
Seed 3 phát hiện ở batch 84 với c=2..5. Đây là kết quả full stream sau reset.

**Legacy 60/20 là cấu hình thực dụng đã kiểm chứng cho dataset này.** Nó mở rộng
tham số mã tác giả; không đồng nghĩa tái lập cấu hình 20/5 hoặc delay trong bài
báo. Kết quả nhiều seed ở một subset không bảo đảm các stream khác đều phát hiện.
Boundary đổi nguồn cũng không cung cấp ground truth cho mọi thay đổi nội bộ;
sự kiện 139 được ghi là chưa ghép, không tự kết luận false positive.

## Chạy lại

```bash
cd /Users/linhkhanh/Downloads/MPCAWIN-main
.venv/bin/python run_imprtotal.py \
  --data ../hoids/datasets-main/ImprTotal.csv \
  --config configs/legacy_60_20.json \
  --output-dir results/run
```

Kết quả nằm ở `results/run/legacy_60_20/batches.csv` và `summary.json`.
Đổi config thành `configs/paper.json` để chạy baseline tác giả.

Evidence đã tạo:

- `results/baseline/paper/`: baseline.
- `results/experiments/`: sáu cấu hình so sánh, seed 20.
- `results/sensitivity/summary.csv`: 25 lượt c/seed và CSV từng lượt.
- `results/validation/summary.json`: parity c=2/c=3 và 33 full-stream run của
  ba biến thể; CSV từng lượt giữ snapshot KS.
- `results/outside_project/legacy_60_20/`: kiểm tra console từ ngoài project.
- `results/data/batch_summary.csv`: batch/source/attack summary.
- `diagnostics/repaired_paper_results/`: đối chiếu scaling/PCA và first-event replay.

Không thể chứng minh subset hiện tại trùng subset tác giả vì CSV cũ thiếu seed
và chỉ mục mẫu. Luồng tạo mới đã lưu đủ thông tin để tái lập; không tái tạo các
dataset lớn trong lần sửa này. Eigenvalue-angle vẫn có giới hạn với mean shift
và isotropic scaling.
