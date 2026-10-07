# Đối chiếu MPCAWIN với bài báo và kế hoạch sửa

**Cập nhật:** Kế hoạch dưới đây đã được triển khai. Xem
[báo cáo sửa và kiểm chứng](BAO_CAO_SUA_DU_AN.md). Các mô tả “hiện tại” và số
dòng trong phần audit bên dưới nói về bản trước sửa được lưu trong `baselines/`.

Ngày kiểm tra: 07/10/2026. Phạm vi là detector MPCAWIN, KSWIN, runner ImprTotal
và luồng tạo dữ liệu; không bao gồm xây thêm classifier, AutoEncoder hay toàn bộ HOIDS.

## 1. Kết luận

**Project triển khai đúng cấu trúc chính MPC → SAD → KSWIN của bài báo, nhưng
còn lỗi kỹ thuật và chưa tái lập được kết quả phát hiện drift của bài báo trên
dataset đang có. Không có cơ sở quy toàn bộ vấn đề cho KSWIN tự viết hoặc cho
cấu hình 20/5: cả hai đều bám theo triển khai gốc.**

Thư mục cần sửa là `/Users/linhkhanh/Downloads/MPCAWIN-main`, nơi có runner
thực tế. Thư mục `/Users/linhkhanh/Downloads/hoids/MPCAWIN-main` là một bản khác,
đã được chuẩn hóa môi trường nhưng hiện không được runner import.

Nguồn đối chiếu:

- [PDF người dùng cung cấp](/Users/linhkhanh/Downloads/hoids/1-s2.0-S1084804526001311-main.pdf):
  mục 4.1 / Fig. 6 / Eq. (6)-(8), trang 7-9; dữ liệu mục 5.1, trang 11-12;
  Table 1, trang 12; mục 5.5 / Table 3, trang 15; Table 8, trang 21.
- [Mã MPCAWIN của tác giả](https://github.com/ndslab-group/MPCAWIN/blob/e810ffc5e1d6d11cc66b74faa3fee6774cd15f52/src/mpcawin.py),
  commit `e810ffc5e1d6d11cc66b74faa3fee6774cd15f52`.
- [KSWIN scikit-multiflow 0.5.3](https://github.com/scikit-multiflow/scikit-multiflow/blob/0.5.3/src/skmultiflow/drift_detection/kswin.py).

`src/mpcawin_original.py` giống từng byte với mã tác giả ở commit trên.
`src/mpcawin.py` chỉ khác dòng import KSWIN. SHA256 mã tác giả:
`19d43f1e80d8318b6ac90834c58cb80b03670922ed33a59de9f70724694e9d6d`.

## 2. Những phần đã phù hợp

| Thành phần | Bài báo / mã tác giả | Project hiện tại | Kết luận |
| --- | --- | --- | --- |
| Đặc trưng batch | Eigenvalue PCA; so với concept hiện tại | Lưu eigenvalue tham chiếu, tính góc với batch mới | Đúng cấu trúc |
| Concept tham chiếu | Giữ đến khi xác nhận drift | Không thay sau mỗi batch bình thường | Đúng |
| SAD | Ngưỡng mean(A) + c*std(A) | Cùng công thức | Đúng |
| Cập nhật A | Không đưa góc vượt ngưỡng vào A | Chỉ thêm khi alert=0; warmup thu thập góc ban đầu | Phù hợp Fig. 6 |
| Input KSWIN | Chuỗi nhị phân 0/1 | Truyền alert, không truyền angle | Đúng; không sửa thành angle dưới tên tái lập bài báo |
| Reset drift | Reset cửa sổ, đặt batch mới làm concept | Có reset và set_concept | Đúng ý tưởng, cần sửa quản lý trạng thái |
| Batch và A | bs=2000, m=10 | Batch=2000; min=max=10 | Khớp |
| KSWIN | ss=5, sw=20 theo mục 5.5 | stat_size=5, window_size=20 | Khớp; 60/20 là cấu hình mở rộng |
| Tạo ImprTotal | Undersample + shuffle từng năm, khoảng 3% attack, rồi nối không shuffle lại | Giữ 3%, shuffle từng năm, nối 2017→2018 | Đúng phương pháp chung |
| Python | Table 8: Python 3.9 | Các lần kiểm tra dùng Python 3.9.25 | Phù hợp runtime chính |

Không bỏ shuffle trong từng năm để “khôi phục drift”: mục 5.1 giải thích shuffle
là chủ đích của thí nghiệm. Chỉ cần giữ nguyên thứ tự hai nguồn khi nối.

## 3. Sai khác và các điểm chưa đủ thông tin

### 3.1. Hệ số c trong runner

Table 1 sử dụng **c=2**, còn `run_imprtotal.py:148` đang dùng **c=3**. Đây là sai
khác cấu hình baseline, không phải lỗi cú pháp hoặc công thức. Table 3 báo cáo
c=2..6 hoạt động trong thí nghiệm tác giả; kết quả đó chưa được tái lập trên
subset này. Đưa c=2 thành cấu hình mặc định để đối chiếu; c=3 thuộc sensitivity.

### 3.2. Chuẩn hóa batch: bài báo và mã tác giả không mô tả hoàn toàn giống nhau

Trang 9 nói chuẩn hóa mỗi batch về mean=0, variance=1 trước PCA. Mã tác giả và
project lại fit scaler trên concept tham chiếu và dùng scaler đó cho batch mới
(`src/mpcawin.py:171`). Batch mới không nhất thiết có mean=0, variance=1.

Đây là **mâu thuẫn cần ghi nhận giữa văn bản và mã công bố**, chưa phải bằng
chứng người dùng sửa sai. Giữ `reference` làm baseline theo mã tác giả. Nếu thêm
`batch` để kiểm tra cách hiểu theo văn bản, phải đánh dấu rõ là một biến thể và
so sánh riêng; không âm thầm đổi mặc định.

### 3.3. Công thức KSWIN và thư viện

Eq. (8) viết D > sqrt(-ln(lambda)/ss). Mã thư viện thực thi p<=alpha cùng D>0.1.
Không tự thay p-value bằng công thức khoảng cách rồi coi hai cách là luôn tương
đương. Với ss=5, alpha=.009 và input 0/1, ngưỡng công thức khoảng 0.970618 cũng
chỉ chấp nhận D=1; vấn đề mẫu nhỏ vẫn còn.

Local KSWIN có thứ tự test trước append và sampling có replacement, giống bản
scikit-multiflow 0.5.3. Kiểm tra 149 update trên chuỗi alert hiện tại bằng class
trích từ source upstream, với base class không dùng trong logic update được
stub: mọi quyết định, cửa sổ và p-value ở các bước đã đủ dữ liệu đều giống nhau.
Đây không phải lần chạy đã cài nguyên bộ scikit-multiflow. Cần giữ
quy tắc này ở chế độ legacy để đối chiếu; append trước test hoặc sampling không
replacement là thay đổi hành vi, không phải sửa tương thích vô điều kiện.

### 3.4. Dataset chưa phải bản subset có thể chứng minh giống tác giả

Subset local dùng 160,000 dòng/năm, seed=42 và quota tối thiểu 5 mẫu/family.
PDF không cung cấp đủ seed, chỉ mục mẫu, thứ tự dòng và danh sách feature cuối
cùng để chứng minh đây là chính xác subset của thí nghiệm. 84 features local
không được kết luận sai chỉ vì thiếu danh sách tác giả. Cần lưu manifest và
hash, xác định intersection feature có thứ tự và quy tắc loại metadata/label.

Không dùng batch 85/86 làm đích phải đạt bằng mọi giá. PDF nêu drift được phát
hiện ở batch 86 và delay 5; cần thống nhất đánh số batch 0/1-based và dữ liệu
khởi tạo trước khi so sánh. Runner local đánh số 0-based, boundary=80.

## 4. Lỗi kỹ thuật cần sửa

| Mức | File / dòng | Lỗi | Cách sửa |
| --- | --- | --- | --- |
| P0 | Runner line 14; hai thư mục cùng tên | Import bản khác với bản đã chuẩn hóa, dễ chạy nhầm môi trường/source | Chọn một thư mục canonical; đóng gói và log đường dẫn module/interpreter |
| P1 | src/mpcawin.py:20 | Một detector mặc định được dùng chung giữa các instance | detector=None; tạo detector mới cho mỗi instance, giữ thông số baseline |
| P1 | src/mpcawin.py:89 | So sánh 'None' dạng chuỗi và seed RNG toàn cục | Xử lý None đúng; tách legacy RNG khỏi RNG của run khác, giữ cách sinh mẫu khi cần đối chiếu |
| P1 | src/mpcawin.py:112 | Reset đổi alert_window_size thành max_window_size*2 | Dùng self.alert_window_size; reset nhất quán mọi buffer |
| P1 | src/mpcawin.py:183-190 | Bỏ qua lần cửa sổ alert vừa đủ dữ liệu | Append/evict rồi kiểm tra khi đã đủ cửa sổ; đây là nhánh alert_rate, không phải nguyên nhân runner KSWIN bỏ sót |
| P1 | src/mpcawin.py:174-178 | Không chặn zero norm, NaN, cosine ngoài [-1,1] | Validate, clip cosine, chính sách rõ cho batch suy biến |
| P1 | src/mpcawin.py:172-174 | Số eigenvalue có thể đổi nếu batch ít hàng hơn số features | Quy định batch/components tương thích hoặc báo lỗi rõ; không tự pad tùy tiện |
| P1 | src/mpcawin.py:67-85 | Chấp nhận cửa sổ âm/số thập phân, coef không hữu hạn | Validate số nguyên dương, min<=max, coef hữu hạn không âm; dùng ValueError/TypeError |
| P1 | src/mpcawin.py:132 | Xóa concept trước khi fit batch mới thành công | Validate/fit vào biến tạm rồi mới thay concept và reset |
| P1 | run_imprtotal.py:289-319 | Đọc thông tin KS sau khi drift đã reset detector | Lưu snapshot của quyết định trước reset; log event khác state mới |
| P2 | src/kswin_compat.py:28-38 | Validation còn thiếu; data có thể dùng chung array | Kiểm tra cửa sổ/stat theo backend; copy input; không chấp nhận alpha=0 hoặc size=0 |
| P2 | src/mpcawin.py:172 | Refit PCA ngay trên object của concept tham chiếu | Dùng PCA riêng cho batch mới, giữ reference model nhất quán; vẫn refit để lấy eigenvalue mới |
| P2 | src/mpcawin.py:179,202 | Làm tròn góc/ngưỡng trước quyết định | Giữ chế độ legacy rounding cho baseline; đánh giá chế độ float đầy đủ riêng; chỉ round khi hiển thị trong chế độ cải tiến |

Các lỗi shared detector, reset, validation và tính góc phần lớn đã có trong mã
tác giả. Bám mã công bố không có nghĩa các lỗi đó không cần sửa. Không phải mọi
lỗi trong bảng đều gây ra miss ở lần chạy này.

## 5. Kiểm tra bổ sung đã thực hiện

Evidence: `diagnostics/paper_results/summary.json`, các CSV tương ứng,
`diagnostics/results/summary.json` từ kiểm tra trước.

| Thử nghiệm toàn bộ 160 batch, seed 20 | Alert trước boundary | Alert sau boundary | Drift |
| --- | ---: | ---: | --- |
| Mã tác giả, c=2, reference scaler, PCA default | 18 | 80 | Không |
| Cách hiểu chuẩn hóa mỗi batch, c=2 | 9 | 78 | Không |
| Mã tác giả, c=2, PCA full solver | 18 | 80 | Không |
| Runner hiện tại, c=3, reference scaler | 18 | 80 | Không |

Như vậy, đổi c về 2, đổi scaler hay đổi solver **chưa tự giải quyết** miss.

Replay input KSWIN của baseline c=2 với 50 seed: 0 seed có sự kiện đầu tiên trước
boundary, 7 seed có sự kiện đầu tiên sau boundary, 43 seed bỏ sót. Đây chỉ là
phân tích **sự kiện đầu tiên của tầng xác nhận**; không dùng nó làm chỉ số của
toàn bộ hệ thống sau reset. Trước sự kiện đầu tiên, upstream vẫn cung cấp đúng
chuỗi cảnh báo đó. Nó cho thấy độ nhạy phụ thuộc mạnh vào sampling với ss=5.

Các thử nghiệm c=3 ở lần kiểm tra trước: local KSWIN 60/20 phát hiện batch 94,
River 60/20 batch 95, alert_rate 20/.5 batch 82. Những cấu hình này được lưu làm
**biến thể cải tiến**, không dùng để tuyên bố đã tái lập kết quả bài báo.

## 6. Kế hoạch sửa theo từng file

### Giai đoạn A — khóa baseline và chọn đúng source

1. **src/mpcawin_original.py**: giữ nguyên làm bản tham chiếu; ghi provenance,
   commit và hash. Không dùng file này làm module runtime mặc định, vì import
   scikit-multiflow nguyên bản chưa được cài trong môi trường hiện tại.
2. **pyproject.toml, requirements.txt, requirements.lock.txt, environment.yml,
   .gitignore, README.md** trong project có runner: thêm cấu hình đóng gói và
   dependency vào đúng thư mục; tạo môi trường riêng tại đó. Baseline runtime
   Python 3.9 tương ứng Table 8; cấu hình Python mới được đánh dấu riêng.
   Cài cả mpcawin và kswin_compat; không phụ thuộc cwd hoặc đường dẫn src chèn
   thủ công. Lock cả SciPy vì đây là thư viện tính p-value. Ghi backend legacy
   hay River rõ ràng; không tự thay backend của baseline.
3. **configs/paper.json** (mới): bs=2000, min=max=10, c=2, legacy KSWIN
   sw=20/ss=5/alpha=.009/seed=20, scaling=reference, rounding=legacy.
   Chú thích alpha=.009 và seed=20 lấy từ mã tác giả; PDF mô tả mức khuyến nghị
   nhưng không liệt kê hai giá trị này đầy đủ trong Table 1.
4. **configs/experiments.json** (mới): cấu hình c=3, per-batch scaling, KSWIN
   60/20 và alert_rate; mọi kết quả phải ghi tên cấu hình.

Tiêu chí A: pip check/import/smoke thành công từ ngoài project; log chỉ đúng
canonical source; baseline trên subset hiện tại vẫn tái lập được miss đã ghi.

### Giai đoạn B — sửa thư viện, bảo toàn hành vi baseline

5. **src/mpcawin.py**: sửa ownership detector, validation, reset, fit concept
   an toàn, feature schema, numerical guards và nhánh alert_rate. Tách model
   tham chiếu khỏi PCA tạm của batch mới. Thêm snapshot last_result gồm angle,
   threshold_used, alert, KS test metadata, drift và concept trước/sau.
   API bool hiện tại vẫn dùng được. Không truyền label/boundary vào detector.
6. **src/kswin_compat.py**: giữ implementation legacy với thứ tự update và KS
   gate như upstream; thêm RNG riêng dùng RandomState để có thể kiểm tra parity
   với chuỗi legacy np.random.seed. Tách reset state khỏi snapshot event;
   validate/copy dữ liệu. Nếu thêm River, dùng backend hoặc adapter riêng với
   tên rõ, không thay âm thầm.

Tiêu chí B: chuỗi finite/nondegenerate của baseline giữ nguyên alerts và quyết
định; hai instance không ảnh hưởng nhau; reset đúng capacity; invalid batch
không phá concept; zero variance không gây NaN chảy vào window; snapshot ở
batch phát hiện vẫn chứa evidence; alert_rate xét ngay khi cửa sổ vừa đầy.

### Giai đoạn C — chuẩn hóa runner và sensitivity

7. **run_imprtotal.py**: chuyển thành main() + argparse. Nhận --data, --config,
   --output-dir và manifest. Dùng c=2 baseline; tạo KSWIN mới rõ ràng. Log
   versions, source paths, dataset hash, feature list, scaling, seed, solver,
   rounding. Đọc last_result thay vì tính lại alert từ state đã reset.
   Ghi batch_id0 và batch_number1; boundary lấy từ manifest. Chốt chính sách
   batch cuối: xử lý khi hợp lệ hoặc bỏ có ghi số mẫu, không âm thầm.
   Tính độ trễ theo số batch mới đã tiêu thụ và chỉ số batch, ghi cả hai để
   tránh sai một batch. Giới hạn vùng ghép detection↔boundary; không mặc định
   mọi detection nhiều batch về sau là true positive.
8. **test_c_values.py**: dùng cùng preprocessing/config với runner; tạo detector
   riêng mỗi lượt; giữ sweep c=2..6 để đối chiếu Table 3 và thêm sweep seed.
   KSWIN sw/ss được khảo sát ở nhóm cấu hình khác. Xuất denominator alert,
   warmup, reset, miss, event trước boundary và delay, không chỉ count.
9. **diagnostics/check_imprtotal.py, diagnostics/paper_audit.py**: giữ làm
   công cụ đối chiếu, cập nhật theo API snapshot mới; ghi rõ replay và full
   stream. Chạy lại parity upstream sau khi sửa RNG/state.

Tiêu chí C: cùng config/input cho kết quả lặp lại; import runner không đọc CSV
hoặc chạy thí nghiệm; CSV/log chứa evidence đầy đủ; c=2 và c=3 không bị trộn.

### Giai đoạn D — dữ liệu và kiểm thử

10. **datasets-main/prepare_impr_stratified.py**: thêm main()/CLI, lưu seed,
    quota, index dòng nguồn, schema/feature selection và SHA256 vào manifest.
    Tính ordered intersection của hai nguồn, loại label/metadata rõ ràng.
    Giữ undersample/shuffle từng năm. Không tự ghi đè dataset hiện có khi audit;
    dùng tên/output directory mới cho các subset thử nghiệm.
11. **datasets-main/create_imprtotal.py**: đọc manifest, kiểm tra schema và
    counts, nối nguyên thứ tự 2017→2018, ghi boundary=len(df17), hashes input
    và output. Thay assert bằng lỗi kiểm tra rõ vì assert có thể bị tắt.
12. **datasets-main/split_batches.py**: đọc batch_size/boundary từ config và
    manifest; xử lý batch Mixed và remainder; gọi đây là source-change boundary.
    Không hardcode 160000/80 khi thay subset hoặc batch size.
13. **datasets-main/compare_columns.py**: xuất ordered intersection và metadata
    loại khỏi features; kiểm tra kiểu và thứ tự cột giữa hai nguồn.
14. **tests/test_mpcawin.py, tests/test_kswin_compat.py,
    tests/test_runner.py, tests/test_dataset_manifest.py** (mới): kiểm thử
    ownership, reset, finite/shape/schema, fit thất bại, legacy parity, window
    boundaries, event snapshot, batch indexing và manifest consistency.
    Kiểm thử tích hợp gồm stream ổn định, outlier đơn lẻ, covariance drift,
    hai drift liên tiếp và ImprTotal. Mean shift/isotropic scale shift được ghi
    là giới hạn của eigenvalue-angle, không ép test phải phát hiện mọi drift.

Tiêu chí D: không leak label/boundary vào model; dữ liệu có thể truy ngược và
tái lập; báo cáo nhiều seed/cấu hình bằng miss rate, delay và báo động trước
boundary. Kết quả trên một seed không đủ để chốt bản cải tiến.

## 7. Điều kiện kết luận đã chạy đúng

- Lỗi kỹ thuật có kiểm thử hồi quy và đã được sửa.
- Baseline bám mã tác giả được giữ độc lập với biến thể cải tiến.
- Dataset/source/version/config được ghi đầy đủ và tái lập được.
- Với subset tự tạo, có thể kết luận “chạy đúng implementation và đo được miss”
  ngay cả khi baseline chưa phát hiện; không sửa ground truth hoặc ép tín hiệu.
- Chỉ tuyên bố tái lập delay của bài báo khi có đủ thông tin subset/timing của
  tác giả và cùng điều kiện thí nghiệm. Nếu thiếu, ghi rõ giới hạn tái lập.

Trạng thái khi lập kế hoạch: chỉ bổ sung bản đối chiếu và công cụ chẩn đoán.
Sau khi được chấp thuận, các file trong kế hoạch đã được sửa; dataset CSV cũ
được giữ nguyên và bổ sung manifest. Kết quả chi tiết nằm trong báo cáo cập nhật.
