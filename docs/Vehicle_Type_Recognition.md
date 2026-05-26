**KẾ HOẠCH TRIỂN KHAI ĐỒ ÁN NÂNG CAO**

**NHẬN DẠNG PHƯƠNG TIỆN GIAO THÔNG (VEHICLE TYPE RECOGNITION) **

**1. TỔNG QUAN DỰ ÁN & QUY MÔ DATASET CẬP NHẬT**

Đồ án xây dựng hệ thống Phân loại hình ảnh (Image Classification) nhằm nhận diện chính xác từ 5 đến 7 loại phương tiện giao thông (bao gồm các lớp cốt lõi: bike, motorbike, car, bus, truck và các lớp mở rộng). Quy mô tập dữ liệu dự kiến đạt từ 1.500 đến 2.000 bức ảnh cho mỗi loại phương tiện, nâng tổng số lượng ảnh toàn bộ tập dữ liệu lên khoảng 7.500 đến 14.000 ảnh màu RGB. Đây là quy mô lý tưởng giúp mô hình Deep Learning học được các đặc trưng tổng quát, tránh hiện tượng quá khớp (overfitting) và đảm bảo thời gian huấn luyện tối ưu.

**1.1. Cấu trúc cây thư mục dự án tiêu chuẩn**

VehicleTypeRecognition/
├── data/
│   ├── raw/                # Dữ liệu ảnh thô 5-7 lớp (1500-2000 ảnh/lớp) tải về từ các nguồn
│   └── processed/          # Dữ liệu sau khi xử lý OpenCV (Resize, Padding) và áp dụng Bộ lọc
│       ├── train/          # Tập huấn luyện (80% dữ liệu tổng)
│       ├── valid/          # Tập kiểm định đặc thù (10% dữ liệu tổng, chứa 5% Train copy)
│       └── test/           # Tập kiểm thử độc lập (10% dữ liệu tổng)
├── src/
│   ├── data_prep.py        # Pipeline xử lý hình ảnh (OpenCV Filters, Padding, Custom Split)
│   ├── dataset.py          # Bộ nạp dữ liệu (PyTorch/TensorFlow DataLoader)
│   ├── model.py            # Khởi tạo kiến trúc 3 mô hình đối sánh
│   └── train.py            # Vòng lặp huấn luyện, tối ưu siêu tham số
├── models/                 # Nơi lưu trữ file trọng số tốt nhất (.pth / .h5)
└── app.py                  # Giao diện tương tác thời gian thực Streamlit

**2. CHI TIẾT KHÂU TIỀN XỬ LÝ & CHIẾN LƯỢC BỘ LỌC MIỀN KHÔNG GIAN**

Vì tập dữ liệu sử dụng hoàn toàn là ảnh màu RGB (3 kênh), hệ thống sẽ áp dụng đồng bộ các bộ lọc trên Miền Không Gian (Spatial Domain) thay vì Miền Tần Số (Frequency Domain). Lý do cốt lõi là các mạng nơ-ron tích chập (CNN) xử lý trích xuất đặc trưng dựa trên các ma trận chập không gian. Bộ lọc miền không gian giúp tính toán trực tiếp trên các giá trị pixel của từng kênh màu nhanh chóng, bảo toàn phân phối màu sắc và đường nét hình học, đồng thời có tốc độ thực thi thời gian thực vượt trội, không tốn chi phí biến đổi Fourier (FFT) ngược xuôi.

Trước khi áp dụng các bộ lọc chuyên sâu, mọi bức ảnh đều trải qua bước chuẩn hóa cấu trúc hình học:
• Resize & Padding: Đưa tất cả ảnh về kích thước chuẩn 224x224. Sử dụng Padding viền đen để bảo toàn Tỷ lệ khung hình (Aspect Ratio), giúp phương tiện không bị méo, biến dạng cấu trúc hình học trước khi đưa vào mạng CNN.

**1. Nhóm bộ lọc Làm trơn / Lọc thông thấp (Smoothing / Lowpass Filters)**

Mục tiêu nhằm làm giảm nhiễu gai, mờ hóa cục bộ để mô hình tập trung vào các khối hình dáng lớn của phương tiện.
• Lọc trung bình / Lọc hộp (Averaging / Box filter): Thay thế giá trị pixel bằng trung bình cộng các pixel lân cận, làm mờ nhanh nhưng giảm độ sắc nét cạnh.
• Lọc Gaussian (Gaussian filter): Sử dụng trọng số dạng phân phối chuẩn Gaussian, giúp làm mịn ảnh tự nhiên hơn, loại bỏ nhiễu tần số cao rất hiệu quả mà vẫn giữ được cấu trúc biên tốt hơn lọc hộp.

**2. Nhóm bộ lọc Trung bình khử nhiễu (Mean Filters)**

Phù hợp để xử lý các dạng nhiễu hạt, nhiễu phân phối toán học trên các kênh màu RGB.
• Lọc trung bình số học (Arithmetic mean filter): Làm mịn các biến động nhiễu cục bộ bằng cách tính trung bình vùng.
• Lọc trung bình hình học (Geometric mean filter): Giúp giữ lại nhiều chi tiết tinh tế của ảnh hơn so với trung bình số học.
• Lọc trung bình điều hòa (Harmonic mean filter): Hoạt động tốt đối với nhiễu kiểu 'nhiễu muối' (salt noise) nhưng không tốt với 'nhiễu tiêu'.
• Lọc trung bình Contraharmonic (Contraharmonic mean filter): Phù hợp nhất để khử các xung nhiễu dạng cô lập (như nhiễu muối tiêu), tùy thuộc vào việc chọn hệ số lệnh Q âm hay dương.

**3. Nhóm bộ lọc Thống kê thứ tự / Phi tuyến (Order-Statistic / Nonlinear Filters)**

Dựa trên việc sắp xếp thứ tự các pixel trong vùng cửa sổ trượt, cực kỳ mạnh mẽ khi xử lý nhiễu xung.
• Lọc trung vị (Median filter): Thay thế pixel bằng giá trị nằm ở giữa chuỗi đã sắp xếp. Đây là bộ lọc quốc dân để loại bỏ hoàn toàn nhiễu muối tiêu trên ảnh RGB mà không làm mờ các đường biên góc cạnh của xe.
• Lọc Max (Max filter): Chọn giá trị lớn nhất, giúp tìm các điểm sáng cục bộ hoặc khử nhiễu tiêu.
• Lọc Min (Min filter): Chọn giá trị nhỏ nhất, giúp tìm các điểm tối cục bộ hoặc khử nhiễu muối.
• Lọc điểm giữa (Midpoint filter): Tính trung bình cộng giữa Max và Min, kết hợp tối ưu cho nhiễu có phân phối đồng đều.
• Lọc trung bình xén Alpha (Alpha-trimmed mean filter): Loại bỏ một số lượng pixel nhỏ nhất và lớn nhất ở hai đầu chuỗi rồi mới tính trung bình, rất hiệu quả khi ảnh bị dính hỗn hợp nhiều loại nhiễu.

**4. Nhóm bộ lọc Làm sắc nét / Lọc thông cao (Sharpening / Highpass Filters)**

Nhằm làm nổi bật các chi tiết góc cạnh, viền cơ cấu của xe (như bánh xe, lưới tản nhiệt, khung kính) giúp mạng CNN bắt đặc trưng tốt hơn.
• Toán tử Laplacian: Sử dụng đạo hàm bậc hai để phát hiện các điểm thay đổi độ sáng đột ngột, làm nổi bật đường biên theo mọi hướng.
• Toán tử Gradient (Sobel, Roberts, Scharr): Sử dụng đạo hàm bậc nhất để tính toán độ dốc góc biên theo hướng ngang và dọc. Mặt nạ Scharr cung cấp độ chính xác xấp xỉ đạo hàm tốt hơn Sobel ở các vùng biên chéo.
• Mặt nạ làm mờ (Unsharp masking): Trừ một phiên bản làm mờ khỏi ảnh gốc để tạo ra một mặt nạ biên, sau đó cộng ngược lại để làm ảnh sắc nét rõ rệt.
• Lọc tăng cường cao (Highboost filtering): Biến thể nâng cao của Unsharp masking, nhân thêm hệ số khuếch đại cho ảnh gốc giúp giữ lại phần nền tốt hơn trong khi tăng mạnh độ nét đường biên.

**3. THUẬT TOÁN CUSTOM SPLIT DATA ĐẶC THÙ THEO YÊU CẦU**

Để tuân thủ tuyệt đối yêu cầu khắt khe của Giảng viên hướng dẫn: Tập Valid (10%) bắt buộc phải được cấu thành từ 5% ảnh trùng với tập Train và 5% ảnh hoàn toàn mới chưa từng xuất hiện trong cả Train và Test. Thuật toán phân rã mảng dữ liệu được thực hiện như sau:

| Tập dữ liệu | Tỷ lệ chuẩn | Cách thức trích xuất thuật toán | Vai trò thực tế trong AI |

| --- | --- | --- | --- |

| Tập TEST | 10% | Trích xuất ngẫu nhiên 10% từ tổng dữ liệu ban đầu và tách biệt hoàn toàn. | Đánh giá khách quan năng lực tổng quát hóa của AI. |

| Tập TRAIN | 80% | Lấy toàn bộ 80% từ phần dữ liệu còn lại sau khi đã trích Test và phần Unseen Valid. | Nguồn tri thức chính để mô hình học tập cập nhật trọng số. |

| Tập VALID (Phần 1 - 5%) | 5% | Trích xuất ngẫu nhiên từ nhóm dữ liệu mới hoàn toàn (chưa từng đưa vào Train và Test). | Kiểm tra xem mô hình có bị hiện tượng học vẹt (Overfitting) trên dữ liệu mới hay không. |

| Tập VALID (Phần 2 - 5%) | 5% | Sao chép nguyên bản (Copy) ngẫu nhiên 5% lượng ảnh từ bên trong tập Train đã tạo. | Đánh giá sai số trực quan trực tiếp giữa dữ liệu cũ và dữ liệu mới. |



**4. ĐỐI SÁNH KIẾN TRÚC MÔ HÌNH & TRIỂN KHAI ỨNG DỤNG**

Đồ án tiến hành thực nghiệm song song Transfer Learning trên 3 kiến trúc mạng nơ-ron tích chập nổi tiếng để tìm ra mô hình tối ưu nhất:
1. ResNet-50: Kiến trúc kết nối tắt (Skip Connections), độ chính xác cao, làm chuẩn đối sánh vững chắc.
2. MobileNet-V3: Tối ưu tích chập tách biệt chiều sâu, siêu nhẹ và siêu nhanh, phù hợp cho triển khai nhúng và thực tế.
3. EfficientNet-B0: Mở rộng đồng đều đa chiều, mang lại sự cân bằng hoàn hảo giữa hiệu năng và tài nguyên mạng.

Ứng dụng thực tế app.py được xây dựng thông qua nền tảng Streamlit, cho phép người dùng upload ảnh, tự động chạy khâu tiền xử lý hình học kết hợp với Bộ lọc không gian tối ưu đã chọn, sau đó xuất kết quả nhận dạng kèm độ tự tin (Confidence score) theo thời gian thực.