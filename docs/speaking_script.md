# Kịch Bản Thuyết Trình Demo Hệ Thống (Presentation Speaking Script)

Kịch bản này được biên soạn với văn phong thuyết trình tự nhiên, chuyên nghiệp và trôi chảy, bám sát từng slide trong tệp PDF **"Xử lý ảnh.pdf"** (từ Slide 7 đến Slide 8.3).

---

## ─── PHẦN 7: DEMO HỆ THỐNG ───

### 🎤 Slide 1: 7. Demo hệ thống (Kiến trúc 3 lớp)
* **Nội dung slide:** Giới thiệu cấu trúc 3 lớp (3-tier architecture) rời rạc của hệ thống demo.
* **Lời thoại thuyết trình:**
  > "Kính thưa thầy cô và hội đồng, tiếp theo em xin phép được đại diện nhóm trình bày về phần Demo hệ thống nhận diện phương tiện giao thông. 
  > 
  > Để đảm bảo tính mô-đun hóa, dễ bảo trì và mở rộng sau này, nhóm chúng em đã thiết kế ứng dụng theo kiến trúc 3 lớp tách biệt hoàn toàn, bao gồm: Lớp trình diễn (Frontend), Lớp ứng dụng trung gian (Backend API) và Lớp suy luận mô hình (Inference Layer). Việc tách rời này giúp chúng em dễ dàng tích hợp và thử nghiệm các kiến trúc học sâu khác nhau mà không làm ảnh hưởng đến giao diện người dùng."

---

### 🎤 Slide 2: Chi tiết các lớp trong kiến trúc (Frontend - Backend - Models)
* **Nội dung slide:** Mô tả chi tiết vai trò của Frontend (Vite + React), Backend (Flask API), và Lớp mô hình (LoadedModel Wrapper).
* **Lời thoại thuyết trình:**
  > "Đi sâu vào chi tiết của từng lớp:
  > * **Phía Frontend:** Nhóm phát triển trên nền tảng React phối hợp với công cụ Vite để tối ưu hóa tốc độ tải trang. Giao diện được thiết kế hiện đại, thân thiện, hỗ trợ kéo-thả ảnh trực tiếp và hiển thị kết quả phân tích thời gian thực bằng biểu đồ trực quan. Việc kết nối tới backend được thực hiện qua các cuộc gọi HTTP bất đồng bộ bằng thư viện Axios.
  > * **Phía Backend:** Đóng vai trò trung gian xử lý logic ứng dụng, được xây dựng bằng Flask API và chia nhỏ thành các Blueprint như `predict.py` để xử lý nhận diện và `metrics.py` để đọc lịch sử huấn luyện từ các tệp JSON tĩnh.
  > * **Lớp mô hình (Model & Inference Layer):** Nhóm thiết kế một lớp Wrapper thống nhất tên là `LoadedModel` để trừu tượng hóa các framework lưu trữ. Dù là mô hình PyTorch tiêu chuẩn dạng `.pth` (dành cho ResNet-50 và ViT) hay mô hình Ultralytics YOLO dạng `.pt`, backend đều nạp và gọi suy luận một cách đồng bộ."

---

### 🎤 Slide 3: 7.1. Luồng xử lý ảnh và kết quả trả về
* **Nội dung slide:** Sơ đồ 5 bước xử lý tuần tự từ lúc tải ảnh đến lúc trả về client.
* **Lời thoại thuyết trình:**
  > "Trên Slide là sơ đồ sequence chi tiết minh họa luồng đi của dữ liệu qua 5 bước tuần tự:
  > * **Bước 1:** Người dùng tải ảnh lên giao diện React, chọn mô hình và cấu hình pipeline. Hệ thống gửi yêu cầu POST đến Flask.
  > * **Bước 2:** Flask chuyển tiếp bytes ảnh sang Preprocessing để thực hiện Base Pipeline (co giãn giữ tỷ lệ và bù viền đen về kích thước 224x224), đồng thời áp dụng hiệu ứng nhiễu thời tiết hoặc mờ/nét tùy chọn.
  > * **Bước 3:** Trả ảnh PIL và Tensor chuẩn hóa về Flask.
  > * **Bước 4:** Flask đẩy dữ liệu vào mô hình AI. Với ResNet và ViT, mô hình nhận Tensor và chạy Softmax; còn với YOLO, mô hình nhận trực tiếp đối tượng ảnh PIL.
  > * **Bước 5:** Flask thu nhận kết quả 10 lớp phương tiện, mã hóa Base64 ảnh đã tiền xử lý, đo trễ (latency) và trả về JSON để React kết xuất lên màn hình."

---

### 🎤 Slide 4: 7.2. Giao diện demo - Tab 1: Thử nghiệm nhận dạng (Inference Tab)
* **Nội dung slide:** Trực quan hóa chức năng nhận diện phương tiện trên giao diện thực tế.
* **Lời thoại thuyết trình:**
  > "Đây là giao diện thực tế của Tab đầu tiên: **Thử nghiệm nhận dạng (Inference Tab)**. 
  > 
  > Ở cột bên trái, người dùng có thể chọn mô hình trong danh sách 12 phiên bản (đã được nhóm phân theo Backbone ResNet, YOLO, ViT để dễ quản lý) và thuật toán tiền xử lý. 
  > Ở giữa là ảnh gốc người dùng tải lên bên cạnh ảnh kết quả sau khi đi qua pipeline tiền xử lý (ở đây là hiệu ứng motion blur). 
  > Cột bên phải hiển thị bảng phân phối xác suất độ tự tin của cả 10 lớp phương tiện giao thông, được sắp xếp giảm dần và cập nhật theo thời gian thực."

---

### 🎤 Slide 5: 7.2. Giao diện demo - Tab 2: Dashboard (Phần Tổng quan)
* **Nội dung slide:** Giao diện Dashboard hiển thị số liệu thống kê tổng quan của mô hình.
* **Lời thoại thuyết trình:**
  > "Bên cạnh việc thử nghiệm nhận dạng đơn lẻ, hệ thống còn tích hợp **Bảng điều khiển phân tích (Dashboard Tab)**. 
  > 
  > Tại Tab Tổng quan này, khi người dùng lựa chọn cấu hình thí nghiệm (Mô hình $\times$ Dataset $\times$ Phiên bản), hệ thống sẽ tự động hiển thị các chỉ số hiệu năng cốt lõi trên tập Test độc lập như: Độ chính xác (Accuracy), F1-score trung bình, Epoch đạt kết quả tốt nhất và độ chính xác tương ứng trên tập Validation unseen. 
  > Phía dưới là biểu đồ thanh phân phối cơ cấu số lượng ảnh của tập dữ liệu huấn luyện, giúp người dùng kiểm tra nhanh sự phân bổ các lớp nhiễu."

---

### 🎤 Slide 6: 7.2. Giao diện demo - Tab 2: Dashboard (Diễn biến huấn luyện & F1 lớp)
* **Nội dung slide:** Biểu đồ đường của Loss/Accuracy qua các Epoch và độ chính xác chi tiết của từng lớp.
* **Lời thoại thuyết trình:**
  > "Cuộn xuống dưới trong Tab tổng quan, người dùng có thể quan sát biểu đồ đường (Line Chart) biểu diễn diễn biến giá trị Loss và Accuracy của tập Train và Validation qua từng Epoch huấn luyện. Đồ thị này giúp chúng em theo dõi trực quan hiện tượng hội tụ và kiểm soát overfitting. 
  > Phía dưới là biểu đồ hiệu quả của từng lớp phương tiện, xếp hạng điểm F1-score từ cao xuống thấp để nhận diện nhanh lớp nào đang phân loại tốt và lớp nào đang là điểm yếu của mô hình."

---

### 🎤 Slide 7: 7.2. Giao diện demo - Tab 2: Dashboard (Chi tiết phân tích & Ma trận nhầm lẫn)
* **Nội dung slide:** Chi tiết bảng số liệu và Confusion Matrix trực quan.
* **Lời thoại thuyết trình:**
  > "Chuyển sang Tab phụ **'Phân tích'**, hệ thống cung cấp bảng số liệu chi tiết về Precision, Recall, F1-score cho từng lớp trên tập Test độc lập. 
  > 
  > Đặc biệt, ma trận nhầm lẫn (Confusion Matrix) được trực quan hóa động bằng màu sắc. Nhóm đã tối ưu hóa hiển thị bằng cách tách biệt thang màu: đường chéo chính (dự đoán đúng) được tô sắc xanh/teal, các ô ngoài đường chéo (dự đoán sai/nhầm lẫn) được tự động co giãn sắc đỏ dựa theo điểm nhầm lẫn lớn nhất của mô hình. Điều này giúp người dùng phát hiện ngay lập tức các cặp lớp xe hay bị nhầm lẫn với nhau."

---

### 🎤 Slide 8: 7.2. Giao diện demo - Tab 2: Dashboard (Top cặp nhầm lẫn & Tóm tắt Train)
* **Nội dung slide:** Thống kê top các lỗi nhầm lẫn nhiều nhất và tóm tắt thời gian chạy huấn luyện.
* **Lời thoại thuyết trình:**
  > "Để hỗ trợ việc phân tích lỗi sâu hơn, hệ thống tự động lọc ra danh sách **Các cặp nhầm lẫn nhiều nhất** (như car nhầm thành truck, taxi nhầm thành car) kèm theo tỷ lệ phần trăm cụ thể trên tổng mẫu thực tế. 
  > 
  > Phía dưới cùng là phần tóm tắt quá trình huấn luyện thực tế trên Kaggle, ghi nhận tiêu chí lựa chọn checkpoint tốt nhất, giá trị loss/accuracy tương ứng và tổng thời gian chạy huấn luyện (ở đây ResNet-50 chạy mất 1 giờ 50 phút)."

---

### 🎤 Slide 9: 7.2. Giao diện demo - Tab 2: Dashboard (Tab So sánh mô hình)
* **Nội dung slide:** So sánh trực tiếp hiệu năng giữa Thí nghiệm A và Thí nghiệm B.
* **Lời thoại thuyết trình:**
  > "Một tính năng rất hữu ích của Dashboard là **Tab So sánh**. 
  > 
  > Người dùng có thể đặt hai thí nghiệm khác nhau lên bàn cân (ví dụ ở đây là so sánh mô hình ResNet-50 chạy trên dữ liệu nhiễu thời tiết V1 với dữ liệu nhiễu mờ/nét V2). Hệ thống sẽ tự động tính toán độ chênh lệch (Delta) của Accuracy, Macro F1 và Weighted F1, kèm theo mũi tên chỉ hướng cải thiện hay suy giảm để nhóm dễ dàng đánh giá tác động của các chiến lược tăng cường dữ liệu."

---

### 🎤 Slide 10: 7.3. Thời gian phản hồi (Latency)
* **Nội dung slide:** Bảng so sánh kích thước mô hình và độ trễ suy luận trung bình trên CPU.
* **Lời thoại thuyết trình:**
  > "Về khía cạnh hiệu năng vận hành thực tế, nhóm đã tiến hành đo đạc thời gian phản hồi (suy luận) trung bình của 3 kiến trúc mô hình trên CPU:
  > * **YOLOv8n-cls** có kích thước siêu nhẹ (~12MB file nạp, tham số thực tế ~3MB), cho phản hồi cực nhanh chỉ **50ms**, mang lại trải nghiệm mượt mà lập tức trên môi trường web và thiết bị nhúng.
  > * **ResNet-50** dung lượng trung bình (~98MB tham số thô), đạt độ trễ **130ms**, đáp ứng rất tốt yêu cầu thời gian thực.
  > * **ViT-B/16** có kích thước lớn nhất (~328MB tham số thô), độ trễ đạt gần **290ms**. Tuy có trễ nhẹ nhưng tốc độ này vẫn hoàn toàn chấp nhận được đối với các tác vụ phân tích ảnh đơn lẻ."

---

### 🎤 Slide 11: 7.3.2. Xử lý lỗi hệ thống (Error Handling)
* **Nội dung slide:** Kiểm soát định dạng đầu vào, CORS và ngoại lệ mô hình.
* **Lời thoại thuyết trình:**
  > "Để hệ thống vận hành ổn định trong môi trường thực tế, nhóm đã triển khai các cơ chế xử lý lỗi chặt chẽ:
  > * **Kiểm soát đầu vào:** Chỉ chấp nhận ảnh có đuôi `.jpg`, `.jpeg`, `.png` và giới hạn dung lượng dưới 10MB để ngăn chặn tấn công tràn bộ nhớ đệm (Buffer Overflow) của API. Nếu thiếu file, API trả ngay lỗi 400 Bad Request.
  > * **Ngoại lệ mô hình:** Nếu tệp trọng số bị hỏng hoặc thiếu, hệ thống sẽ ném ra lỗi kiểm soát để Flask không bị sập đột ngột và hiển thị cảnh báo chi tiết cho người dùng kiểm tra đường dẫn checkpoint.
  > * **Cấu hình CORS:** Kích hoạt CORS trên Flask để cho phép giao tiếp an toàn, xuyên suốt với frontend React chạy cổng 5173 mà không bị trình duyệt chặn tài nguyên."

---

## ─── PHẦN 8: KẾT LUẬN & HƯỚNG PHÁT TRIỂN ───

### 🎤 Slide 12: 8. Kết luận - 8.1. Mô hình tốt nhất (Raw Original)
* **Nội dung slide:** Bảng so sánh hiệu năng trên tập dữ liệu thô chưa lọc nhiễu (Raw Original).
* **Lời thoại thuyết trình:**
  > "Sau quá trình nghiên cứu thực nghiệm, chúng em rút ra kết luận về mô hình tốt nhất. 
  > 
  > Đối với nhóm thí nghiệm trên **Tập dữ liệu gốc chưa lọc nhiễu (Raw Original)**:
  > * **Vision Transformer (ViT)** đạt hiệu năng vượt trội nhất với Accuracy **94.06%** trên tập dữ liệu thời tiết V1 và **93.42%** trên tập dữ liệu mờ/nét V2. Điểm F1-score tương ứng đạt **92.31%** và **96.62%**.
  > * ResNet-50 xếp thứ hai và YOLOv8n-cls xếp cuối với độ chính xác dao động quanh mức 90%-91%, tuy nhiên YOLO lại chiến thắng tuyệt đối về tốc độ phản hồi."

---

### 🎤 Slide 13: 8. Kết luận - 8.1. Mô hình tốt nhất (Raw Cleaning)
* **Nội dung slide:** Bảng so sánh hiệu năng trên tập dữ liệu đã lọc sạch nhiễu thủ công (Raw Cleaning).
* **Lời thoại thuyết trình:**
  > "Khi chuyển sang **Tập dữ liệu đã được nhóm lọc sạch nhiễu thủ công (Raw Cleaning)**, hiệu năng của tất cả các mô hình đều ghi nhận mức tăng trưởng rõ rệt:
  > * **ViT** vẫn giữ vị thế dẫn đầu với độ chính xác đạt **94.37% (V1)** và **94.79% (V2)**, điểm F1-score tiệm cận mức **97.04% - 97.33%**.
  > * ResNet-50 tăng lên mức **93.01%** và YOLOv8n-cls cải thiện mạnh lên **92.08%**. 
  > 
  > Kết quả này chứng minh rằng việc đầu tư làm sạch dữ liệu ban đầu (data cleaning) đóng vai trò quyết định, giúp cải thiện đáng kể độ chính xác của các mô hình học sâu mà chưa cần thay đổi kiến trúc."

---

### 🎤 Slide 14: 8.2. Hạn chế (Nhầm lẫn lớp tương đồng cao)
* **Nội dung slide:** Lỗi nhầm lẫn Taxi vs Car và Minibus vs Bus/Car.
* **Lời thoại thuyết trình:**
  > "Dù đạt kết quả khả quan, hệ thống vẫn tồn tại một số hạn chế kỹ thuật:
  > 
  > Đầu tiên là sự nhầm lẫn giữa các lớp có độ tương đồng thị giác cao (Fine-grained classification):
  > * **Taxi nhầm với Car:** Đây là lỗi phổ biến nhất ở cả 3 mô hình (ViT nhầm 17 ảnh, YOLO nhầm 33 ảnh, ResNet nhầm 24 ảnh). Về đặc trưng thị giác, taxi là phân nhóm của ô tô con, chỉ khác biệt ở chiếc mào trên nóc hoặc tem logo. Khi các đặc trưng này bị mờ hoặc góc chụp che khuất, mô hình sẽ mặc định phân loại thành car.
  > * **Minibus nhầm với Bus/Car:** Lớp minibus có F1-score thấp nhất (chỉ 80.56% ở ViT) do đây là lớp trung gian về mặt kích thước, rất dễ bị nhầm với xe khách lớn hoặc xe hơi 7 chỗ tùy thuộc vào góc chụp xa hay gần của camera giám sát."

---

### 🎤 Slide 15: 8.2. Hạn chế (Mô hình phân loại thuần túy)
* **Nội dung slide:** Hạn chế của Image Classification so với Object Detection.
* **Lời thoại thuyết trình:**
  > "Hạn chế thứ hai là bản chất của mô hình phân loại ảnh thuần túy. 
  > 
  > Hệ thống hiện tại đang hoạt động dựa trên giả định phương tiện giao thông là chủ thể duy nhất chiếm phần lớn bức ảnh đầu vào. Tuy nhiên trong thực tế, ảnh từ camera giám sát giao thông luôn là cảnh toàn chứa nhiều đối tượng đan xen phức tạp. Một mô hình phân loại thuần túy không thể xác định vị trí tọa độ của từng xe (Bounding Box) hoặc đếm số lượng xe trong một khung cảnh rộng."

---

### 🎤 Slide 16: 8.2. Hạn chế (Phụ thuộc dữ liệu tăng cường giả lập)
* **Nội dung slide:** Phân tích hạn chế của các hiệu ứng nhiễu ngoại tuyến (offline).
* **Lời thoại thuyết trình:**
  > "Hạn chế thứ ba là sự phụ thuộc vào dữ liệu tăng cường giả lập. 
  > 
  > Các hiệu ứng thời tiết như mưa, nắng lóa hay đêm tối đều được nhóm chèn ngoại tuyến bằng thuật toán xử lý ảnh truyền thống (OpenCV/PIL). Các biến thể toán học này chưa phản ánh hoàn toàn tính chất quang học phức tạp của môi trường thực tế như hiện tượng khúc xạ ánh sáng qua hạt nước đọng trên ống kính camera, sương mù dày đặc che khuất chiều sâu hay hiện tượng lóa sáng do đèn pha của xe đi ngược chiều."

---

### 🎤 Slide 17: 8.2. Hạn chế (Chi phí tính toán cao của ViT)
* **Nội dung slide:** Hạn chế về phần cứng và thời gian phản hồi của Vision Transformer.
* **Lời thoại thuyết trình:**
  > "Cuối cùng là chi phí tính toán cao đối với mô hình Vision Transformer. 
  > 
  > Do trọng số của ViT-B/16 rất lớn (dung lượng tham số ~328MB, file lưu trữ thực tế ~543MB), mô hình đòi hỏi rất nhiều RAM và năng lượng xử lý. Điều này dẫn đến thời gian phản hồi đạt gần 300ms trên CPU. Độ trễ lớn này sẽ gây khó khăn rất lớn khi cần triển khai ứng dụng thực tế trên các thiết bị biên (Edge Devices) hoặc thiết bị nhúng có cấu hình phần cứng hạn chế."

---

### 🎤 Slide 18: 8.3. Hướng phát triển (Xử lý ảnh nâng cao & Object Detection)
* **Nội dung slide:** Tích hợp xử lý ảnh nâng cao và chuyển dịch sang Object Detection.
* **Lời thoại thuyết trình:**
  > "Để nâng cấp hệ thống vượt qua các hạn chế kỹ thuật nêu trên, nhóm đề xuất lộ trình phát triển tiếp theo gồm 4 hướng chính. 
  > 
  > Hướng thứ 1 và thứ 2 là:
  > * **Tích hợp các thuật toán xử lý ảnh nâng cao:** Nghiên cứu áp dụng thuật toán khử sương mù (Dehazing) giúp khôi phục chi tiết ảnh chụp xe trong thời tiết xấu; kết hợp với cân bằng sáng thích ứng CLAHE để cải thiện độ tương phản cho ảnh ban đêm hoặc ngược sáng trước khi đưa vào nhận diện.
  > * **Chuyển dịch sang Phát hiện đối tượng (Object Detection):** Nâng cấp mô hình từ YOLO phân loại (YOLO-cls) sang YOLO phát hiện đối tượng (YOLOv8-det hoặc YOLOv10) để vừa xác định tọa độ khung bao (Bounding Box), vừa nhận diện và đếm được nhiều phương tiện giao thông cùng lúc trong cảnh toàn phức tạp."

---

### 🎤 Slide 19: 8.3. Hướng phát triển (Tích hợp giải thích XAI & Tối ưu hóa mô hình)
* **Nội dung slide:** Tích hợp công nghệ Grad-CAM/Attention Map và tối ưu hóa suy luận (TensorRT, ONNX, Quantization).
* **Lời thoại thuyết trình:**
  > "Hướng thứ 3 và thứ 4 trong lộ trình phát triển tương lai là:
  > * **Bổ sung tính năng giải thích mô hình (Explainable AI - XAI):** Tích hợp công nghệ Grad-CAM đối với ResNet-50 và bản đồ Attention Map đối với Vision Transformer trực tiếp lên giao diện React. Việc này giúp người dùng nhìn thấy trực quan mô hình đang tập trung vào vùng đặc trưng nào của xe (ví dụ mào taxi, cabin hay bánh xe) trước khi đưa ra dự đoán, tăng tính minh bạch của AI.
  > * **Tối ưu hóa mô hình phục vụ triển khai (Model Optimization):** Áp dụng TensorRT, ONNX Runtime và đặc biệt là kỹ thuật lượng tử hóa mô hình (Quantization từ float32 sang int8) để giảm dung lượng tệp trọng số và rút ngắn độ trễ suy luận của ViT xuống dưới 30ms trên các thiết bị nhúng phần cứng yếu.
  > 
  > Phần thuyết trình về Demo hệ thống và Kết luận của nhóm chúng em đến đây là kết thúc. Em xin chân thành cảm ơn thầy cô và hội đồng đã lắng nghe!"
