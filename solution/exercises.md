# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Khi temperature tăng từ 0.0 lên 0.7, phản hồi trở nên đa dạng và sáng tạo hơn nhưng vẫn giữ được tính mạch lạc. Ở mức 1.2, câu trả lời bắt đầu có xu hướng lan man hoặc thêm chi tiết ít chắc chắn, còn tại 1.8 thì độ nhất quán giảm rõ rệt và phản hồi dễ kém mạch lạc nhất.


### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> Với trợ lý soạn thảo hợp đồng pháp lý, một công việc nghiêm túc và yêu cầu tính chính xác tuyệt đối, em sẽ cài đặt temperature = 0. Công việc trợ lý slogan quảng cáo cần có sự sáng tạo trong việc thiết kế nội dung, cho nên em mong muốn đầu ra của trợ lý cũng có sự sáng tạo, vậy nên biến temperature sẽ được cài đặt tương đối cao, tuy nhiên sẽ không quá cao để đảm bảo sự mạch lạc (có lẽ sẽ từ 0.5 đến 1.2 là hợp lý).

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> Mỗi ngày có 40.000 lượt gọi API, tương ứng 40.000 * 500 = 20.000.000 token đầu ra. Vì vậy, GPT-4 tốn khoảng 20 * 60 = 1200$/ngày, trong khi GPT-4o-mini chỉ tốn 20 * 0,60 = 12USD/ngày; GPT-4 đắt hơn khoảng 100 lần.

GPT-4 xứng đáng với chi phí khi cần xử lý các yêu cầu phức tạp như phân tích tài liệu dài, lập luận nhiều bước hoặc giải quyết lỗi lập trình khó. GPT-4o-mini là lựa chọn phù hợp cho chatbot hỏi đáp, phân loại nội dung và các tác vụ đơn giản có lưu lượng lớn, nơi chi phí và tốc độ được ưu tiên.


---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> Phản hồi với persona nhà thơ thường giàu hình ảnh ví von, nhẹ nhàng, dễ hiểu và hầu như không sử dụng thuật ngữ kỹ thuật. Trong khi đó, persona kỹ sư phần mềm có giọng văn trực tiếp, chính xác hơn, giải thích theo cấu trúc rõ ràng và có thể kèm ví dụ code, nên mức độ kỹ thuật cũng cao hơn. Độ dài của phản hồi cũng có thể thay đổi tùy vai trò: nhà thơ thường cô đọng và giàu cảm xúc, còn kỹ sư có xu hướng trình bày chi tiết hơn. Qua đó, system prompt có thể điều khiển giọng văn, từ vựng, mức độ chuyên môn, độ dài, cách tổ chức nội dung và loại ví dụ được sử dụng.


### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> Với đoạn văn 150 từ, cách ước lượng thô cho (150 / 0,75 = 200) token, trong khi `count_tokens` trả về khoảng 260 token. Như vậy, số token thực tế cao hơn ước lượng khoảng ((260-200)/200 * 100% = 30%), hay nói cách khác phương pháp thô dự toán thiếu khoảng 23,1% tổng chi phí thực tế. Với ứng dụng tiếng Việt, cách lấy số từ chia cho 0,75 thường dẫn đến dự toán thiếu vì tokenizer có thể tách một từ có dấu thành nhiều token hoặc token con; mức chênh lệch cụ thể còn phụ thuộc vào nội dung và encoding của model.


---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> Trong ba ứng dụng trên, trợ lý giọng nói (b) hưởng lợi nhiều nhất từ streaming, trong khi pipeline dịch tài liệu chạy ngầm (c) hoàn toàn không cần đến tính năng này. Trợ lý giọng nói đòi hỏi phản hồi thời gian thực khắt khe nhất; nhờ streaming, hệ thống có thể bắt đầu tổng hợp và phát ngay đoạn âm thanh của câu đầu tiên trong lúc AI vẫn đang tiếp tục tạo ra phần nội dung còn lại, giúp xóa bỏ những khoảng lặng chờ đợi ngượng ngùng và tạo ra cuộc hội thoại tự nhiên (dù chatbot văn bản cũng cần streaming để người dùng có thể đọc ngay từng chữ, nhưng khoảng lặng trong âm thanh luôn gây khó chịu hơn việc chờ đợi trên màn hình). Ngược lại, pipeline dịch tài liệu ban đêm là một tác vụ xử lý hàng loạt bất đồng bộ (batch processing) không có sự tương tác trực tiếp của con người; do đó, hệ thống chỉ quan tâm đến việc nhận được toàn bộ văn bản hoàn chỉnh ở đầu ra để lưu trữ, khiến cho việc truyền dữ liệu từng mảnh nhỏ (streaming) trở nên không cần thiết và thậm chí có thể làm giảm hiệu suất tổng thể.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> Khi API quá tải, việc sử dụng thời gian chờ cố định để thử lại (retry) sẽ khiến hàng nghìn client tiếp tục gửi yêu cầu cùng lúc, làm máy chủ càng thêm cạn kiệt tài nguyên. Kỹ thuật exponential backoff giải quyết điều này bằng cách tăng thời gian chờ theo cấp số nhân sau mỗi lần thất bại, giúp giảm dần tần suất gửi yêu cầu và tạo khoảng trống cho máy chủ xử lý. Tuy nhiên, nếu các client bị ngắt kết nối tại cùng một thời điểm, các nhịp thời gian chờ giãn cách này vẫn có thể đồng bộ với nhau và tạo ra những đợt bùng nổ lưu lượng lặp lại. Kỹ thuật "jitter" được bổ sung để khắc phục triệt để lỗ hổng đó bằng cách cộng thêm một khoảng thời gian trễ ngẫu nhiên vào chu kỳ chờ của mỗi client. Sự ngẫu nhiên này phá vỡ hoàn toàn tính đồng bộ, giúp phân tán các yêu cầu thử lại rải rác một cách mượt mà trên trục thời gian để máy chủ có thể phục hồi an toàn.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> System prompt: Bạn là một trợ lý AI thân thiện. Hãy đưa ra câu trả lời thật đơn giản. Hai chỗ trong prompt đó là : Trợ lý AI thân thiện, nếu không đề cập phần này, thái độ và sắc thái trong câu trả lời của AI có thể thay đổi. Điểm thứ hai là câu trả lời đơn giản, yêu cầu này định hình sự phức tạp của câu trả lời.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> Giới hạn bộ nhớ 4 lượt sẽ khiến trợ lý ảo dễ dàng đánh mất các thông tin nền tảng được cung cấp từ đầu cuộc trò chuyện. Ví dụ như bạn mở đầu đoạn hội thoại với chủ đề bóng đá, nhưng trong 4-5 câu sau bạn lại chuyển sang cầu lông, mô hình sẽ quên mất câu chuyện bóng đá ở trước . Nguyên nhân là do các lượt trao đổi chi tiết tiếp theo đã đẩy những thông tin cốt lõi này ra khỏi bộ nhớ ngắn hạn của hệ thống. Để khắc phục vấn đề này, hệ thống có thể áp dụng kỹ thuật tóm tắt cửa sổ trượt nhằm liên tục cô đọng nội dung của các lượt cũ vào bộ nhớ gốc. Một phương án hiệu quả khác là sử dụng cơ sở dữ liệu vector để lưu trữ ngầm toàn bộ lịch sử và chủ động truy xuất lại thông tin ngữ nghĩa khi cần thiết. Cuối cùng, trợ lý có thể được thiết kế để tự động nhận diện và ghim các thực thể quan trọng vào một không gian độc lập, giúp duy trì bối cảnh xuyên suốt mà không bị ảnh hưởng bởi giới hạn số lượt trao đổi.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
