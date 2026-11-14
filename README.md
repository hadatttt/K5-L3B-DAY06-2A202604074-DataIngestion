# Bài làm — Day 06 Data Ingestion (corpus: Python PEPs)

| Thành phần | Đường dẫn |
|---|---|
| Thông tin học viên | [STUDENT.md](STUDENT.md) |
| Code pipeline | [starter/src/](starter/src/) (`run.py` điều phối, `ingestion.py`, `chunking.py`, `embeddings.py`, `store.py`, `evaluation.py`, `audit.py`) |
| Corpus gốc + manifest | [starter/data/raw/](starter/data/raw/), [data/manifest.json](data/manifest.json) |
| Benchmark (20 câu, 15 dev / 5 test, gold evidence) | [benchmark/questions.json](benchmark/questions.json) |
| Nhật ký thí nghiệm A/B/C | [artifacts/experiment_log.csv](artifacts/experiment_log.csv) |
| Trace, chỉ số, mẫu audit, nhãn thủ công | [starter/results/](starter/results/) |
| Báo cáo cá nhân | [report/individual/2A202604074_HaVanKhanhDat.md](report/individual/2A202604074_HaVanKhanhDat.md) |

**Stack:** embedding local `BAAI/bge-small-en-v1.5` (384 chiều, sentence-transformers, CPU) + **PostgreSQL 16 + pgvector 0.8.7** (Docker Compose của starter, cosine `<=>`, tìm kiếm chính xác, metadata filter trong SQL). Không dùng API, chi phí 0. Chroma 1.5.9 vẫn dùng được làm store thứ hai (`VECTOR_STORE=chroma` trong `.env`); hai store cho cùng top-3 trên 60/60 truy vấn.

**Chạy lại (Windows PowerShell, từ thư mục `starter/`, Python 3.11, cần Docker Desktop):**

```bash
docker compose -p day6_student_starter up -d --wait
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env
.venv\Scripts\python scripts/check_setup.py   # Database / PostgreSQL / pgvector -> Setup OK
.venv\Scripts\python -m src.run all           # prepare -> validate -> ingest x2 (không trùng) -> evaluate
.venv\Scripts\python -m src.audit             # mẫu chấm coherence / metadata / grounding
.venv\Scripts\python -m scripts.reingest_check  # so top-3 với lần chạy Chroma + mô phỏng tài liệu thay đổi
```

Nếu database không chạy, pipeline báo lỗi và dừng, không tự chuyển sang store khác.

Lần đầu tải model (~130 MB) từ Hugging Face. Corpus đã có sẵn trong `starter/data/raw/`; nếu muốn tải lại: `https://raw.githubusercontent.com/python/peps/main/peps/pep-XXXX.rst` cho các PEP liệt kê trong manifest (bản `main` có thể đã đổi — so `sha256` trong manifest).

Kết quả chính (Hit@1 / Evidence Hit@3): A dev 7/15, 11/15 · test 3/5, 5/5 — B dev 10/15, 11/15 · test 4/5, 4/5 — C dev 10/15, 11/15 · test 4/5, 4/5 (C tuân thủ filter 5/5 câu).

---

# Đề bài gốc

# Day 06 · Data Ingestion Engineering

**Lab cá nhân** · **4 giờ** · **Dữ liệu tự chọn**

Biến tài liệu thành kho tri thức có thể truy xuất cho AI agent, rồi đo xem cách chuẩn bị dữ liệu nào hiệu quả nhất.

| 📄 Dữ liệu | 🧩 Chiến lược | 🎯 Benchmark |
|---|---|---|
| Tối thiểu **30 trang nội dung** | Tối thiểu **3 cấu hình** | Tối thiểu **20 câu hỏi** |

## Bắt đầu nhanh

Dùng [starter PostgreSQL + pgvector](starter/README.md) để dựng database và hoàn thiện các hàm TODO, hoặc tự chọn vector store tương đương. Starter chỉ là khung: **bạn tự hiện thực pipeline và benchmark**.

## 1. Xây pipeline

![Đọc tài liệu, chuẩn hóa, chia chunk, tạo vector, lưu và kiểm tra](assets/ingestion-pipeline.svg)

Chọn tài liệu được phép sử dụng; giữ **ID tài liệu, nguồn, trang/vị trí** và metadata phù hợp. Dùng embedding model thật qua API hoặc chạy local; lưu và truy xuất bằng pgvector, Qdrant, Chroma hoặc một store tương đương.

## 2. So sánh và đánh giá

![Ba chiến lược được đánh giá trên cùng corpus và benchmark dev/test](assets/benchmark-workflow.svg)

Dùng cùng dữ liệu, model và câu hỏi cho cả ba cấu hình. Tách **15 câu dev / 5 câu test** trước khi chạy; chọn cấu hình trên dev, rồi đánh giá test sau khi cố định. Mỗi câu hỏi cần có đoạn bằng chứng và trang nguồn; thiết kế bộ câu hỏi và đánh giá theo [BENCHMARK.md](BENCHMARK.md).

Đánh giá theo năm khía cạnh: **độ chính xác truy xuất**, **tính mạch lạc của chunk**, **tính hữu dụng của metadata**, **chất lượng grounding** và **tác động của chiến lược dữ liệu**. Giải thích lựa chọn bằng kết quả, ví dụ lỗi và các đánh đổi.

## 3. Chuẩn bị bài nộp

1. Chạy pipeline và lưu cấu hình, kết quả top-3, trace cùng nhãn đánh giá thủ công.
2. Điền [thông tin học viên](STUDENT.md), [nhật ký thí nghiệm](artifacts/experiment_log.csv) và bản sao [mẫu báo cáo cá nhân](report/individual/MSSV_HoTen.md).
3. Kiểm tra mã nguồn, cách chạy, manifest dữ liệu và checklist trong [SUBMISSION.md](SUBMISSION.md).

## Tài liệu hướng dẫn

| Bạn cần tìm | Tài liệu |
|---|---|
| Setup baseline PostgreSQL + pgvector | [starter/README.md](starter/README.md) |
| Các bước làm và cách tính chỉ số | [GUIDE.md](GUIDE.md) |
| Tiến độ trong 4 giờ | [CHECKPOINTS.md](CHECKPOINTS.md) |
| Năm tiêu chí benchmark và bộ câu hỏi | [BENCHMARK.md](BENCHMARK.md) |
| Tiêu chí chấm điểm | [RUBRIC.md](RUBRIC.md) |
| Quy tắc sử dụng dữ liệu, AI và API | [RULES.md](RULES.md) |
| Cách nộp bài | [SUBMISSION.md](SUBMISSION.md) |

[data/format-example.json](data/format-example.json) chỉ minh họa định dạng, không phải corpus đủ điều kiện. Phần xử lý và benchmark do bạn tự hiện thực; không bắt buộc xây chatbot.
