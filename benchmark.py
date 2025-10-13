import os
import time
import torch
from sentence_transformers import SentenceTransformer
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Tạo danh sách 1000 câu ngắn để benchmark
sentences = [f"Câu số {i}" for i in range(1000)]

# Hàm đo thời gian xử lý embedding
def benchmark_embedding(device):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device=device)
    start_time = time.time()
    embeddings = model.encode(sentences, show_progress_bar=False)
    end_time = time.time()
    return end_time - start_time

# Benchmark trên CPU
cpu_time = benchmark_embedding("cpu")

# Benchmark trên GPU (nếu có)
gpu_time = None
if torch.cuda.is_available():
    gpu_time = benchmark_embedding("cuda")

# In kết quả so sánh
print(f"Thời gian xử lý embedding trên CPU: {cpu_time:.2f} giây")
if gpu_time is not None:
    print(f"Thời gian xử lý embedding trên GPU: {gpu_time:.2f} giây")
else:
    print("Không có GPU khả dụng để benchmark.")