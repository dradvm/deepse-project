# Bài 1

import requests
import base64
import os
github_token = "GITHUB_TOKEN" # Tạo token từ GitHub settings
headers = {
    "Authorization": f"token {github_token}",
    "Accept": "application/vnd.github.v3+json"
}
# Hàm lấy nội dung file từ repository
def get_file_content(owner, repo, path, branch="main"):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = response.json()
    # Kiểm tra nếu là file
    if "type" in content and content["type"] == "file":
        return base64.b64decode(content["content"]).decode("utf-8")
    return None
# Lấy danh sách các file Python trong một repository
def get_python_files(owner, repo, path="", branch="main"):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    response = requests.get(url, headers=headers)
    files = []
    if response.status_code == 200:
        contents = response.json()
        for item in contents:
            if item["type"] == "file" and item["name"].endswith(".py"):
                files.append(item["path"])
            elif item["type"] == "dir":
            # Đệ quy cho thư mục con
                files.extend(get_python_files(owner, repo, item["path"], branch))
    return files
# Ví dụ sử dụng
owner = "tensorflow"
repo = "models"
python_files = get_python_files(owner, repo, path="official/legacy/bert",
branch="master")
print(f"Found {len(python_files)} Python files")
# Lưu các file vào thư mục local
os.makedirs("data/raw", exist_ok=True)

for i, file_path in enumerate(python_files[:10]): # Lấy 10 file đầu tiên
    content = get_file_content(owner, repo, file_path, branch="master")
    if content:
        local_path = f"data/raw/file_{i}.py"
        with open(local_path, "w", encoding="utf-8") as f: f.write(content)
        print(f"Saved to {local_path}")



# Bài 2

import ast
import re
def preprocess_python_code(code):
# Loại bỏ comments
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'"""[\s\S]*?"""', '', code)
    code = re.sub(r"'''[\s\S]*?'''", '', code)
# Chuẩn hóa khoảng trắng
    code = re.sub(r'\s+', ' ', code)
    return code.strip()
def extract_functions(code):
    try:
        tree = ast.parse(code)
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_code = ast.get_source_segment(code, node)
                functions.append({
                'name': node.name,
                'code': func_code,
                'processed_code': preprocess_python_code(func_code)
                })
        return functions
    except SyntaxError:
        return []
# Ví dụ sử dụng
import os
processed_data = []
for i in range(10): # Cho 10 file đã tải
    file_path = f"data/raw/file_{i}.py"

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        # Tiền xử lý và trích xuất hàm
            functions = extract_functions(code)
            processed_data.extend(functions)
            print(f"Extracted {len(processed_data)} functions")
    # Lưu dữ liệu đã xử lý
import pandas as pd
df = pd.DataFrame(processed_data)
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/functions.csv", index=False)

# Bài 3.1
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
# Đọc dữ liệu đã xử lý
df = pd.read_csv("data/processed/functions.csv")
# Khởi tạo TF-IDF vectorizer
tfidf = TfidfVectorizer(
    max_features=5000, # Giới hạn số lượng từ
    ngram_range=(1, 3), # Sử dụng unigram, bigram và trigram
    stop_words='english'
)
# Tạo ma trận TF-IDF
tfidf_matrix = tfidf.fit_transform(df['processed_code'])
# Chuyển ma trận thành DataFrame để dễ xem
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf.get_feature_names_out()
)
print(f"Shape of TF-IDF matrix: {tfidf_matrix.shape}")
print(tfidf_df.head())
# Lưu vectorizer để tái sử dụng
import pickle
import os
os.makedirs("models", exist_ok=True)
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
# Lưu ma trận TF-IDF
import numpy as np
np.save("data/processed/tfidf_matrix.npy", tfidf_matrix.toarray())

# Bài 3.2

import gensim
import pandas as pd
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
# Tải tokenizer
nltk.download('punkt')
nltk.download('punkt_tab')
# Chuẩn bị dữ liệu cho Word2Vec
tokenized_code = []
df = pd.read_csv("data/processed/functions.csv")
for code in df['processed_code']:
    tokens = word_tokenize(code)
    tokenized_code.append(tokens)
    # Huấn luyện mô hình Word2Vec
w2v_model = Word2Vec(
    sentences=tokenized_code,
    vector_size=100, # Kích thước vector
    window=5, # Kích thước cửa sổ ngữ cảnh
    min_count=2, # Tối thiểu số lần xuất hiện của từ
    workers=4 # Số luồng
)
# Lưu mô hình
w2v_model.save("models/w2v_code.model")
# Tạo embedding cho mỗi hàm bằng cách lấy trung bình các vector từ
def create_document_vector(doc_tokens, model):
    doc_vector = []
    for token in doc_tokens:
        if token in model.wv:
            doc_vector.append(model.wv[token])
            if not doc_vector:
                return np.zeros(model.vector_size)
    return np.mean(doc_vector, axis=0)
# Tạo embedding cho mỗi hàm
doc_vectors = []
for tokens in tokenized_code:
    doc_vectors.append(create_document_vector(tokens, w2v_model))

# Lưu các vector
doc_vectors_array = np.array(doc_vectors)
np.save("data/processed/w2v_vectors.npy", doc_vectors_array)
print(f"Shape of Word2Vec embeddings: {doc_vectors_array.shape}")

# Bài 4.1
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# Đọc dữ liệu vector
tfidf_vectors = np.load("data/processed/tfidf_matrix.npy")
w2v_vectors = np.load("data/processed/w2v_vectors.npy")
# Giả định: Gán nhãn cho mỗi hàm (ví dụ: phân loại theo chức năng)
# Trong thực tế, bạn cần có dữ liệu đã được gán nhãn
# Ở đây, chúng ta tạo nhãn giả cho mục đích demo
df = pd.read_csv("data/processed/functions.csv")
# Ví dụ: Phân loại hàm theo tên
# 0: hàm bắt đầu bằng "get_" hoặc "fetch_"
# 1: hàm bắt đầu bằng "create_" hoặc "build_"
# 2: các hàm còn lại
def assign_label(func_name):
    if func_name.startswith(('get_', 'fetch_')):
        return 0
    elif func_name.startswith(('create_', 'build_')):
        return 1
    else:
        return 2
df['label'] = df['name'].apply(assign_label)
# Chia dữ liệu
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
tfidf_vectors, df['label'], test_size=0.3, random_state=42
)
X_train_w2v, X_test_w2v, _, _ = train_test_split(
w2v_vectors, df['label'], test_size=0.3, random_state=42
)

# Bài 4.2
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
# Mô hình SVM
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_tfidf, y_train)
# Dự đoán
y_pred = svm_model.predict(X_test_tfidf)
# Đánh giá
print("SVM với TF-IDF:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
# Lưu mô hình
import pickle
with open("models/svm_tfidf.pkl", "wb") as f:
    pickle.dump(svm_model, f)

# Bài 4.3
from sklearn.ensemble import RandomForestClassifier
# Mô hình Random Forest
rf_model = RandomForestClassifier(
n_estimators=100,
max_depth=10,
random_state=42
)
rf_model.fit(X_train_w2v, y_train)
# Dự đoán
y_pred = rf_model.predict(X_test_w2v)
# Đánh giá
print("\nRandom Forest với Word2Vec:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
# Lưu mô hình
with open("models/rf_w2v.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# Bài 4.4
# Huấn luyện Random Forest với TF-IDF để so sánh
rf_tfidf = RandomForestClassifier(
n_estimators=100,

max_depth=10,
random_state=42
)
rf_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = rf_tfidf.predict(X_test_tfidf)
# Đánh giá
print("\nRandom Forest với TF-IDF:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tfidf):.4f}")
print(classification_report(y_test, y_pred_tfidf))
# Kết luận
print("\nSo sánh biểu diễn vector:")
print(f"SVM + TF-IDF: {accuracy_score(y_test,
svm_model.predict(X_test_tfidf)):.4f}")
print(f"RF + Word2Vec: {accuracy_score(y_test,
rf_model.predict(X_test_w2v)):.4f}")
print(f"RF + TF-IDF: {accuracy_score(y_test, y_pred_tfidf):.4f}")