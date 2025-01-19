import os
import json
import openai
from flask import Flask, render_template, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_fixed

# Flask app initialization
app = Flask(__name__)

# Set up OpenAI API key
openai.api_key = "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"  # Replace with your actual API key

# Load all JSON files from the local folder
def load_json_files_from_folder(folder_path):
    """Load all JSON files from a local folder."""
    json_data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as file:
                json_data[file_name] = json.load(file)
    return json_data

# Split text into chunks using RecursiveCharacterTextSplitter
def split_text_into_chunks(text, chunk_size=500, chunk_overlap=100):
    """Split text into manageable chunks with overlap using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# Retry mechanism for embedding requests
@retry(stop=stop_after_attempt(3), wait=wait_fixed(30))
def generate_embeddings_with_retry(texts, model="text-embedding-ada-002", batch_size=50):
    """Generate embeddings in batches with retry mechanism for rate limits."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = openai.Embedding.create(input=batch, model=model)
        embeddings.extend([item["embedding"] for item in response["data"]])
    return embeddings

# Generate embeddings for the chunks
def generate_embeddings_for_chunks(data, model="text-embedding-ada-002"):
    """Generate embeddings for the chunks of JSON content."""
    chunk_embeddings = {}
    chunk_metadata = {}

    for file_name, content in data.items():
        text = json.dumps(content)
        chunks = split_text_into_chunks(text)
        embeddings = generate_embeddings_with_retry(chunks, model=model)
        chunk_embeddings[file_name] = embeddings
        chunk_metadata[file_name] = chunks

    return chunk_embeddings, chunk_metadata

# Find the most relevant chunks using cosine similarity
def find_relevant_chunks(user_query, chunk_embeddings, chunk_metadata, model="text-embedding-ada-002"):
    """Find the most relevant chunks based on the user's query."""
    response = openai.Embedding.create(input=user_query, model=model)
    query_embedding = response["data"][0]["embedding"]

    similarities = []
    for file_name, embeddings in chunk_embeddings.items():
        for i, embedding in enumerate(embeddings):
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            similarities.append((similarity, file_name, i))

    sorted_similarities = sorted(similarities, key=lambda x: x[0], reverse=True)
    top_chunks = sorted_similarities[:2]

    relevant_chunks = []
    for _, file_name, chunk_index in top_chunks:
        chunk_text = chunk_metadata[file_name][chunk_index]
        relevant_chunks.append(chunk_text)

    return relevant_chunks

# Generate chatbot response
def generate_chatbot_response(user_query, relevant_chunks):
    """Generate a chatbot response using GPT-4o."""
    combined_relevant_data = "\n".join(relevant_chunks)

    messages = [
        {"role": "system", "content": "Bu sohbet robotu, sağlanan kaynaklardan tıbbi sorular oluşturacaktır."},
        {"role": "system", "content": "Kullanıcı her soru sorduğunda, 5 şıklı (A, B, C, D,E) çoktan seçmeli test soruları oluştur."},
        {"role": "system", "content": "Dili Türkçe tut ve soruları anlaşılır, net cümlelerle yaz."},
        {"role": "system", "content": "Sorular tıbbi bilgilere dayanmalı; ancak tam teşhis yerine eğitici içerik hedefle."},
        {"role": "system", "content": "Her şık, konuyla alakalı ancak yalnızca biri doğru olacak şekilde hazırlanmalı."},
        {"role": "system", "content": "Yanıt vermeden önce, kullanıcının bağlamını (Context) göz önünde bulundur."},
        {"role": "system", "content": "Cevap içinde gereksiz veya alakasız detay vermekten kaçın."},
        {"role": "system", "content": "Tıbbi kaynaklara dayalı kesin bilgiler paylaşmaya özen göster."},
        {"role": "system", "content": "Sorular aşırı kısa olmamalıdır. Dengeli ve anlaşılır olmalıdır."},
        {"role": "system", "content": "Soruların içerikleri mutlaka verilen PDF kaynaklarından alınmalıdır."},
        {"role": "system", "content": "Kullanıcı sorusu direkt teşhis veya tedavi talep ediyorsa, mutlaka sağlık uzmanına danışmasını hatırlat."},
        {"role": "system", "content": "Özellikle mutlaka seçmeli soru biçiminde cevap ver ve A, B, C, D, E seçenekleri sun."},
        {"role": "system", "content": "Sen, tıbbi içerik üreten bir sohbet robotusun. Kullanıcı soruları için her zaman 5 seçenekli test soruları oluştur."},
        {"role": "system", "content": "Kullanıcı tıbbi bağlam içeren sorular sorduğunda, vaka bazlı sorular oluşturabilirsin."},
        {"role": "system", "content": "Vaka sorularını 5 şıklı (A, B, C, D, E) çoktan seçmeli formatında hazırla ve bir hastanın semptomlarını veya durumunu açıklayarak başla."},
        {"role": "system", "content": "Vaka soruları tıbbi bir durum, teşhis veya tedaviyle ilgili olmalı, ancak eğitici ve rehberlik edici nitelikte olmalıdır."},
        {"role": "system", "content": "Vaka bazlı soruların metni anlaşılır, sistematik ve gerçek dünya durumlarına uygun olmalıdır."},
        {"role": "user", "content": f"Relevant Data:\n{combined_relevant_data}\n\nUser Query:\n{user_query}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=8000,
        temperature=0.5
    )
    return response["choices"][0]["message"]["content"].strip()

# Initialize JSON data and embeddings
folder_path = "/Users/buluttok/My Drive/Json"  # Replace with your folder path
json_data = load_json_files_from_folder(folder_path)
chunk_embeddings, chunk_metadata = generate_embeddings_for_chunks(json_data)

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    try:
        relevant_chunks = find_relevant_chunks(user_input, chunk_embeddings, chunk_metadata)
        response = generate_chatbot_response(user_input, relevant_chunks)
        return jsonify({"reply": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
