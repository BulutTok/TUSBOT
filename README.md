# Medical Chatbot for Generating Multiple-Choice Test Questions

## Overview

This repository contains a Flask-based web application that leverages the OpenAI API to generate medical multiple-choice test questions. The application processes a collection of local JSON files, splits their content into manageable text chunks, generates embeddings for these chunks, and then uses cosine similarity to retrieve the most relevant pieces of information based on a user query. The final response is produced by the GPT-4o model, which formulates 5-option (A, B, C, D, E) multiple-choice questions in Turkish.

## Features

- **JSON Data Loading:** Automatically loads all JSON files from a specified local folder.
- **Text Chunking:** Uses the RecursiveCharacterTextSplitter from LangChain to split large JSON content into overlapping text chunks.
- **Embeddings Generation:** Generates text embeddings with OpenAI’s API (model: `text-embedding-ada-002`), complete with a retry mechanism to handle rate limits.
- **Relevance Ranking:** Applies cosine similarity to determine which text chunks best match the user query.
- **Chatbot Response:** Crafts system prompts that instruct GPT-4o to create medical test questions with 5 answer choices.
- **Flask Web Interface:** Provides a simple web interface to interact with the chatbot.

## Prerequisites

- **Python 3.7 or later**
- **Flask** for the web application
- **OpenAI API key** (set in the code or via environment variables)
- Required Python packages:
  - `os`
  - `json`
  - `flask`
  - `openai`
  - `langchain`
  - `scikit-learn`
  - `tenacity`

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/medical-chatbot.git
   cd medical-chatbot
   ```

2. **Set Up a Virtual Environment and Activate It:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```

3. **Install the Required Packages:**
   ```bash
   pip install flask openai langchain scikit-learn tenacity
   ```

4. **Configure the OpenAI API Key:**
   - In the code, replace `"API-KEY"` with your actual API key:
     ```python
     openai.api_key = "YOUR_ACTUAL_API_KEY"
     ```
   - Alternatively, set it as an environment variable and update the code to retrieve it from there.

## Configuration

- **JSON Files Folder:**
  - The application expects JSON files to be located in a specified folder:
    ```python
    folder_path = "/Users/buluttok/My Drive/Json"  # Update this path if necessary
    ```
- **Models Used:**
  - **Embeddings:** `text-embedding-ada-002`
  - **Chat Completion:** `gpt-4o`
  
  These models can be adjusted in the code if needed.

## How to Run

1. **Ensure Your JSON Files Are in the Designated Folder.**

2. **Run the Flask Application:**
   ```bash
   python app.py
   ```
   
3. **Access the Web Interface:**
   - Open your web browser and navigate to `http://127.0.0.1:5000/`.

4. **Interacting with the Chatbot:**
   - The home page provides an interface to submit a query.
   - Alternatively, you can send a POST request to the `/chat` endpoint with a JSON payload containing a `"message"` key.

## Code Overview

- **`load_json_files_from_folder(folder_path)`**  
  Loads all JSON files from the specified folder into a dictionary.

- **`split_text_into_chunks(text, chunk_size=500, chunk_overlap=100)`**  
  Splits the text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.

- **`generate_embeddings_with_retry(texts, model, batch_size)`**  
  Generates embeddings for text chunks with a retry mechanism (using Tenacity) to handle potential rate limit issues.

- **`generate_embeddings_for_chunks(data, model)`**  
  Processes each JSON file by splitting its content and generating embeddings for its text chunks.

- **`find_relevant_chunks(user_query, chunk_embeddings, chunk_metadata, model)`**  
  Computes cosine similarity between the user query embedding and each chunk’s embedding to identify the top relevant chunks.

- **`generate_chatbot_response(user_query, relevant_chunks)`**  
  Constructs a detailed prompt with system instructions (in Turkish) and the retrieved relevant data. It then calls the GPT-4o model to generate 5-option multiple-choice test questions.

- **Flask Routes:**
  - `/` : Renders the home page.
  - `/chat` : Accepts a user query and returns the generated chatbot response.

## Customization

- **System Prompts:**  
  The system messages in `generate_chatbot_response` dictate the chatbot's behavior. Modify these prompts to change the style or focus of the generated questions.
- **Response Format:**  
  The chatbot is configured to produce 5-option multiple-choice questions (A, B, C, D, E). Adjust the instructions if a different format is needed.

## Contact

For questions, feedback, or contributions, please reach out to:

**Bulut**  
Email: [buluttok2013@gmail.com](mailto:buluttok2013@gmail.com)

## Disclaimer

This chatbot is intended for educational and informational purposes only and should not be used as a substitute for professional medical advice. Always consult a qualified healthcare provider for medical concerns.

