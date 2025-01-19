![Image](https://github.com/user-attachments/assets/83ce3945-9102-4c78-b87d-4ab35aec29e1)
# Multitool Chat Assistant 🌟

The **Multitool Chat Assistant** is an all-in-one Streamlit-based application that leverages modern APIs and tools to provide various functionalities, such as PDF summarization, weather updates, query processing, image generation, and more. The application is designed for ease of use and versatility, catering to users with diverse needs.

---

## 🚀 Features

### 1. **Query Processing**
- Submit queries for processing.
- Integrated support for:
  - Weather inquiries (current and specific locations).
  - DuckDuckGo search results.
  - Image searches.
  - General queries answered using AI models.

### 2. **Weather Information**
- Fetches real-time weather details.
- Supports both current location and user-specified locations.

### 3. **PDF Summarization**
- Upload a PDF file to:
  - View it within the app.
  - Extract and summarize its content.
  - Submit queries for context-specific answers.

### 4. **Image Search**
- Generate AI-based images based on user-defined prompts.
- Download generated images directly.

### 5. **Picture Explanation**
- Upload an image to:
  - Summarize its content.
  - Answer specific user-defined queries about the image.

### 6. **History**
- Maintains a history of user interactions across all features.

---
![Image](https://github.com/user-attachments/assets/bcf2e8f1-baed-4f32-90a9-a1a7dbc71343)
## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher.
- Required libraries (install via `requirements.txt`).

### Steps
1. Clone the repository:
   ```bash
   https://github.com/Swagatade/custom-ai.git
---
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
---
3. Run the Streamlit application:
    ```bash
    streamlit run app.py
---
🔑 Configuration

API Keys

Obtain the following keys:

1. OpenRouter API Key: [Get it here](https://openrouter.ai/settings/keys).


2. Weather API Key: [Get it here](https://www.tomorrow.io/weather-api/).


Enter the API keys in the sidebar of the application.

---

📂 Directory Structure

multitool-chat-assistant/
│
├── app.py                # Main Streamlit application.
├── requirements.txt      # List of dependencies.
├── temp_data/            # Temporary storage for uploaded and processed files.
├── README.md             # Project documentation.
└── .gitignore            # Files and folders to ignore in the repository.


---

📦 Dependencies

Key dependencies include:

Streamlit: For creating the web application interface.

PyPDF2: For handling and extracting text from PDFs.

Pillow (PIL): For image handling and processing.

Requests: For making API requests.

DuckDuckGo Search: For web search capabilities.


Install all dependencies using the provided requirements.txt file.


---

✨ Usage

1. Launch the app using:
    ```bash
   streamlit run app.py

2. Use the sidebar to navigate through features:

Upload PDFs for summarization.

Generate images or summarize uploaded ones.

Fetch weather details for any location.

Process and answer custom queries.





---

🛡️ Security Notes

Keep your API keys confidential.

Avoid hardcoding sensitive data into the codebase.



---

📜 License

This project is licensed under the [MIT License](https://github.com/Swagatade/custom-ai/blob/4a84a423f45a764c7bf2bd36a0c27a4dc017b866/LICENSE).


---

Happy coding! 🎉
