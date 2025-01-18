![Image](https://github.com/user-attachments/assets/83ce3945-9102-4c78-b87d-4ab35aec29e1)
# custom-ai
Here is a comprehensive guide to uploading your Streamlit application, complete with documentation and steps to integrate it into GitHub for version control and collaboration:
________________________________________
Streamlit Application Documentation
Features
1.	Query Processing
o	Accepts natural language queries.
o	Detects and processes weather queries, DuckDuckGo searches, and general queries using OpenRouter APIs.
2.	Weather Information
o	Fetches real-time weather details for the current or specified location using the Tomorrow.io Weather API.
3.	PDF Summarization
o	Uploads a PDF file for processing.
o	Summarizes content and answers queries related to the document using OpenRouter APIs.
4.	Image Search
o	Generates images based on user-defined prompts.
o	Offers an option to download generated images using Flux.
5.	Picture Explanation
o	Uploads an image.
o	Provides a detailed explanation of the image content based on user queries.
6.	History
o	Tracks user activity within the app for reference.
________________________________________
Technologies Used
•	Python Libraries: 
o	Streamlit for UI and interactivity.
o	PyPDF2 for PDF processing.
o	Pillow (PIL) for image handling.
o	requests for API communication.
o	duckduckgo_search for search engine results.
•	APIs: 
o	OpenRouter AI for language model queries.
o	Tomorrow.io for weather data.
________________________________________
Setup Instructions
Local Installation
1.	Clone the repository:
2.	git clone <repository_url>
3.	cd <repository_directory>
4.	Install dependencies:
5.	pip install -r requirements.txt
6.	Set environment variables:
o	OPENROUTER_API_KEY: Your OpenRouter API key.
o	WEATHER_API_KEY: Your Tomorrow.io API key.
Example:
export OPENROUTER_API_KEY=<your_openrouter_api_key>
export WEATHER_API_KEY=<your_weather_api_key>
7.	Run the Streamlit app:
8.	streamlit run app.py
Deploying on Streamlit Cloud
1.	Push the code to a GitHub repository.
2.	Sign in to Streamlit Cloud.
3.	Link your GitHub repository and deploy the app.py file.
4.	Add the required secrets (OPENROUTER_API_KEY, WEATHER_API_KEY) under the app's Settings > Secrets.
5.	Demo: https://custom-ai-123.streamlit.app/
________________________________________
GitHub Repository Structure
📂 Repository_Name/
├── 📂 temp_data/                  # Temporary folder for storing files
├── 📄 app.py                      # Main Streamlit app
├── 📄 requirements.txt            # Python dependencies
└── 📄 README.md                   # Documentation
________________________________________
Adding GitHub Documentation
Create a README.md
# Multitool Chat Assistant

A Streamlit application integrating LLMs, APIs, and other tools for text and image summarization, weather updates, and more.

## Features
- Query Processing
- Real-Time Weather Data
- PDF Summarization
- Image Generation
- History Tracking

## Installation
1. Clone the repository:
git clone <repository_url> cd <repository_directory>
2. Install dependencies:
pip install -r requirements.txt

3. Run the application:
streamlit run app.py

## Deployment
Use Streamlit Cloud or other hosting platforms. Add API keys as environment variables.

## License
[MIT License](LICENSE)
Create requirements.txt
Include:
streamlit
requests
PyPDF2
duckduckgo-search
________________________________________
Best Practices
1.	Use .gitignore to avoid committing sensitive data (e.g., API keys).
2.	Regularly update documentation with changes in features or APIs.
3.	Test the app locally before deployment.
Would you like help with a deployment script or specific GitHub workflows?

