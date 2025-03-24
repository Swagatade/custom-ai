import streamlit as st
import requests
import base64
import os
from duckduckgo_search import DDGS
import webbrowser
import PyPDF2
from PIL import Image
import io
import sqlite3
from datetime import datetime
import googleapiclient.discovery
import arxiv
import sys
import time
from fake_useragent import UserAgent
from langdetect import detect  # new import for language detection
import re  # ensure regex is imported

# API Configuration
API_BASE_URL = "https://openrouter.ai/api/v1"
st.sidebar.header("API Configuration")
API_KEY = st.sidebar.text_input("Enter your API Key using this link: https://openrouter.ai/settings/keys")

# Weather API Configuration
st.sidebar.header("Weather API Configuration")
WEATHER_API_KEY = st.sidebar.text_input("Enter your Weather API Key using this link: https://www.tomorrow.io/weather-api/")

# Function to extract text from a PDF
@st.cache_data
def extract_pdf_text(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {e}"

# Function to display PDF
@st.cache_data
def display_pdf(file):
    try:
        with open(file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")

# Function to preprocess PDF
def file_preprocessing(file):
    return extract_pdf_text(file)

# Model options for summarization and query processing
model_options = [
     ## meta-llama models
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
    "meta-llama/llama-3.2-90b-vision-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "meta-llama/llama-3.2-1b-instruct:free",
    "meta-llama/llama-3.1-70b-instruct:free",
    "meta-llama/llama-3.1-405b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "nvidia/llama-3.1-nemotron-70b-instruct:free",
    "meta-llama/llama-3-8b-instruct:free",
    "nousresearch/deephermes-3-llama-3-8b-preview:free",
     ## gemma models
    "google/gemma-3-27b-it:free",
    "google/gemma-3-12b-it:free",
    "google/gemma-3-4b-it:free",
    "google/gemma-3-1b-it:free",
    "google/gemma-2-9b-it:free",
    "google/gemini-2.0-flash-thinking-exp-1219:free",
    "google/gemini-2.0-flash-exp:free",
    "google/gemini-2.0-flash-thinking-exp:free",
    "google/gemini-2.0-flash-lite-preview-02-05:free",
    "google/learnlm-1.5-pro-experimental:free",
    "google/learnlm-1.5-pro-experimental:free",
    "google/gemini-exp-1206:free",
     ## deepseek models
    "deepseek/deepseek-r1-zero:free",
    "deepseek/deepseek-r1:free",
    "deepseek/deepseek-r1-distill-llama-70b:free",
    "deepseek/deepseek-chat:free",
    "deepseek/deepseek-r1-distill-qwen-32b:free",
    "deepseek/deepseek-r1-distill-qwen-14b:free",
    "deepseek/deepseek-chat-v3-0324:free",
     ## mistral models
    "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
    "mistralai/mistral-small-24b-instruct-2501:free",
    "cognitivecomputations/dolphin3.0-mistral-24b:free"
    "mistralai/mistral-nemo:free",
    "mistralai/mistral-7b-instruct:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
     ## phi models
    "microsoft/phi-3-mini-128k-instruct:free",
    "microsoft/phi-3-medium-128k-instruct:free",
     ## qwen models
    "qwen/qwq-32b:free",
    "qwen/qwq-32b-preview:free",
    "qwen/qwen2.5-vl-72b-instruct:free",
    "qwen/qwen-2-7b-instruct:free",
    "qwen/qwen-2.5-coder-32b-instruct:free",
    "qwen/qwen-vl-plus:free",
     ## olympiccoder
    "open-r1/olympiccoder-7b:free",
    "open-r1/olympiccoder-32b:free",
     ## other models
    "moonshotai/moonlight-16b-a3b-instruct:free",
    "rekaai/reka-flash-3:free",
    "huggingfaceh4/zephyr-7b-beta:free",
    "openchat/openchat-7b:free",
    "gryphe/mythomax-l2-13b:free",
    "undi95/toppy-m-7b:free"
]
selected_model = st.sidebar.selectbox("Choose a Model", model_options)

# Function to summarize a document and answer queries
def llm_pipeline(filepath, query):
    input_text = file_preprocessing(filepath)

    payload = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": "You are an assistant summarizing a document."},
            {"role": "user", "content": f"Summarize the following text:\n{input_text}\nBased on the document, answer the question: {query}"}
        ],
        "max_tokens":10000
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
    }
    response = requests.post(f"{API_BASE_URL}/chat/completions", json=payload, headers=headers)
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        return f"Error during summarization: {response.json().get('message', 'Unknown error')}"

# Function to fetch current location weather
def fetch_current_location_weather():
    try:
        geo_url = "https://ipinfo.io/json"
        geo_response = requests.get(geo_url)
        geo_data = geo_response.json()
        latitude, longitude = geo_data['loc'].split(',')

        weather_url = f"https://api.tomorrow.io/v4/weather/realtime?location={latitude},{longitude}&apikey={WEATHER_API_KEY}"
        headers = {"accept": "application/json"}
        weather_response = requests.get(weather_url, headers=headers)

        if weather_response.status_code == 200:
            weather_data = weather_response.json()["data"]["values"]
            return (
                f"**Current Location Weather:**\n"
                f"- Temperature: {weather_data['temperature']}\u00b0C\n"
                f"- Apparent Temperature: {weather_data['temperatureApparent']}\u00b0C\n"
                f"- Humidity: {weather_data['humidity']}%\n"
                f"- Wind Speed: {weather_data['windSpeed']} m/s\n"
                f"- Cloud Cover: {weather_data['cloudCover']}%\n"
                f"- Visibility: {weather_data['visibility']} km\n"
                f"- UV Index: {weather_data['uvIndex']}"
            )
        else:
            return f"Failed to fetch weather data. Status Code: {weather_response.status_code}, Message: {weather_response.text}"
    except Exception as e:
        return f"An error occurred: {e}"

# Function to fetch weather for a specified location
def fetch_specified_location_weather(location):
    try:
        if not location.strip():
            return "Location cannot be empty."
        url = f"https://api.tomorrow.io/v4/weather/realtime?location={location}&apikey={WEATHER_API_KEY}"
        headers = {"accept": "application/json"}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            weather_data = response.json()["data"]
            return (
                f"**Weather in {location}:**\n"
                f"- Time: {weather_data['time']}\n"
                f"- Temperature: {weather_data['values']['temperature']}\u00b0C\n"
                f"- Apparent Temperature: {weather_data['values']['temperatureApparent']}\u00b0C\n"
                f"- Humidity: {weather_data['values']['humidity']}%\n"
                f"- Wind Speed: {weather_data['values']['windSpeed']} m/s\n"
                f"- Cloud Cover: {weather_data['values']['cloudCover']}%\n"
                f"- Visibility: {weather_data['values']['visibility']} km\n"
                f"- UV Index: {weather_data['values']['uvIndex']}"
            )
        else:
            return f"Failed to fetch weather data for {location}. Status Code: {response.status_code}, Message: {response.text}"
    except Exception as e:
        return f"An error occurred: {e}"

# Function to perform DuckDuckGo search
def search_duckduckgo(query, max_results=10):
    try:
        results = []
        # Initialize DDGS without the 'proxies' parameter.
        with DDGS() as ddgs:
            for idx, result in enumerate(ddgs.text(query, max_results=max_results, region="in-en"), start=1):
                domain = result['href'].split('/')[2].replace('www.', '')
                # Use fallback logo for the specific domain
                if domain == "blog.google":
                    logo_url = "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_92x30dp.png"
                else:
                    logo_url = f"https://logo.clearbit.com/{domain}"
                company_name = 'Wikipedia' if 'wikipedia.org' in domain else domain.split('.')[0]
                result_entry = (
                    f"<div style='margin-bottom: 20px;'>"
                    f"<img src='{logo_url}' alt='Company Logo' style='width: 50px; height: 50px; vertical-align: middle; margin-right: 10px;'>"
                    f"<span style='font-size: 20px; font-weight: bold;'>{company_name}</span><br>"
                    f"<a href='{result['href']}' target='_blank'>{result['href']}</a>"
                    f"<p>{result['body']}</p>"
                    f"</div>"
                )
                results.append(result_entry)
        return results
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 429:
            return [f"Rate limit exceeded. Please try again later."]
        else:
            return [f"HTTP error occurred: {http_err}"]
    except Exception as e:
        return [f"An error occurred: {e}"]

# Function to download and display image
def download_image(prompt, width=768, height=768, model='flux', seed=None):
    url = f"https://image.pollinations.ai/prompt/{prompt}?width={width}&height={height}&model={model}&seed={seed}"
    response = requests.get(url)
    image_path = 'generated_image.jpg'
    with open(image_path, 'wb') as file:
        file.write(response.content)
    return image_path

# Function to summarize an image
def summarize_image(image_path, query):
    try:
        # Convert image to JPG and ensure size is under 10MB
        with Image.open(image_path) as img:
            with io.BytesIO() as output:
                img.save(output, format="JPEG", quality=95)
                output_size = output.tell()
                if output_size > 10 * 1024 * 1024:  # 10MB
                    return "Error: Image size exceeds 10MB after conversion."
                encoded_image = base64.b64encode(output.getvalue()).decode("utf-8")
        
        payload = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": "You are an assistant providing a detailed description of an image."},
                {"role": "user", "content": f"Describe the image in detail based on the following query: {query}\n{encoded_image}"}
            ],
            "max_tokens": 200
        }
        headers = {
            "Authorization": f"Bearer {API_KEY}",
        }
        response = requests.post(f"{API_BASE_URL}/chat/completions", json=payload, headers=headers)
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result:
                description = result['choices'][0]['message']['content']
                return description
            else:
                return "Error: 'choices' not found in the response."
        else:
            return f"Error during summarization: {response.json().get('message', 'Unknown error')}"
    except Exception as e:
        return f"An error occurred: {e}"

from PyPDF2 import PdfWriter, PdfReader

# Function to compress PDF without reducing quality
def compress_pdf(input_pdf_path, output_pdf_path):
    try:
        pdf_writer = PdfWriter()
        with open(input_pdf_path, "rb") as input_pdf:
            pdf_reader = PdfReader(input_pdf)
            for page in pdf_reader.pages:
                pdf_writer.add_page(page)
            with open(output_pdf_path, "wb") as output_pdf:
                pdf_writer.write(output_pdf)
        return output_pdf_path
    except Exception as e:
        return f"Error compressing PDF: {e}"

# Function to handle web search queries
def handle_web_search(query):
    results = search_duckduckgo(query)
    return "\n".join(results)

def search_youtube(query, max_results=10):  # updated default max_results
    try:
        youtube = googleapiclient.discovery.build(
            "youtube", "v3", 
            developerKey="AIzaSyCdLr1l8bbi_u6EiM4pwRzuIjk3ztx3xVk"
        )
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=max_results,
            order="relevance"  # new parameter for top results
        )
        response = request.execute()

        results = []
        for item in response.get('items', []):
            title = item['snippet']['title']
            if "reel" in title.lower():
                continue
            try:
                lang = detect(title)
            except Exception:
                lang = "en"
            if lang not in ['en', 'hi']:
                continue
            video_id = item['id']['videoId']
            result_entry = (
                f"<div style='margin-bottom: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px;'>"
                f"<iframe width='320' height='180' src='https://www.youtube.com/embed/{video_id}' frameborder='0' allowfullscreen></iframe><br>"
                f"<a href='https://www.youtube.com/watch?v={video_id}' target='_blank'>"
                f"<h3>{title}</h3></a>"
                f"</div>"
            )
            results.append(result_entry)
        return results
    except Exception as e:
        return [f"An error occurred: {str(e)}"]

def perform_combined_search(query):
    results = {
        'web': [],
        'ai': '',
        'youtube': []
    }
    
    # Web Search with 10 results
    web_results = search_duckduckgo(query, max_results=10)  # updated max_results
    results['web'] = web_results

    # AI Model Response remains unchanged
    payload = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ],
        "max_tokens": 1000
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    try:
        response = requests.post(f"{API_BASE_URL}/chat/completions", json=payload, headers=headers)
        if response.status_code == 200:
            result_data = response.json()
            if 'choices' in result_data:
                results['ai'] = result_data['choices'][0]['message']['content']
    except Exception as e:
        results['ai'] = f"AI Error: {str(e)}"

    # YouTube Search with 10 results
    youtube_results = search_youtube(query, max_results=10)  # updated max_results
    results['youtube'] = youtube_results

    return results

# Initialize database
def init_db():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feature TEXT,
            input TEXT,
            output TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def upgrade_db():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("PRAGMA table_info(history)")
    columns = [col[1] for col in c.fetchall()]
    if 'timestamp' not in columns:
        # Add column without DEFAULT clause.
        c.execute("ALTER TABLE history ADD COLUMN timestamp DATETIME")
        # Set current timestamp for existing rows.
        c.execute("UPDATE history SET timestamp = CURRENT_TIMESTAMP WHERE timestamp IS NULL")
    conn.commit()
    conn.close()

def add_history(feature, input_data, output_data):
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("INSERT INTO history (feature, input, output) VALUES (?, ?, ?)",
              (feature, str(input_data), str(output_data)))
    conn.commit()
    conn.close()

def get_all_history():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("SELECT id, feature, input, output, timestamp FROM history ORDER BY id")
    rows = c.fetchall()
    conn.close()
    return rows

def clear_all_history():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("DELETE FROM history")
    conn.commit()
    conn.close()

def delete_history_entry(entry_id):
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("DELETE FROM history WHERE id=?", (entry_id,))
    conn.commit()
    conn.close()

def search_research_papers(query, max_results=5):
    try:
        # ArXiv search using new Client method
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        arxiv_results = []
        for paper in client.results(search):
            # Convert Author objects to strings properly
            author_names = [str(author.name) if hasattr(author, 'name') else str(author) for author in paper.authors]
            
            result_entry = (
                f"<div style='margin-bottom: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px;'>"
                f"<h3>{paper.title}</h3>"
                f"<p><strong>Authors:</strong> {', '.join(author_names)}</p>"
                f"<p><strong>Published:</strong> {paper.published}</p>"
                f"<p>{paper.summary[:300]}...</p>"
                f"<a href='{paper.pdf_url}' target='_blank'>Download PDF</a> | "
                f"<a href='{paper.entry_id}' target='_blank'>View on ArXiv</a>"
                f"</div>"
            )
            arxiv_results.append(result_entry)

        # Add Medium search
        medium_results = []
        try:
            medium_url = f"https://api.medium.com/v1/search?q={requests.utils.quote(query)}"
            medium_response = requests.get(
                f"https://medium.com/search?q={requests.utils.quote(query)}",
                headers={'User-Agent': UserAgent().random}
            )
            
            if medium_response.status_code == 200:
                # Parse Medium articles from response
                for i in range(max_results):
                    try:
                        result_entry = (
                            f"<div style='margin-bottom: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px;'>"
                            f"<h3>Medium Article</h3>"
                            f"<a href='https://medium.com/search?q={requests.utils.quote(query)}' target='_blank'>"
                            f"View Medium Articles about {query}</a>"
                            f"</div>"
                        )
                        medium_results.append(result_entry)
                        break
                    except Exception as e:
                        st.warning(f"Warning: Could not process Medium result: {str(e)}")
                        continue
        except Exception as medium_error:
            st.info("Medium search results may be limited.")

        # Add WordPress search
        wordpress_results = []
        try:
            # Search WordPress.com public posts
            wp_url = f"https://public-api.wordpress.com/rest/v1.1/read/search?q={requests.utils.quote(query)}&number={max_results}"
            wp_response = requests.get(wp_url)
            
            if wp_response.status_code == 200:
                result_entry = (
                    f"<div style='margin-bottom: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px;'>"
                    f"<h3>WordPress Articles</h3>"
                    f"<a href='https://wordpress.com/read/search?q={requests.utils.quote(query)}' target='_blank'>"
                    f"View WordPress Articles about {query}</a>"
                    f"</div>"
                )
                wordpress_results.append(result_entry)
        except Exception as wp_error:
            st.info("WordPress search results may be limited.")

        return {
            'arxiv': arxiv_results,
            'medium': medium_results,
            'wordpress': wordpress_results
        }
    except Exception as e:
        return {'error': f"An error occurred: {str(e)}"}

# Initialize DB on startup
init_db()
upgrade_db()

# Main Streamlit App
st.title("Multitool Chat Assistant")

# Sidebar Menu
menu = ["Query Processing", "Weather Information", "PDF Summarization", "Image Search", "Picture Explanation", "Web Search", "1-Click Search", "Deep Research", "AI Only", "History"]
choice = st.sidebar.selectbox("Choose a Feature", menu)

if choice == "Query Processing":
    st.subheader("Query Processing")
    user_query = st.text_input("Enter your query:")

    if st.button("Submit Query"):
        if user_query.strip().lower() in ["hi", "hellow"]:
            headers = {"Authorization": f"Bearer {API_KEY}"}
            payload = {
                "model": selected_model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Please provide responses in a structured format with sections for Introduction, Main Points, and Conclusion."},
                    {"role": "user", "content": user_query}
                ],
                "max_tokens": 1500
            }
            response_model = requests.post(f"{API_BASE_URL}/chat/completions", json=payload, headers=headers)
            if response_model.status_code == 200 and 'choices' in response_model.json():
                result = response_model.json()['choices'][0]['message']['content']
            else:
                result = f"Error in model response: {response_model.text}"
        else:
            if 'weather' in user_query.lower():
                if 'in' in user_query.lower():
                    location = user_query.split('in')[-1].strip()
                    result = fetch_specified_location_weather(location)
                else:
                    result = fetch_current_location_weather()
            else:
                headers = {"Authorization": f"Bearer {API_KEY}"}
                system_prompt = """You are a helpful assistant. Please structure your responses in the following format:

                🎯 Main Answer:
                - Provide a direct, concise answer to the query
                
                📝 Detailed Explanation:
                - Break down the topic into key points
                - Include relevant examples or analogies
                
                💡 Additional Insights:
                - Provide context or related information
                - Mention any important considerations
                
                🔑 Key Takeaways:
                - List 2-3 main points to remember
                
                Please ensure all responses are clear, accurate, and well-organized."""

                payload_model = {
                    "model": selected_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    "max_tokens": 2000
                }
                response_model = requests.post(f"{API_BASE_URL}/chat/completions", json=payload_model, headers=headers)
                if response_model.status_code == 200 and 'choices' in response_model.json():
                    model_answer = response_model.json()['choices'][0]['message']['content']
                else:
                    model_answer = f"Error in model response: {response_model.text}"
                
                web_results = handle_web_search(user_query)
                analysis_prompt = f"""Analyze the following web search results and provide insights in this structure:

                📊 Data Analysis:
                - Key findings from web results
                - Common themes and patterns
                
                🔄 Integration:
                - Compare with the model's response
                - Identify any contradictions or confirmations
                
                Web Results: {web_results}"""

                payload_analysis = {
                    "model": selected_model,
                    "messages": [
                        {"role": "system", "content": "You are an assistant analyzing search results."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    "max_tokens": 1500
                }
                response_analysis = requests.post(f"{API_BASE_URL}/chat/completions", json=payload_analysis, headers=headers)
                if response_analysis.status_code == 200 and 'choices' in response_analysis.json():
                    web_analysis = response_analysis.json()['choices'][0]['message']['content']
                else:
                    web_analysis = f"Error during web analysis: {response_analysis.text}"
                
                combine_prompt = f"""Create a comprehensive response combining these insights:

                Model Analysis: {model_answer}
                Web Analysis: {web_analysis}

                Structure the final response with:
                
                📌 Executive Summary
                📊 Detailed Analysis
                💡 Key Insights
                ✅ Final Recommendations"""

                payload_combine = {
                    "model": selected_model,
                    "messages": [
                        {"role": "system", "content": "You are an assistant skilled at synthesizing information."},
                        {"role": "user", "content": combine_prompt}
                    ],
                    "max_tokens": 1500
                }
                response_combine = requests.post(f"{API_BASE_URL}/chat/completions", json=payload_combine, headers=headers)
                if response_combine.status_code == 200 and 'choices' in response_combine.json():
                    result = response_combine.json()['choices'][0]['message']['content']
                else:
                    result = f"Error during final combination: {response_combine.text}"

        result = re.sub(r'[#\$%\^]+', '', result)
        st.markdown(result)
        add_history("Query Processing", user_query, result)

elif choice == "Weather Information":
    st.subheader("Weather Information")
    location = st.text_input("Enter a location for weather information (leave blank for current location):")

    if st.button("Get Weather"):
        if location.strip():
            result = fetch_specified_location_weather(location)
        else:
            result = fetch_current_location_weather()

        st.write(result)
        add_history("Weather Information", location or "Current Location", result)

elif choice == "Image Search":
    st.subheader("Image Search")
    prompt = st.text_input("Enter a prompt for image generation:")
    width = 768
    height =768
    model = "flux"
    seed = st.number_input("Seed", value=42, step=1)

    if st.button("Generate Image"):
        image_path = download_image(prompt, width, height, model, seed)
        st.image(image_path)
        with open(image_path, "rb") as file:
            btn = st.download_button(
                label="Download Image",
                data=file,
                file_name="generated_image.jpg",
                mime="image/jpeg"
            )
        add_history("Image Search", prompt, image_path)

elif choice == "Picture Explanation":
    st.subheader("Picture Explanation")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        os.makedirs("temp_data", exist_ok=True)
        image_path = os.path.join("temp_data", uploaded_image.name)
        with open(image_path, "wb") as file:
            file.write(uploaded_image.read())
        query = st.text_input("Enter your query about the image:")
        if st.button("Generate Text"):
            try:
                summary = summarize_image(image_path, query)
            except Exception as e:
                summary = f"Error during image summarization: {e}"
            st.write(summary)
            add_history("Picture Explanation", uploaded_image.name, summary)

elif choice == "PDF Summarization":
    st.subheader("PDF Summarization")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        temp_dir = "temp_data"
        os.makedirs(temp_dir, exist_ok=True)
        filepath = os.path.join(temp_dir, uploaded_file.name)
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())
        if os.path.getsize(filepath) > 10 * 1024 * 1024:  # 10MB
            compressed_filepath = os.path.join(temp_dir, f"compressed_{uploaded_file.name}")
            compress_result = compress_pdf(filepath, compressed_filepath)
            if "Error" in compress_result:
                st.error(compress_result)
                filepath = None
            else:
                filepath = compressed_filepath
        if filepath:
            query = st.text_input("Enter your query about the document:")
            if st.button("Submit Query"):
                summary = llm_pipeline(filepath, query)
                st.write(summary)
                add_history("PDF Summarization", query, summary)

elif choice == "Web Search":
    st.subheader("Web Search")
    search_query = st.text_input("Enter your search query:")

    if st.button("Search"):
        result = handle_web_search(search_query)
        if "Rate limit exceeded" in result:
            st.error(result)
        else:
            st.markdown(result, unsafe_allow_html=True)
        add_history("Web Search", search_query, result)

elif choice == "1-Click Search":
    st.subheader("1-Click Search")
    search_query = st.text_input("Enter your search query:")

    if st.button("Search Everything"):
        with st.spinner("Searching across multiple platforms..."):
            results = perform_combined_search(search_query)
            
            # Display Web Results
            st.subheader("Web Results")
            st.markdown("\n".join(results['web']), unsafe_allow_html=True)
            
            # Display YouTube Results
            st.subheader("YouTube Results")
            st.markdown("\n".join(results['youtube']), unsafe_allow_html=True)
            
            # Combined analysis of web and YouTube results using the AI model
            web_text = " ".join(results['web'])
            youtube_text = " ".join(results['youtube'])
            analysis_prompt = f"Analyze the following web search and YouTube video results and provide a comprehensive, concise answer in English: Web: {web_text} YouTube: {youtube_text}"
            payload_analysis = {
                "model": selected_model,
                "messages": [
                    {"role": "system", "content": "You are an assistant analyzing search results."},
                    {"role": "user", "content": analysis_prompt}
                ],
                "max_tokens": 1000
            }
            headers = {
                "Authorization": f"Bearer {API_KEY}"
            }
            response_analysis = requests.post(f"{API_BASE_URL}/chat/completions", json=payload_analysis, headers=headers)
            if response_analysis.status_code == 200:
                analysis_result = response_analysis.json()['choices'][0]['message']['content']
            else:
                analysis_result = "Error in combined analysis."
            st.subheader("Analyzed Combined Result")
            st.write(analysis_result)
            
            # Add to history with analysis summary snippet
            add_history("1-Click Search", search_query, 
                f"Web: {len(results['web'])} items | YouTube: {len(results['youtube'])} items | Analysis: {analysis_result[:100]}...")

elif choice == "Deep Research":
    st.subheader("Deep Research")
    
    # Add research mode selection
    research_mode = st.radio("Select Research Mode", ["Quick Research", "Deep Research"])
    search_query = st.text_input("Enter your research query:")
    
    if st.button("Search and Analyze"):
        with st.spinner("Searching and analyzing..."):
            # Get research papers and web results
            paper_results = search_research_papers(search_query)
            web_results = handle_web_search(search_query)
            
            if research_mode == "Quick Research":
                # Quick Research Mode
                tabs = st.tabs(["📚 Research Summary", "🌐 Web Analysis", "🤖 AI Insights"])
                
                with tabs[0]:
                    st.subheader("Research Overview")
                    # Analyze papers quickly
                    quick_analysis_prompt = f"""Analyze these research papers and provide a quick summary:
                    Papers: {str(paper_results)}
                    
                    Provide:
                    1. Key Findings (3-5 bullet points)
                    2. Main Trends
                    3. Quick Takeaways"""
                    
                    headers = {"Authorization": f"Bearer {API_KEY}"}
                    payload = {
                        "model": selected_model,
                        "messages": [
                            {"role": "system", "content": "You are a research analyst providing quick insights."},
                            {"role": "user", "content": quick_analysis_prompt}
                        ],
                        "max_tokens": 1000
                    }
                    
                    response = requests.post(f"{API_BASE_URL}/chat/completions", json=payload, headers=headers)
                    if response.status_code == 200:
                        quick_summary = response.json()['choices'][0]['message']['content']
                        st.markdown(quick_summary)
                
                with tabs[1]:
                    st.subheader("Web Content Analysis")
                    # Analyze web results
                    web_analysis_prompt = f"""Analyze these web search results and extract key insights:
                    Content: {web_results}
                    
                    Provide:
                    1. Main Topics Discussed
                    2. Common Themes
                    3. Contradictions (if any)"""
                    
                    payload["messages"][1]["content"] = web_analysis_prompt
                    response = requests.post(f"{API_BASE_URL}/chat/completions", json=payload, headers=headers)
                    if response.status_code == 200:
                        web_analysis = response.json()['choices'][0]['message']['content']
                        st.markdown(web_analysis)
                
                with tabs[2]:
                    st.subheader("AI Model Insights")
                    # Get AI model's perspective
                    ai_analysis_prompt = f"""Analyze this topic as an AI model:
                    Topic: {search_query}
                    
                    Provide:
                    1. Current State of Research
                    2. Future Implications
                    3. Potential Research Gaps"""
                    
                    payload["messages"][1]["content"] = ai_analysis_prompt
                    response = requests.post(f"{API_BASE_URL}/chat/completions", json=payload, headers=headers)
                    if response.status_code == 200:
                        ai_insights = response.json()['choices'][0]['message']['content']
                        st.markdown(ai_insights)
                
                # Quick download option
                quick_report = f"""
                # Quick Research Report: {search_query}
                
                ## Research Summary
                {quick_summary}
                
                ## Web Analysis
                {web_analysis}
                
                ## AI Insights
                {ai_insights}
                """
                
                st.download_button(
                    label="📥 Download Quick Report",
                    data=quick_report,
                    file_name="quick_research_report.txt",
                    mime="text/plain"
                )
                
                add_history("Quick Research", search_query, 
                          f"Quick Research Report Generated | Papers analyzed: {len(paper_results.get('arxiv', []))}")
            
            else:
                # Original Deep Research Mode
                with st.spinner("Searching papers and articles..."):
                    results = search_research_papers(search_query)
                    
                    if 'error' in results:
                        st.error(results['error'])
                    else:
                        # Create tabs for different sources
                        tabs = st.tabs(["📚 ArXiv Papers", "📱 Medium Articles", "📝 WordPress Posts", "🔍 Research Summary"])
                        
                        # ArXiv Papers Tab
                        with tabs[0]:
                            st.markdown("""
                            <style>
                            .paper-container {
                                border: 1px solid #e0e0e0;
                                border-radius: 5px;
                                padding: 15px;
                                margin: 10px 0;
                                background-color: #f9f9f9;
                                color: #000000;
                            }
                            .paper-container h3 {
                                color: #000000 !important;
                            }
                            .paper-container p {
                                color: #000000 !important;
                            }
                            .paper-container strong {
                                color: #000000 !important;
                            }
                            .paper-container a {
                                color: #0066cc !important;
                            }
                            </style>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("<h2 style='color: #ffffff;'>📚 Academic Research Papers</h2>", unsafe_allow_html=True)
                            for paper in results['arxiv']:
                                st.markdown(f'<div class="paper-container">{paper}</div>', unsafe_allow_html=True)
                        
                        # Medium Articles Tab
                        with tabs[1]:
                            st.subheader("📱 Medium Articles")
                            for article in results['medium']:
                                st.markdown(article, unsafe_allow_html=True)
                        
                        # WordPress Tab
                        with tabs[2]:
                            st.subheader("📝 WordPress Content")
                            for post in results['wordpress']:
                                st.markdown(post, unsafe_allow_html=True)
                        
                        # Research Summary Tab
                        with tabs[3]:
                            st.subheader("🔍 Research Summary")
                            
                            # Define headers for API calls
                            headers = {"Authorization": f"Bearer {API_KEY}"}
                            research_summary = {}  # Dictionary to store all summaries
                            
                            # Step 1: Research Focus
                            with st.spinner("Analyzing Research Focus..."):
                                focus_prompt = f"""Analyze the research focus for: {search_query}
                                Provide:
                                🎯 Research Focus:
                                - Main topic and scope
                                - Key research questions"""
                                
                                payload = {
                                    "model": selected_model,
                                    "messages": [
                                        {"role": "system", "content": "You are a research analyst focusing on research scope and questions."},
                                        {"role": "user", "content": focus_prompt}
                                    ],
                                    "max_tokens": 1000
                                }
                                response = requests.post(f"{API_BASE_URL}/chat/completions", json=payload, headers=headers)
                                if response.status_code == 200:
                                    research_summary['focus'] = response.json()['choices'][0]['message']['content']
                                    with st.expander("🎯 Research Focus", expanded=True):
                                        st.markdown(research_summary['focus'])

                            # Step 2: Key Findings
                            with st.spinner("Analyzing Key Findings..."):
                                findings_prompt = f"""Analyze the key findings for: {search_query}
                                Provide:
                                📊 Key Findings:
                                - Major discoveries
                                - Statistical significance
                                - Research outcomes"""
                                
                                payload["messages"][1]["content"] = findings_prompt
                                response = requests.post(f"{API_BASE_URL}/chat/completions", json=payload, headers=headers)
                                if response.status_code == 200:
                                    research_summary['findings'] = response.json()['choices'][0]['message']['content']
                                    with st.expander("📊 Key Findings", expanded=True):
                                        st.markdown(research_summary['findings'])

                            # Step 3: Methodologies
                            with st.spinner("Analyzing Research Methodologies..."):
                                methods_prompt = f"""Analyze the research methodologies for: {search_query}
                                Provide:
                                🔬 Methodologies:
                                - Research approaches
                                - Data collection methods
                                - Analysis techniques"""
                                
                                payload["messages"][1]["content"] = methods_prompt
                                response = requests.post(f"{API_BASE_URL}/chat/completions", json=payload, headers=headers)
                                if response.status_code == 200:
                                    research_summary['methods'] = response.json()['choices'][0]['message']['content']
                                    with st.expander("🔬 Methodologies", expanded=True):
                                        st.markdown(research_summary['methods'])

                            # Step 4: Insights
                            with st.spinner("Generating Insights..."):
                                insights_prompt = f"""Generate insights for: {search_query}
                                Provide:
                                💡 Insights:
                                - Main implications
                                - Practical applications
                                - Future research directions"""
                                
                                payload["messages"][1]["content"] = insights_prompt
                                response = requests.post(f"{API_BASE_URL}/chat/completions", json=payload, headers=headers)
                                if response.status_code == 200:
                                    research_summary['insights'] = response.json()['choices'][0]['message']['content']
                                    with st.expander("💡 Insights", expanded=True):
                                        st.markdown(research_summary['insights'])

                            # Step 5: Critical Analysis
                            with st.spinner("Performing Critical Analysis..."):
                                critical_prompt = f"""Provide critical analysis for: {search_query}
                                Provide:
                                📌 Critical Analysis:
                                - Strengths and limitations
                                - Gaps in research
                                - Contradicting findings"""
                                
                                payload["messages"][1]["content"] = critical_prompt
                                response = requests.post(f"{API_BASE_URL}/chat/completions", json=payload, headers=headers)
                                if response.status_code == 200:
                                    research_summary['critical'] = response.json()['choices'][0]['message']['content']
                                    with st.expander("📌 Critical Analysis", expanded=True):
                                        st.markdown(research_summary['critical'])

                            # Step 6: Conclusions
                            with st.spinner("Generating Conclusions..."):
                                conclusions_prompt = f"""Provide conclusions for: {search_query}
                                Provide:
                                ✅ Conclusions:
                                - Summary of findings
                                - Recommendations
                                - Future perspectives"""
                                
                                payload["messages"][1]["content"] = conclusions_prompt
                                response = requests.post(f"{API_BASE_URL}/chat/completions", json=payload, headers=headers)
                                if response.status_code == 200:
                                    research_summary['conclusions'] = response.json()['choices'][0]['message']['content']
                                    with st.expander("✅ Conclusions", expanded=True):
                                        st.markdown(research_summary['conclusions'])

                            # Combine all summaries for the full report
                            full_research_summary = f"""
                            # Research Summary: {search_query}
                            
                            {research_summary.get('focus', 'Focus analysis not available')}
                            
                            {research_summary.get('findings', 'Findings analysis not available')}
                            
                            {research_summary.get('methods', 'Methods analysis not available')}
                            
                            {research_summary.get('insights', 'Insights analysis not available')}
                            
                            {research_summary.get('critical', 'Critical analysis not available')}
                            
                            {research_summary.get('conclusions', 'Conclusions not available')}
                            """

                        # Enhanced Web Analysis with error handling
                        try:
                            web_analysis_section = f"""
                            ## 🌐 Web Analysis
                            
                            ### Main Sources
                            {web_results[:500] if isinstance(web_results, str) else ''.join(web_results[:5])}
                            
                            ### Key Topics
                            {research_summary.get('web_content', 'Web content analysis in progress...')}
                            """
                        except Exception as e:
                            web_analysis_section = "## 🌐 Web Analysis\n\nError processing web analysis."
                            st.error(f"Web analysis error: {str(e)}")

                        # Enhanced Additional Resources with error handling
                        try:
                            # Safely encode search query
                            encoded_query = requests.utils.quote(search_query)
                            
                            # Create database links with error checking
                            database_links = {
                                "Google Scholar": f"https://scholar.google.com/scholar?q={encoded_query}",
                                "Semantic Scholar": f"https://www.semanticscholar.org/search?q={encoded_query}",
                                "ResearchGate": f"https://www.researchgate.net/search/publication?q={encoded_query}",
                                "Science Direct": f"https://www.sciencedirect.com/search?qs={encoded_query}",
                                "PubMed Central": f"https://www.ncbi.nlm.nih.gov/pmc/?term={encoded_query}",
                                "Springer Link": f"https://link.springer.com/search?query={encoded_query}"
                            }

                            # Format academic resources section
                            academic_resources = "\n".join([
                                f"- [{name}]({url})" for name, url in database_links.items()
                            ])

                            # Calculate metrics safely
                            total_sources = sum(len(results.get(source, [])) for source in ['arxiv', 'medium', 'wordpress'])
                            arxiv_count = len(results.get('arxiv', []))
                            web_count = len(web_results) if isinstance(web_results, list) else 'Multiple'

                            additional_resources = f"""
                            ## 🔗 Additional Resources
                            
                            ### Academic Databases
                            {academic_resources}
                            
                            ### Research Metrics
                            - Total Sources Analyzed: {total_sources}
                            - Academic Papers Found: {arxiv_count}
                            - Web Articles Found: {web_count}
                            
                            Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                            """
                        except Exception as e:
                            additional_resources = "## 🔗 Additional Resources\n\nError processing additional resources."
                            st.error(f"Additional resources error: {str(e)}")

                        # Update full report with new sections
                        full_report = f"""
                        # Comprehensive Research Report: {search_query}
                        
                        {full_research_summary}
                        
                        {web_analysis_section}
                        
                        {additional_resources}
                        """

                        # Add HTML formatting for better display
                        st.markdown("""
                        <style>
                        .report-download {
                            background-color: #4CAF50;
                            color: white;
                            padding: 10px 20px;
                            border-radius: 5px;
                            text-decoration: none;
                            margin: 10px 0;
                            display: inline-block;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        # Offer multiple download formats
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="📥 Download Full Report (TXT)",
                                data=full_report,
                                file_name=f"research_report_{search_query.replace(' ', '_')}.txt",
                                mime="text/plain",
                                key="txt_download"
                            )
                        with col2:
                            st.download_button(
                                label="📥 Download Full Report (MD)",
                                data=full_report,
                                file_name=f"research_report_{search_query.replace(' ', '_')}.md",
                                mime="text/markdown",
                                key="md_download"
                            )
                        
                        add_history("Deep Research", search_query, 
                                  f"Research Report Generated | ArXiv: {len(results['arxiv'])} papers, "
                                  f"Medium: {len(results['medium'])} articles, "
                                  f"WordPress: {len(results['wordpress'])} posts")

elif choice == "AI Only":
    st.subheader("AI Only Responses")
    user_query = st.text_input("Enter your question:")
    
    if st.button("Get AI Response"):
        with st.spinner("Generating AI response..."):
            try:
                headers = {"Authorization": f"Bearer {API_KEY}"}
                payload = {
                    "model": selected_model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful and knowledgeable assistant. Provide detailed and accurate responses."},
                        {"role": "user", "content": user_query}
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.7
                }
                
                response = requests.post(f"{API_BASE_URL}/chat/completions", json=payload, headers=headers)
                
                if response.status_code == 200:
                    result = response.json()['choices'][0]['message']['content']
                    st.markdown("### AI Response:")
                    st.write(result)
                    add_history("AI Only", user_query, result)
                else:
                    st.error(f"Error: {response.json().get('error', {}).get('message', 'Unknown error occurred')}")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

elif choice == "History":
    st.subheader("History")
    # Clear history button
    if st.button("Clear All History"):
        clear_all_history()
        st.success("All history cleared successfully.")
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
    # Load history from database and display
    db_history = get_all_history()
    if db_history:
        for entry in db_history:
            entry_id, feature, input_data, output_data, timestamp = entry
            st.markdown(f"**ID:** {entry_id} | **Feature:** {feature} | **Timestamp:** {timestamp}")
            st.markdown(f"**Input:** {input_data}")
            st.markdown(f"**Output:** {output_data}")
            if st.button(f"Delete Entry {entry_id}", key=f"delete_{entry_id}"):
                delete_history_entry(entry_id)
                if hasattr(st, "experimental_rerun"):
                    st.experimental_rerun()
    else:
        st.write("No history available.")
