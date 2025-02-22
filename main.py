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
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
    "meta-llama/llama-3.2-90b-vision-instruct:free",
    "meta-llama/llama-3.1-70b-instruct:free",
    "meta-llama/llama-3.1-405b-instruct:free",
    "nvidia/llama-3.1-nemotron-70b-instruct:free",
    "google/gemma-2-9b-it:free",
    "google/gemini-2.0-flash-thinking-exp-1219:free",
    "google/gemini-2.0-flash-exp:free",
    "google/gemini-2.0-flash-thinking-exp:free",
    "google/gemini-2.0-flash-lite-preview-02-05:free",
    "google/learnlm-1.5-pro-experimental:free",
    "google/learnlm-1.5-pro-experimental:free",
    "deepseek/deepseek-r1:free",
    "deepseek/deepseek-r1-distill-llama-70b:free",
    "deepseek/deepseek-chat:free",
    "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
    "mistralai/mistral-small-24b-instruct-2501:free",
    "cognitivecomputations/dolphin3.0-mistral-24b:free",
    "mistralai/mistral-nemo:free",
    "mistralai/mistral-7b-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "microsoft/phi-3-medium-128k-instruct:free",
    "qwen/qwen2.5-vl-72b-instruct:free",
    "qwen/qwen-2-7b-instruct:free",
    "qwen/qwen-vl-plus:free",
    "huggingfaceh4/zephyr-7b-beta:free",
    "openchat/openchat-7b:free"
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
menu = ["Query Processing", "Weather Information", "PDF Summarization", "Image Search", "Picture Explanation", "Web Search", "1-Click Search", "Deep Research", "History"]
choice = st.sidebar.selectbox("Choose a Feature", menu)

if choice == "Query Processing":
    st.subheader("Query Processing")
    user_query = st.text_input("Enter your query:")

    if st.button("Submit Query"):
        # Check if user query is "hi" or "hellow" separately
        if user_query.strip().lower() in ["hi", "hellow"]:
            headers = {"Authorization": f"Bearer {API_KEY}"}
            payload = {
                "model": selected_model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
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
            # ...existing code for other queries...
            if 'weather' in user_query.lower():
                if 'in' in user_query.lower():
                    location = user_query.split('in')[-1].strip()
                    result = fetch_specified_location_weather(location)
                else:
                    result = fetch_current_location_weather()
            else:
                headers = {"Authorization": f"Bearer {API_KEY}"}
                payload_model = {
                    "model": selected_model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_query}
                    ],
                    "max_tokens": 1500
                }
                response_model = requests.post(f"{API_BASE_URL}/chat/completions", json=payload_model, headers=headers)
                if response_model.status_code == 200 and 'choices' in response_model.json():
                    model_answer = response_model.json()['choices'][0]['message']['content']
                else:
                    model_answer = f"Error in model response: {response_model.text}"
                
                web_results = handle_web_search(user_query)
                analysis_prompt = f"Analyze the following web search results and provide key insights:\n{web_results}"
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
                
                combine_prompt = (f"Combine the following insights into a single, coherent answer that integrates both perspectives. "
                                  f"Web analysis: {web_analysis} "
                                  f"Model suggestion: {model_answer}")
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
        st.write(result)
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
    
    search_query = st.text_input("Enter your research query:")
    
    if st.button("Search and Analyze"):
        with st.spinner("Searching papers and articles..."):
            results = search_research_papers(search_query)
            
            if 'error' in results:
                st.error(results['error'])
            else:
                # Display ArXiv results
                st.subheader("ArXiv Papers")
                for paper in results['arxiv']:
                    st.markdown(paper, unsafe_allow_html=True)
                    
                # Display Medium results
                st.subheader("Medium Articles")
                for article in results['medium']:
                    st.markdown(article, unsafe_allow_html=True)
                
                # Display WordPress results
                st.subheader("WordPress Articles")
                for post in results['wordpress']:
                    st.markdown(post, unsafe_allow_html=True)
                
                # AI Analysis of papers with improved error handling
                with st.spinner("Generating research summary..."):
                    summary_prompt = f"Analyze and summarize the key findings from the research papers about: {search_query}"
                    payload = {
                        "model": selected_model,
                        "messages": [
                            {"role": "system", "content": "You are a research assistant providing comprehensive analysis."},
                            {"role": "user", "content": summary_prompt}
                        ],
                        "max_tokens": 1000
                    }
                    headers = {
                        "Authorization": f"Bearer {API_KEY}"
                    }
                    try:
                        response = requests.post(f"{API_BASE_URL}/chat/completions", json=payload, headers=headers)
                        response_data = response.json()
                        
                        if response.status_code == 200 and 'choices' in response_data:
                            if response_data['choices'] and len(response_data['choices']) > 0:
                                if 'message' in response_data['choices'][0]:
                                    summary = response_data['choices'][0]['message'].get('content', 'No summary generated.')
                                    st.subheader("Research Summary")
                                    st.write(summary)
                                else:
                                    st.error("Error: Unexpected API response format - missing message content")
                            else:
                                st.error("Error: No choices returned from API")
                        else:
                            error_message = response_data.get('error', {}).get('message', 'Unknown error occurred')
                            st.error(f"API Error: {error_message}")
                    
                    except requests.exceptions.RequestException as e:
                        st.error(f"Network error occurred: {str(e)}")
                    except ValueError as e:
                        st.error(f"Error parsing API response: {str(e)}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {str(e)}")
                
                add_history("Deep Research", search_query, 
                          f"Found {len(results['arxiv'])} ArXiv papers, "
                          f"{len(results['medium'])} Medium articles, and "
                          f"{len(results['wordpress'])} WordPress posts")

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
