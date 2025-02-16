import streamlit as st
import requests
import base64
import os
from duckduckgo_search import DDGS
import PyPDF2
from PIL import Image
import io
import sqlite3
import urllib.parse  # Added for URL parsing

# Database setup
conn = sqlite3.connect('history.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS history
             (id INTEGER PRIMARY KEY, feature TEXT, input TEXT, output TEXT)''')
conn.commit()

def save_history_to_db(feature, input_data, output_data):
    # Convert list-type parameters to a string representation
    if isinstance(input_data, list):
        input_data = "\n".join(map(str, input_data))
    if isinstance(output_data, list):
        output_data = "\n".join(map(str, output_data))
    c.execute("INSERT INTO history (feature, input, output) VALUES (?, ?, ?)", (feature, input_data, output_data))
    conn.commit()

def load_history_from_db():
    c.execute("SELECT feature, input, output FROM history")
    return c.fetchall()

def clear_history_from_db():
    c.execute("DELETE FROM history")
    conn.commit()

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
        "model": selected_model,  # Use the globally selected model
        "messages": [
            {"role": "system", "content": "You are an assistant summarizing a document."},
            {"role": "user", "content": f"Summarize the following text:\n{input_text}\nBased on the document, answer the question: {query}"}
        ],
        "max_tokens": 10000
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
    }
    response = requests.post(f"{API_BASE_URL}/chat/completions", json=payload, headers=headers)
    if response.status_code == 200:
        result = response.json()
        if 'choices' in result:
            return result['choices'][0]['message']['content']
        else:
            return "Error: 'choices' not found in the response."
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
            return st.error("Location cannot be empty.")
        url = f"https://api.tomorrow.io/v4/weather/realtime?location={location}&apikey={WEATHER_API_KEY}"
        headers = {"accept": "application/json"}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            weather_data = response.json()["data"]["values"]
            return st.markdown(format_weather_display(weather_data, location), unsafe_allow_html=True)
        else:
            return st.error(f"Failed to fetch weather data for {location}.")
    except Exception as e:
        return st.error(f"An error occurred: {e}")

# Function to perform DuckDuckGo search
def search_duckduckgo(query, max_results=10):
    try:
        results = []
        with DDGS() as ddgs:
            for idx, result in enumerate(ddgs.text(query, max_results=max_results, region="in-en"), start=1):
                results.append(f"{idx}. {result['title']}\nURL: {result['href']}\nSnippet: {result['body']}")
        return results
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 429:
            return [f"Rate limit exceeded. Please try again later."]
        else:
            return [f"HTTP error occurred: {http_err}"]
    except Exception as e:
        return [f"An error occurred: {e}"]

# Function to perform DuckDuckGo search in incognito mode
def search_duckduckgo_incognito(query, max_results=10):
    try:
        results = []
        with DDGS() as ddgs:
            for idx, result in enumerate(ddgs.text(query, max_results=max_results, region="in-en", safesearch="Off"), start=1):
                results.append(f"{idx}. {result['title']}\nURL: {result['href']}\nSnippet: {result['body']}")
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

# Move format_weather_display here
def format_weather_display(weather_data, location="Current Location"):
    icons = {
        "temperature": "🌡️",
        "apparent": "🌡️",
        "humidity": "💧",
        "wind": "💨",
        "cloud": "☁️",
        "visibility": "👁️",
        "uv": "☀️",
        "time": "🕒"
    }
    weather_html = f"""
    <div class="weather-container">
        <h2>{icons.get("location", "📍")} {location} Weather</h2>
        <div class="weather-icon">🌤️</div>
        <div class="weather-info">
            {icons["temperature"]} Temperature: <span class="weather-value">{weather_data.get("temperature", "N/A")}°C</span>
        </div>
        <div class="weather-info">
            {icons["apparent"]} Feels Like: <span class="weather-value">{weather_data.get("temperatureApparent", "N/A")}°C</span>
        </div>
        <div class="weather-info">
            {icons["humidity"]} Humidity: <span class="weather-value">{weather_data.get("humidity", "N/A")}%</span>
        </div>
        <div class="weather-info">
            {icons["wind"]} Wind Speed: <span class="weather-value">{weather_data.get("windSpeed", "N/A")} m/s</span>
        </div>
        <div class="weather-info">
            {icons["cloud"]} Cloud Cover: <span class="weather-value">{weather_data.get("cloudCover", "N/A")}%</span>
        </div>
        <div class="weather-info">
            {icons["visibility"]} Visibility: <span class="weather-value">{weather_data.get("visibility", "N/A")} km</span>
        </div>
        <div class="weather-info">
            {icons["uv"]} UV Index: <span class="weather-value">{weather_data.get("uvIndex", "N/A")}</span>
        </div>
    </div>
    """
    return weather_html

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
            "model": selected_model,  # Use the globally selected model
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
                # Remove technical details from the description
                description = description.split("9j/")[0]
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
def handle_web_search(query, incognito=False):
    if incognito:
        results = search_duckduckgo_incognito(query)
    else:
        results = search_duckduckgo(query)
    return "\n".join(results)

# Function to fetch weather for the device's current location
def fetch_device_location_weather(latitude, longitude):
    try:
        weather_url = f"https://api.tomorrow.io/v4/weather/realtime?location={latitude},{longitude}&apikey={WEATHER_API_KEY}"
        headers = {"accept": "application/json"}
        weather_response = requests.get(weather_url, headers=headers)

        if weather_response.status_code == 200:
            weather_data = weather_response.json()["data"]["values"]
            return st.markdown(format_weather_display(weather_data), unsafe_allow_html=True)
        else:
            return st.error(f"Failed to fetch weather data. Status Code: {weather_response.status_code}")
    except Exception as e:
        return st.error(f"An error occurred: {e}")

# Add JavaScript to get device location
st.markdown("""
<script>
navigator.geolocation.getCurrentPosition(
    function(position) {
        const latitude = position.coords.latitude;
        const longitude = position.coords.longitude;
        document.getElementById('latitude').value = latitude;
        document.getElementById('longitude').value = longitude;
        document.getElementById('location-form').submit();
    },
    function(error) {
        console.error("Error getting location: ", error);
    }
);
</script>
<form id="location-form" method="post">
    <input type="hidden" id="latitude" name="latitude">
    <input type="hidden" id="longitude" name="longitude">
</form>
""", unsafe_allow_html=True)

# Handle form submission to get weather data
query_params = st.experimental_get_query_params()
if query_params.get("latitude") and query_params.get("longitude"):
    latitude = query_params.get("latitude")[0]
    longitude = query_params.get("longitude")[0]
    fetch_device_location_weather(latitude, longitude)

# Main Streamlit App
st.title("Multitool Chat Assistant")

# Update the CSS section with dark theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');

/* Modern container styling with dark theme */
.stApp {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
    color: #e1e1e1 !important;
}

/* Animated title with darker theme */
.title-animation {
    background: linear-gradient(120deg, #00fff2 0%, #4d8cff 100%);
    background-clip: text;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient 5s ease infinite;
}

/* Glowing button effect with darker colors */
.stButton>button {
    background: linear-gradient(45deg, #4d8cff, #00fff2) !important;
    color: #1a1a2e !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0, 255, 242, 0.2) !important;
    font-weight: bold !important;
}

.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 255, 242, 0.4) !important;
}

/* Card-like containers with dark theme */
.css-1r6slb0 {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 20px !important;
    padding: 20px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 255, 242, 0.1) !important;
    border: 1px solid rgba(0, 255, 242, 0.18) !important;
    margin-bottom: 20px !important;
}

/* Sidebar styling with dark theme */
.css-1d391kg {
    background: rgba(26, 26, 46, 0.95) !important;
    backdrop-filter: blur(15px) !important;
}

/* Input fields styling with dark theme */
.stTextInput>div>div>input {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(0, 255, 242, 0.2) !important;
    border-radius: 10px !important;
    color: #e1e1e1 !important;
    padding: 10px !important;
}

/* Selectbox styling with dark theme */
.stSelectbox>div>div {
    background: rgba(255, 255, 255, 0.05) !important;
    border-radius: 10px !important;
    color: #e1e1e1 !important;
}

/* Progress bar with darker theme */
.progress-bar {
    height: 4px;
    background: linear-gradient(90deg, #00fff2 0%, #4d8cff 100%);
    animation: progress 2s ease-in-out;
}

/* Loading spinner with darker theme */
.loading-spinner {
    border: 5px solid rgba(255, 255, 255, 0.1);
    border-top: 5px solid #00fff2;
}

/* Text color adjustments for dark theme */
p, span, label, .stMarkdown {
    color: #e1e1e1 !important;
}

/* Header text colors */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

/* Links color */
a {
    color: #00fff2 !important;
}

/* Markdown text color */
.stMarkdown div {
    color: #e1e1e1 !important;
}

/* Enhanced button animation keyframes */
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

@keyframes glow {
    0% { box-shadow: 0 0 5px rgba(255, 0, 0, 0.5); }
    50% { box-shadow: 0 0 20px rgba(255, 0, 0, 0.8); }
    100% { box-shadow: 0 0 5px rgba(255, 0, 0, 0.5); }
}

/* Updated button styling with red theme */
.stButton>button {
    background: linear-gradient(45deg, #ff0000, #ff4444) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: bold !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    transition: all 0.3s ease !important;
    animation: pulse 2s infinite !important;
    position: relative !important;
    overflow: hidden !important;
    box-shadow: 0 4px 15px rgba(255, 0, 0, 0.3) !important;
}

.stButton>button::before {
    content: '' !important;
    position: absolute !important;
    top: -50% !important;
    left: -50% !important;
    width: 200% !important;
    height: 200% !important;
    background: rgba(255, 255, 255, 0.1) !important;
    transform: rotate(45deg) !important;
    transition: all 0.3s ease !important;
}

.stButton>button:hover {
    transform: translateY(-2px) !important;
    background: linear-gradient(45deg, #ff4444, #ff0000) !important;
    animation: pulse 2s infinite, glow 2s infinite !important;
}

.stButton>button:hover::before {
    left: 100% !important;
}

.stButton>button:active {
    transform: translateY(1px) !important;
    box-shadow: 0 2px 10px rgba(255, 0, 0, 0.2) !important;
}

/* Enhanced Button Animations */
@keyframes neon-glow {
    0% {
        box-shadow: 0 0 5px #ff0000,
                   0 0 10px #ff0000,
                   0 0 15px #ff0000,
                   0 0 20px #ff0000;
    }
    50% {
        box-shadow: 0 0 10px #ff0000,
                   0 0 20px #ff0000,
                   0 0 30px #ff0000,
                   0 0 40px #ff0000;
    }
    100% {
        box-shadow: 0 0 5px #ff0000,
                   0 0 10px #ff0000,
                   0 0 15px #ff0000,
                   0 0 20px #ff0000;
    }
}

@keyframes button-pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.02); }
    100% { transform: scale(1); }
}

@keyframes shine {
    0% { background-position: -100% 50%; }
    100% { background-position: 200% 50%; }
}

/* Updated Button Styling */
.stButton>button {
    background: linear-gradient(45deg, #ff0000, #ff4444, #ff0000) !important;
    background-size: 200% 200% !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: bold !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    transition: all 0.3s ease !important;
    position: relative !important;
    overflow: hidden !important;
    animation: button-pulse 2s infinite, shine 3s infinite !important;
    box-shadow: 0 0 10px rgba(255, 0, 0, 0.5),
                0 0 20px rgba(255, 0, 0, 0.3),
                0 0 30px rgba(255, 0, 0, 0.2),
                inset 0 0 15px rgba(255, 255, 255, 0.1) !important;
}

.stButton>button::before {
    content: '' !important;
    position: absolute !important;
    top: -50% !important;
    left: -50% !important;
    width: 200% !important;
    height: 200% !important;
    background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%) !important;
    transform: rotate(45deg) !important;
    transition: all 0.5s ease !important;
}

.stButton>button:hover {
    transform: translateY(-3px) !important;
    background-position: right center !important;
    animation: neon-glow 1.5s infinite, button-pulse 2s infinite !important;
    box-shadow: 0 0 15px rgba(255, 0, 0, 0.7),
                0 0 30px rgba(255, 0, 0, 0.5),
                0 0 45px rgba(255, 0, 0, 0.3),
                inset 0 0 20px rgba(255, 255, 255, 0.2) !important;
}

.stButton>button:hover::before {
    left: 100% !important;
    transition: 0.8s all ease !important;
}

.stButton>button:active {
    transform: translateY(2px) !important;
    box-shadow: 0 0 20px rgba(255, 0, 0, 0.8),
                inset 0 0 10px rgba(255, 0, 0, 0.4) !important;
}

/* Add 3D effect on hover */
.stButton>button::after {
    content: '' !important;
    position: absolute !important;
    left: 0 !important;
    top: 0 !important;
    width: 100% !important;
    height: 100% !important;
    background: linear-gradient(rgba(255,255,255,0.2), transparent) !important;
    clip-path: polygon(0 0, 100% 0, 100% 25%, 0 45%) !important;
    transition: all 0.3s ease !important;
}

.stButton>button:hover::after {
    transform: translateY(2px) !important;
    opacity: 0.7 !important;
}

/* Add ripple effect on click */
@keyframes ripple {
    to {
        transform: scale(4);
        opacity: 0;
    }
}

.stButton>button .ripple {
    position: absolute !important;
    border-radius: 50% !important;
    transform: scale(0) !important;
    animation: ripple 0.6s linear !important;
    background-color: rgba(255, 255, 255, 0.7) !important;
}

</style>

<script>
function createRipple(event) {
    const button = event.currentTarget;
    const circle = document.createElement('span');
    const diameter = Math.max(button.clientWidth, button.clientHeight);
    const radius = diameter / 2;

    circle.style.width = circle.style.height = `${diameter}px`;
    circle.style.left = `${event.clientX - button.offsetLeft - radius}px`;
    circle.style.top = `${event.clientY - button.offsetTop - radius}px`;
    circle.classList.add('ripple');

    const ripple = button.getElementsByClassName('ripple')[0];
    if (ripple) {
        ripple.remove();
    }

    button.appendChild(circle);
}

document.querySelectorAll('.stButton>button').forEach(button => {
    button.addEventListener('click', createRipple);
});
</script>

<style>
/* ... existing CSS ... */

/* Weather Icons and Animations */
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
    100% { transform: translateY(0px); }
}

@keyframes pulse-weather {
    0% { transform: scale(1); }
    50% { transform: scale(1.1); }
    100% { transform: scale(1); }
}

.weather-container {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 20px;
    margin: 10px 0;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    animation: float 6s ease-in-out infinite;
}

.weather-icon {
    font-size: 2.5em;
    margin: 10px 0;
    display: inline-block;
    animation: pulse-weather 2s infinite;
}

.weather-info {
    display: flex;
    align-items: center;
    margin: 8px 0;
    padding: 8px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    transition: all 0.3s ease;
}

.weather-info:hover {
    background: rgba(255, 255, 255, 0.1);
    transform: translateX(5px);
}

.weather-value {
    color: #ff4444;
    font-weight: bold;
    margin-left: 10px;
}

</style>

<style>
/* Add glitch and broken light flickering effect for headlines */
@keyframes broken-flicker {
    0% { 
        opacity: 1;
        text-shadow: 
            0 0 5px #fff,
            0 0 10px #fff,
            0 0 20px #ff0000,
            0 0 40px #ff0000,
            0 0 80px #ff0000;
    }
    2% { 
        opacity: 0.1;
        text-shadow: none;
    }
    4% { 
        opacity: 1;
        text-shadow: 
            0 0 5px #fff,
            0 0 10px #fff,
            0 0 20px #ff0000;
    }
    19% { 
        opacity: 1;
        text-shadow: 
            0 0 5px #fff,
            0 0 10px #fff,
            0 0 20px #ff0000,
            0 0 40px #ff0000;
    }
    21% { 
        opacity: 0.2;
        text-shadow: none;
    }
    23% { 
        opacity: 1;
        text-shadow: 
            0 0 5px #fff,
            0 0 10px #fff,
            0 0 20px #ff0000,
            0 0 40px #ff0000;
    }
    80% { 
        opacity: 1;
        text-shadow: 
            0 0 5px #fff,
            0 0 10px #fff,
            0 0 20px #ff0000;
    }
    83% { 
        opacity: 0.4;
        text-shadow: none;
    }
    87% { 
        opacity: 1;
        text-shadow: 
            0 0 5px #fff,
            0 0 10px #fff,
            0 0 20px #ff0000;
    }
}

h1, h2, h3, h4, h5, h6, .subheader {
    color: #ffffff !important;
    animation: broken-flicker 5s infinite;
    text-shadow: 
        0 0 5px #fff,
        0 0 10px #fff,
        0 0 20px #ff0000,
        0 0 40px #ff0000,
        0 0 80px #ff0000;
}

/* Enhanced glitch effect for main title */
.title-animation {
    animation: broken-flicker 5s infinite, gradient 5s ease infinite !important;
    position: relative;
}

/* Additional distortion effect */
.title-animation::before {
    content: attr(data-text);
    position: absolute;
    left: 2px;
    text-shadow: -1px 0 #ff0000;
    top: 0;
    color: #ffffff;
    overflow: hidden;
    clip: rect(0, 900px, 0, 0);
    animation: broken-noise-anim 3s infinite linear alternate-reverse;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Update headline styling with Caribbean green and softer glow */
@keyframes soft-flicker {
    0% { 
        opacity: 1;
        text-shadow: 
            0 0 5px rgba(0, 206, 201, 0.5),
            0 0 10px rgba(0, 206, 201, 0.3),
            0 0 15px rgba(0, 206, 201, 0.2);
    }
    50% { 
        opacity: 0.95;
        text-shadow: 
            0 0 7px rgba(0, 206, 201, 0.6),
            0 0 12px rgba(0, 206, 201, 0.4),
            0 0 17px rgba(0, 206, 201, 0.3);
    }
    100% { 
        opacity: 1;
        text-shadow: 
            0 0 5px rgba(0, 206, 201, 0.5),
            0 0 10px rgba(0, 206, 201, 0.3),
            0 0 15px rgba(0, 206, 201, 0.2);
    }
}

h1, h2, h3, h4, h5, h6, .subheader {
    color: #00CEC9 !important;
    animation: soft-flicker 3s infinite;
    text-shadow: 
        0 0 5px rgba(0, 206, 201, 0.5),
        0 0 10px rgba(0, 206, 201, 0.3),
        0 0 15px rgba(0, 206, 201, 0.2);
}

/* Update title animation */
.title-animation {
    background: linear-gradient(120deg, #00CEC9 0%, #81ECEC 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: soft-flicker 3s infinite;
}
</style>
""", unsafe_allow_html=True)

# Add loading animation function
def show_loading_animation():
    with st.spinner(''):
        st.markdown("""
            <div class="loading-spinner"></div>
            <div class="progress-bar"></div>
        """, unsafe_allow_html=True)

# Modified title with natural emoji and animation
st.markdown("""
<h1>
    <span style="font-size: 32px; margin-right: 10px;">🤖</span>
    <span class="title-animation">Multitool Chat Assistant v2.0</span>
</h1>

<style>
.title-animation {
    background: linear-gradient(120deg, #00fff2 0%, #4d8cff 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# Add feature icons to the menu
feature_icons = {
    "Query Processing": "💭",
    "Weather Information": "🌤️",
    "PDF Summarization": "📄",
    "Image Search": "🎨",
    "Picture Explanation": "🖼️",
    "Web Search": "🔍",
    "History": "📚"
}

# Update menu with icons
menu = [f"{feature_icons[item]} {item}" for item in ["Query Processing", "Weather Information", "PDF Summarization", 
                                                    "Image Search", "Picture Explanation", "Web Search", "History"]]

# Sidebar Menu
choice = st.sidebar.selectbox("Choose a Feature", menu)

history = st.session_state.get("history", load_history_from_db())

# Add a button to clear history
if st.sidebar.button("Clear History"):
    history = []
    st.session_state["history"] = history
    clear_history_from_db()
    st.success("History cleared successfully.")

# Modify feature sections to add animations
if choice == "💭 Query Processing":
    st.markdown('<div class="feature-container">', unsafe_allow_html=True)
    st.subheader("💭 Query Processing")
    user_query = st.text_input("Enter your query:")

    if st.button("Submit Query"):
        if 'weather' in user_query.lower():
            if 'in' in user_query.lower():
                location = user_query.split('in')[-1].strip()
                result = fetch_specified_location_weather(location)
            else:
                result = fetch_current_location_weather()
        else:
            payload = {
                "model": selected_model,  # Use the globally selected model
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_query}
                ],
                "max_tokens": 4000  # Add missing max_tokens value
            }
            headers = {
                "Authorization": f"Bearer {API_KEY}"
            }
            response = requests.post(f"{API_BASE_URL}/chat/completions", json=payload, headers=headers)
            if response.status_code == 200:
                result_data = response.json()
                if 'choices' in result_data:
                    result = result_data['choices'][0]['message']['content']
                else:
                    result = "Error: 'choices' not found in the response."
            else:
                result = f"Error in query processing: {response.text}"

        st.write(result)
        history.append(("Query Processing", user_query, result))
        save_history_to_db("Query Processing", user_query, result)

elif choice == "🌤️ Weather Information":
    st.markdown('<div class="feature-container">', unsafe_allow_html=True)
    st.subheader("🌤️ Weather Information")
    
    # Add tabs for different weather views
    weather_tab1, weather_tab2 = st.tabs(["📍 Current Location", "🔍 Search Location"])
    
    with weather_tab1:
        if st.button("Get Current Weather", key="current_weather"):
            show_loading_animation()
            fetch_current_location_weather()
    
    with weather_tab2:
        location = st.text_input("Enter a location:", placeholder="e.g., London, UK")
        if st.button("Get Weather", key="search_weather"):
            if location:
                show_loading_animation()
                fetch_specified_location_weather(location)
                history.append(("Weather Information", location, "Weather data fetched"))
                save_history_to_db("Weather Information", location, "Weather data fetched")

elif choice == "🎨 Image Search":
    st.markdown('<div class="feature-container">', unsafe_allow_html=True)
    st.subheader("🎨 Image Search")
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
        history.append(("Image Search", prompt, image_path))
        save_history_to_db("Image Search", prompt, image_path)

elif choice == "🖼️ Picture Explanation":
    st.markdown('<div class="feature-container">', unsafe_allow_html=True)
    st.subheader("🖼️ Picture Explanation")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image_path = os.path.join("temp_data", uploaded_image.name)
        with open(image_path, "wb") as file:
            file.write(uploaded_image.read())
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        # Display the uploaded image on the left
        with col1:
            st.image(image_path)
        
        # Display the summarized output on the right
        with col2:
            query = st.text_input("Enter your query about the image:")
            if st.button("Generate Text"):
                summary = summarize_image(image_path, query)
                if "Error" in summary:
                    st.error(summary)
                else:
                    st.write(summary)
                history.append(("Picture Explanation", uploaded_image.name, summary))
                save_history_to_db("Picture Explanation", uploaded_image.name, summary)

elif choice == "📄 PDF Summarization":
    st.markdown('<div class="feature-container">', unsafe_allow_html=True)
    st.subheader("📄 PDF Summarization")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_dir = "temp_data"
        os.makedirs(temp_dir, exist_ok=True)
        filepath = os.path.join(temp_dir, uploaded_file.name)
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        # Check file size and compress if necessary
        if os.path.getsize(filepath) > 10 * 1024 * 1024:  # 10MB
            compressed_filepath = os.path.join(temp_dir, f"compressed_{uploaded_file.name}")
            compress_result = compress_pdf(filepath, compressed_filepath)
            if "Error" in compress_result:
                st.error(compress_result)
                filepath = None
            else:
                filepath = compressed_filepath

        if filepath:
            # Create two columns
            col1, col2 = st.columns(2)

            # Display the PDF on the left
            with col1:
                st.info("Uploaded File")
                display_pdf(filepath)

            # Query and answer section on the right
            with col2:
                st.info("Enter your query(s) about the document:")
                queries = st.text_area("Your Queries", placeholder="Enter each query separated by new lines...")

                if st.button("Submit Queries"):
                    queries_list = queries.strip().split("\n")
                    answers = []
                    st.subheader("Summarized Answers")

                    # Process each query one by one
                    for i, query in enumerate(queries_list, start=0):
                        with st.spinner(f"Processing query {i}: {query}..."):
                            summary = llm_pipeline(filepath, query)
                            st.markdown(f"**{i}.** {summary}")
                            answers.append(summary)
                    history.append(("PDF Summarization", queries_list, answers))
                    save_history_to_db("PDF Summarization", queries_list, answers)

elif choice == "🔍 Web Search":
    st.markdown('<div class="feature-container">', unsafe_allow_html=True)
    st.subheader("🔍 Web Search")
    search_query = st.text_input("Enter your search query:")
    incognito_mode = st.checkbox("Incognito Mode")

    if st.button("Search"):
        if incognito_mode:
            results = search_duckduckgo_incognito(search_query)
        else:
            results = search_duckduckgo(search_query)

        if isinstance(results, list):
            st.markdown("### Search Results")
            logo_mapping = {
                "duckduckgo.com": "https://duckduckgo.com/assets/logo_homepage.normal.v108.svg",
                "google.com": "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png",
                "bing.com": "https://www.bing.com/sa/simg/bing_p_rr_teal_min.ico",
            }
            for res in results:
                parts = res.splitlines()
                url_line = next((line for line in parts if line.startswith("URL:")), "")
                url_value = url_line.split("URL:")[-1].strip() if "URL:" in url_line else ""
                domain = urllib.parse.urlparse(url_value).netloc.replace("www.", "")
                # Use a lightweight favicon service with smaller size (sz=16)
                logo = logo_mapping.get(domain) or f"https://www.google.com/s2/favicons?domain={domain}&sz=16"
                st.markdown(f'<img src="{logo}" width="16" style="vertical-align: middle;">', unsafe_allow_html=True)
                st.markdown(f"**{domain}**")
                st.markdown(res)
        else:
            st.error(results)

        if not incognito_mode:
            history.append(("Web Search", search_query, results))
            save_history_to_db("Web Search", search_query, results)

elif choice == "📚 History":
    st.markdown('<div class="feature-container">', unsafe_allow_html=True)
    st.subheader("📚 History")
    if history:
        # Add a button to clear all history
        if st.button("Clear All History"):
            history = []
            st.session_state["history"] = history
            clear_history_from_db()
            st.success("All history cleared successfully.")
        
        for idx, entry in enumerate(history, start=1):
            st.markdown(f"**{idx}. Feature:** {entry[0]}")
            st.markdown(f"**Input:** {entry[1]}")
            st.markdown(f"**Output:** {entry[2]}")
            if st.button(f"Delete Entry {idx}", key=f"delete_{idx}"):
                history.pop(idx-1)
                st.session_state["history"] = history
                c.execute("DELETE FROM history WHERE id = ?", (idx,))
                conn.commit()
                st.experimental_rerun()
    else:
        st.write("No history available.")

st.session_state["history"] = history

# Add footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; animation: fadeIn 2s;'>
        <p style='color: #666; font-size: 0.8em;'>
            Created with ❤️ by Swagata Dey | Version 2.0
        </p>
    </div>
""", unsafe_allow_html=True)

# Update the weather display functions to include icons
def format_weather_display(weather_data, location="Current Location"):
    icons = {
        "temperature": "🌡️",
        "apparent": "🌡️",
        "humidity": "💧",
        "wind": "💨",
        "cloud": "☁️",
        "visibility": "👁️",
        "uv": "☀️",
        "time": "🕒"
    }
    
    weather_html = f"""
    <div class="weather-container">
        <h2>{icons.get("location", "📍")} {location} Weather</h2>
        <div class="weather-icon">🌤️</div>
        <div class="weather-info">
            {icons["temperature"]} Temperature: <span class="weather-value">{weather_data.get("temperature", "N/A")}°C</span>
        </div>
        <div class="weather-info">
            {icons["apparent"]} Feels Like: <span class="weather-value">{weather_data.get("temperatureApparent", "N/A")}°C</span>
        </div>
        <div class="weather-info">
            {icons["humidity"]} Humidity: <span class="weather-value">{weather_data.get("humidity", "N/A")}%</span>
        </div>
        <div class="weather-info">
            {icons["wind"]} Wind Speed: <span class="weather-value">{weather_data.get("windSpeed", "N/A")} m/s</span>
        </div>
        <div class="weather-info">
            {icons["cloud"]} Cloud Cover: <span class="weather-value">{weather_data.get("cloudCover", "N/A")}%</span>
        </div>
        <div class="weather-info">
            {icons["visibility"]} Visibility: <span class="weather-value">{weather_data.get("visibility", "N/A")} km</span>
        </div>
        <div class="weather-info">
            {icons["uv"]} UV Index: <span class="weather-value">{weather_data.get("uvIndex", "N/A")}</span>
        </div>
    </div>
    """
    return weather_html
