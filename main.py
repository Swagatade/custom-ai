import streamlit as st
import requests
import base64
import os
from duckduckgo_search import DDGS
import webbrowser
import PyPDF2
from PIL import Image
import io

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
model_options = ["meta-llama/llama-3.2-11b-vision-instruct:free", "meta-llama/llama-3.2-90b-vision-instruct:free" , "meta-llama/llama-3.1-70b-instruct:free", "huggingfaceh4/zephyr-7b-beta:free", "microsoft/phi-3-mini-128k-instruct:free", "mistralai/mistral-7b-instruct:free", "qwen/qwen-2-7b-instruct:free", "openchat/openchat-7b:free", "google/learnlm-1.5-pro-experimental:free", "google/gemini-2.0-flash-exp:free", "google/gemini-2.0-flash-thinking-exp:free", "meta-llama/llama-3.1-405b-instruct:free", "google/gemini-2.0-flash-thinking-exp:free"]
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
def handle_web_search(query):
    results = search_duckduckgo(query)
    return "\n".join(results)

# Main Streamlit App
st.title("Multitool Chat Assistant")

# Sidebar Menu
menu = ["Query Processing", "Weather Information", "PDF Summarization", "Image Search", "Picture Explanation", "Web Search", "History"]
choice = st.sidebar.selectbox("Choose a Feature", menu)

history = st.session_state.get("history", [])

# Add a button to clear history
if st.sidebar.button("Clear History"):
    history = []
    st.session_state["history"] = history
    st.success("History cleared successfully.")

if choice == "Query Processing":
    st.subheader("Query Processing")
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
                "model": selected_model,
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

elif choice == "Weather Information":
    st.subheader("Weather Information")
    location = st.text_input("Enter a location for weather information (leave blank for current location):")

    if st.button("Get Weather"):
        if location.strip():
            result = fetch_specified_location_weather(location)
        else:
            result = fetch_current_location_weather()

        st.write(result)
        history.append(("Weather Information", location or "Current Location", result))

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
        history.append(("Image Search", prompt, image_path))

elif choice == "Picture Explanation":
    st.subheader("Picture Explanation")
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

elif choice == "PDF Summarization":
    st.subheader("PDF Summarization")
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

elif choice == "Web Search":
    st.subheader("Web Search")
    search_query = st.text_input("Enter your search query:")

    if st.button("Search"):
        result = handle_web_search(search_query)
        if "Rate limit exceeded" in result:
            st.error(result)
        else:
            st.write(result)
        history.append(("Web Search", search_query, result))

elif choice == "History":
    st.subheader("History")
    if history:
        # Add a button to clear all history
        if st.button("Clear All History"):
            history = []
            st.session_state["history"] = history
            st.success("All history cleared successfully.")
        
        for idx, entry in enumerate(history, start=1):
            st.markdown(f"**{idx}. Feature:** {entry[0]}")
            st.markdown(f"**Input:** {entry[1]}")
            st.markdown(f"**Output:** {entry[2]}")
            if st.button(f"Delete Entry {idx}", key=f"delete_{idx}"):
                history.pop(idx-1)
                st.session_state["history"] = history
                st.rerun()
    else:
        st.write("No history available.")

st.session_state["history"] = history
