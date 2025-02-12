import streamlit as st
from openai import OpenAI
import base64
import PyPDF2
from dotenv import load_dotenv


load_dotenv(override=True)
client = OpenAI()


PROMPT = """I have provided a script file and an audio file. The script includes a
'Key Elements' section, which serves as a benchmark for evaluation. The rest of
the script outlines the information that should be conveyed in the conversation.
The audio file contains a spoken version of this content. Please analyze how
closely the spoken content in the audio aligns with the script, specifically
assessing whether it covers the key elements effectively and the language used
completely correct.

Create a table with 3 columns:
- Key Element
- Recording Matches (with options Yes/No/Partially)
- Issues identified

Make sure you are strict about the word choice, formulation, clarity and conciseness.
Offer recommendations and corrections if necessary, especially if incorrect words or
ambiguous formulations are used.
Make sure the table is properly formatted and doesn't have any missing or extra columns
or rows.

Finally, at the end of your response, in a separate section from the previous ones,
analyze and make some comments regarding the
confidence level of the person speaking and make some recommendations on errors, tonality
and word slection.
"""


def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def call_openai_api(prompt, script, audio):
    encoded_string = base64.b64encode(audio).decode("utf-8")

    return client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text"],
        stream=True,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": f"Here is the script: {script}"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": encoded_string, "format": "wav"},
                    },
                ],
            },
        ],
    )


st.title("Streamlit Demo")

uploaded_audio = st.file_uploader("Upload an audio file", type=["wav"])
uploaded_pdf = st.file_uploader("Upload a PDF script", type=["pdf"])
prompt_input = st.text_area("Enter a prompt", value=PROMPT)

if st.button("Process"):
    if uploaded_pdf and uploaded_audio:
        with st.spinner("Extracting text from PDF..."):
            script_text = extract_text_from_pdf(uploaded_pdf)

        with st.spinner("Loading audio file..."):
            audio = uploaded_audio.read()

        with st.spinner("AI analysis..."):
            response = call_openai_api(prompt_input, script_text, audio)

        st.success("AI response:")
        st.write_stream(response)

    else:
        st.error("Please upload both a video file and a PDF script.")
