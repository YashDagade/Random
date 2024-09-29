import PyPDF2
import openai
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import re

client = OpenAI(api_key="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")

# Function to convert PDF to text
def pdf_to_text(pdf_file_path):
    with open(pdf_file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks of ~10,000 characters, ensuring it doesn’t split mid-sentence
def split_text_into_chunks(text, max_chars=10000):
    chunks = []
    while len(text) > max_chars:
        split_index = max_chars
        while split_index > 0 and text[split_index] not in ['.', '!', '?']:
            split_index -= 1
        if split_index == 0:
            split_index = max_chars
        chunks.append(text[:split_index + 1].strip())
        text = text[split_index + 1:].strip()
    chunks.append(text)
    return chunks

# Function to enhance and format the text using OpenAI API
def format_text_with_openai(chunk):
    prompt = (
        "Please reformat the following text with better formatting, "
        "remove footnotes, page numbers, and any other irrelevant content that is not part of the main text. "
        "Ensure the output is in Markdown format with appropriate titles, headings, and subheadings. "
        "Include all the titles and subtitles and everything in the original text word by word:\n\n"
        f"{chunk}"
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who will rewrite the text with better formatting and remove any advertisements or irrelevant content."},
            {"role": "user", "content": prompt}
        ]
    )

    return str(completion.choices[0].message.content)

# Function to process chunks in parallel
def process_chunks_in_parallel(chunks):
    with ThreadPoolExecutor() as executor:
        formatted_chunks = list(executor.map(format_text_with_openai, chunks))
    return formatted_chunks

# Function to save formatted text as markdown
def save_as_markdown(text, output_markdown_file_path):
    with open(output_markdown_file_path, 'w', encoding='utf-8') as markdown_file:
        markdown_file.write(text)

# Main function to process PDF to formatted markdown
def pdf_to_formatted_markdown(pdf_file_path):
    extracted_text = pdf_to_text(pdf_file_path)
    chunks = split_text_into_chunks(extracted_text)
    formatted_chunks = process_chunks_in_parallel(chunks)
    formatted_text = "\n\n".join(formatted_chunks)

    output_markdown_file_path = pdf_file_path.replace('.pdf', '.md')
    save_as_markdown(formatted_text, output_markdown_file_path)
    return output_markdown_file_path

# Example usage
pdf_file_path = 'Gogol Overcoat.pdf'  # Replace with your PDF file path
output_markdown_file = pdf_to_formatted_markdown(pdf_file_path)
print(f'Formatted text saved as Markdown to: {output_markdown_file}')