# BookScribe AI 📚

BookScribe AI is a Streamlit based application that transforms your PDFs into interactive knowledge bases with personalized summaries. It uses advanced AI models like Hugging Face embeddings and Groq LLMs to provide chapter wise summaries and answer questions about the content.

## Features

- **PDF Upload**: Upload your PDF documents to process and analyze.
- **Chapter Summaries**: Automatically generate chapter wise summaries in learning style.
- **Interactive Q&A**: Ask questions about the document and get AI generated answers with relevant context.
- **Learning Style Customization**: Choose from different learning styles (Visual, Auditory, Reading/Writing, Kinesthetic) for personalized summaries.
- **Groq LLM Integration**: Use Groq's powerful language models for summarization and Q&A.
- **Hugging Face Embeddings**: Create vector stores for efficient document search and retrieval.

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Upload a PDF**: Use the file uploader to upload a PDF document.
2. **Configure Settings**:
   - Enter your Groq API Key.
   - Select your learning style.
   - Choose a Groq LLM model.
   - Adjust temperature and max tokens for the language model.
3. **Process the PDF**: Click the "Process PDF" button to extract text, create embeddings, and generate chapter summaries.
4. **Explore Summaries**: View chapter wise summaries in the "Chapter Summaries" section.
5. **Ask Questions**: Use the Q&A section to ask questions about the document and get AI-generated answers.

## Configuration

- **Groq API Key**: Required for using Groq's LLMs. Enter it in the sidebar.
- **Learning Style**: Choose from Visual, Auditory, Reading/Writing, or Kinesthetic.
- **LLM Model**: Select from available Groq models (e.g., `llama3-8b-8192`).

## Limitations

- The application currently supports PDF files.
- Summaries are limited to the first 3000 characters of each chapter for API efficiency.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the web application framework.
- [Hugging Face](https://huggingface.co/) for embedding models.
- [Groq](https://groq.com/) for LLM integration.
- [LangChain](https://langchain.com/) for document processing utilities.

**BookScribe AI** - Transforming the way you interact with knowledge.
