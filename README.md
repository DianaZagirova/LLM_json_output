# 🔬 Research Paper Analyzer

Transform academic papers into structured, actionable insights with this powerful research paper analysis tool. Built with state-of-the-art language models and designed for researchers, academics, and knowledge enthusiasts.

## ✨ Features

- **Comprehensive Paper Analysis**: Extract key information including research questions, methods, findings, and conclusions
- **Domain Intelligence**: Automatically categorizes papers across 20+ research domains, from Computer Science to Neuroscience
- **Structured Output**: Generates clean, structured JSON output for easy integration with other tools
- **PDF Support**: Direct analysis of PDF research papers
- **Rich Metadata Extraction**: Captures detailed information about:
  - Authors and affiliations
  - Experimental setups and methodologies
  - Datasets and computational resources
  - Equipment specifications
  - Performance metrics
  - Citations and references

## 🚀 Getting Started

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up your environment variables:
```bash
# Create a .env file with your OpenAI API key
OPENAI_API_KEY=your_api_key_here
```

## 📊 Usage

Analyze a research paper:
```bash
python research_paper_analyzer.py path/to/your/paper.pdf
```

The analyzer will generate a detailed JSON output file containing structured information about the paper.

## 🏗️ Architecture

Built with modern Python tools and libraries:
- LangChain for robust language model interactions
- Pydantic for type-safe data modeling
- PyPDF for PDF processing
- OpenAI's language models for advanced text analysis

## 📝 Output Format

The analyzer generates comprehensive, structured data including:
- Basic paper information (title, authors, publication details)
- Research context and objectives
- Methodology details
- Key findings and conclusions
- Technical specifications and resources used
- Citation information

---
Built with 💡 for the research community
