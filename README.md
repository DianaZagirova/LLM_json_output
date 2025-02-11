# 🎯 Universal JSON Output from LLM

A flexible approach to get JSON outputs from Large Language Models (LLMs) of any complexity. 
Tested to work more efficient than other solutions indluding `response_format={ "type": "json_object" }` and other approaches.

See Medium post: [Forcing LLM JSON Outputs: How to Make LLMs Output Complex JSONs](https://medium.com/@d.zagirowa/forcing-llm-json-outputs-how-to-make-llm-output-complex-jsons-a8bb00e87f71)

## 🔑 Key Features

- **Flexible Schema Definition**: Define your desired output structure using Python dataclasses (use LLM to help you convert text description of the output to dataclass definition)
- **Type Safety**: Built-in type validation ensures your LLM outputs match your schema
- **Nested Structures**: Handle complex, deeply nested JSON structures with ease
- **Universal Applicability**: Adapt the approach to any use case by modifying the output dataclass

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

## 💡 How It Works

1. **Define Your Schema**: Create a Python dataclass that describes your desired JSON structure

```python
# Example: Define your output structure
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class YourCustomOutput:
    title: str
    categories: List[str]
    metadata: dict
    # Add any fields you need to match your use case
```
2. **Prompt Engineering**: Use JsonOutputParser to auto-generates JSON Schema from Pydantic model. Insert it in your prompt with {format_instructions}

```python
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser(pydantic_object=ResearchPaper)
    
prompt_template = """Your main task is to extract comprehensive information from the following text and output it in the defined JSON format.

### TEXT: 
{user_input}

### OUTPUT FORMAT: 
{format_instructions}
"""

user_message = PromptTemplate(
      template=prompt_template,
      input_variables=["user_input"],
      partial_variables={"format_instructions": parser.get_format_instructions()},
    )
```

3. **Run the Chain & Parse**: Now you can run the llm with the prompt or create a chain for easy parsing directly to JSON

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        model="gpt-4o-mini", 
        temperature=0.3  # Lower temperature for more factual outputs
    )
chain = user_message | llm | parser
json_output = chain.invoke({"user_input": text_to_parse})    

```

## 📊 Example Use Case - Research Paper Analyzer

This repository includes a practical example that transforms academic papers into structured data. To try it:

```bash
python research_paper_analyzer.py path/to/your/paper.pdf
```

The analyzer will generate a detailed JSON output file containing structured information based on the defined schema.

## 🔄 Adapt to Your Needs

To use this approach for your own use case:

1. Create a new dataclass that defines your desired output structure
2. Modify the prompt template to match your domain
3. Update the validation logic if needed

---
Built with 💡 for the research community
