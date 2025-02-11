import os
import logging
import argparse
from datetime import date
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl, EmailStr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.callbacks import get_openai_callback
import json

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ResearchDomain(str, Enum):
    """Major research domains with broader categories."""
    # Science and Technology
    COMPUTER_SCIENCE = "computer_science"
    ARTIFICIAL_INTELLIGENCE = "artificial_intelligence"
    DATA_SCIENCE = "data_science"
    ENGINEERING = "engineering"
    ROBOTICS = "robotics"
    
    # Life Sciences
    BIOLOGY = "biology"
    MEDICINE = "medicine"
    NEUROSCIENCE = "neuroscience"
    GENETICS = "genetics"
    BIOTECHNOLOGY = "biotechnology"
    
    # Physical Sciences
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    MATERIALS_SCIENCE = "materials_science"
    ASTRONOMY = "astronomy"
    EARTH_SCIENCE = "earth_science"
    
    # Mathematics and Computing
    MATHEMATICS = "mathematics"
    STATISTICS = "statistics"
    COMPUTATIONAL_SCIENCE = "computational_science"
    QUANTUM_COMPUTING = "quantum_computing"
    
    # Interdisciplinary
    BIOINFORMATICS = "bioinformatics"
    ENVIRONMENTAL_SCIENCE = "environmental_science"
    COGNITIVE_SCIENCE = "cognitive_science"
    NANOTECHNOLOGY = "nanotechnology"

class StudyType(str, Enum):
    """Types of research studies with methodology classification."""
    # Primary Research
    EXPERIMENTAL = "experimental"
    OBSERVATIONAL = "observational"
    COMPUTATIONAL = "computational"
    THEORETICAL = "theoretical"
    
    # Secondary Research
    META_ANALYSIS = "meta_analysis"
    SYSTEMATIC_REVIEW = "systematic_review"
    LITERATURE_REVIEW = "literature_review"
    
    # Clinical Studies
    CLINICAL_TRIAL = "clinical_trial"
    CASE_STUDY = "case_study"
    COHORT_STUDY = "cohort_study"
    
    # Other Methods
    SURVEY = "survey"
    QUALITATIVE = "qualitative"
    MIXED_METHODS = "mixed_methods"
    LONGITUDINAL = "longitudinal"
    CROSS_SECTIONAL = "cross_sectional"

class DataCategory(str, Enum):
    """Categories of research data."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TIME_SERIES = "time_series"
    SPATIAL = "spatial"
    GENOMIC = "genomic"
    SPECTROSCOPIC = "spectroscopic"
    IMAGING = "imaging"
    SURVEY_RESPONSES = "survey_responses"
    CLINICAL = "clinical"
    ENVIRONMENTAL = "environmental"

# Optional AI-specific enums for backward compatibility
class ModelArchitecture(str, Enum):
    """AI model architectures (optional)."""
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    GAN = "gan"
    DIFFUSION = "diffusion"
    MLP = "mlp"
    HYBRID = "hybrid"

class Equipment(BaseModel):
    """Model representing equipment or resources used in research."""
    name: str = Field(description="Name of the equipment")
    type: str = Field(description="Type of equipment")
    specifications: Dict[str, Any] = Field(description="Technical specifications")
    manufacturer: Optional[str] = Field(None, description="Equipment manufacturer")
    model: Optional[str] = Field(None, description="Model number or version")
    settings: Optional[Dict[str, Any]] = Field(None, description="Equipment settings used")
    calibration_date: Optional[date] = Field(None, description="Last calibration date")

class ComputeResource(BaseModel):
    """Model representing computational resources used (if applicable)."""
    hardware_type: str = Field(description="Type of hardware (GPU, TPU, CPU, HPC cluster)")
    model: str = Field(description="Specific model of the hardware")
    quantity: int = Field(description="Number of units used")
    memory: str = Field(description="Memory per unit")
    provider: Optional[str] = Field(None, description="Provider or facility")
    hours_used: float = Field(description="Total compute hours used")
    cost_estimate: Optional[float] = Field(None, description="Estimated cost in USD")

class Dataset(BaseModel):
    """Model representing a dataset or samples used in the research."""
    name: str = Field(description="Name or identifier of the dataset/samples")
    category: DataCategory = Field(description="Category of data")
    size: str = Field(description="Size or quantity")
    collection_method: str = Field(description="Method of data collection")
    time_period: Optional[str] = Field(None, description="Time period of data collection")
    location: Optional[str] = Field(None, description="Geographic location if applicable")
    source: Optional[HttpUrl] = Field(None, description="URL to the dataset")
    access_restrictions: Optional[str] = Field(None, description="Data access restrictions")
    quality_controls: Optional[List[str]] = Field(None, description="Quality control measures")

class ModelPerformance(BaseModel):
    """Model representing performance metrics."""
    metric_name: str = Field(description="Name of the metric")
    value: float = Field(description="Metric value")
    confidence_interval: Optional[List[float]] = Field(None, description="Confidence interval")
    comparison_baseline: Optional[str] = Field(None, description="Baseline model for comparison")
    improvement: Optional[float] = Field(None, description="Improvement over baseline")

class ExperimentalSetup(BaseModel):
    """Model representing experimental setup details."""
    batch_size: int = Field(description="Training batch size")
    num_epochs: int = Field(description="Number of training epochs")
    optimizer: str = Field(description="Optimization algorithm")
    learning_rate: float = Field(description="Learning rate")
    loss_function: str = Field(description="Loss function used")
    regularization: Optional[Dict[str, Any]] = Field(None, description="Regularization techniques")
    data_augmentation: Optional[List[str]] = Field(None, description="Data augmentation methods")

class Author(BaseModel):
    """Model representing a paper author."""
    name: str = Field(description="Author's full name")
    affiliation: str = Field(description="Author's institution")
    email: Optional[EmailStr] = Field(None, description="Author's email")
    is_corresponding: bool = Field(description="Whether this is the corresponding author")
    orcid: Optional[str] = Field(None, description="ORCID identifier")

class Citation(BaseModel):
    """Model representing a citation."""
    title: str = Field(description="Title of the cited paper")
    authors: List[str] = Field(description="Authors of the cited paper")
    year: int = Field(description="Publication year")
    venue: str = Field(description="Publication venue")
    doi: Optional[str] = Field(None, description="DOI of the cited paper")
    citations_count: Optional[int] = Field(None, description="Number of citations")

class ResearchPaper(BaseModel):
    """Model for research paper analysis with basic and advanced fields."""
    # Basic Information (Always Required)
    title: str = Field(description="Paper title")
    authors: List[Author] = Field(description="List of authors")
    publication_date: date = Field(description="Publication date")
    venue: str = Field(description="Publication venue or journal")
    abstract: str = Field(description="Paper abstract")
    keywords: List[str] = Field(description="Keywords or key terms")
    domains: List[ResearchDomain] = Field(description="Research domains")
    study_type: StudyType = Field(description="Type of study")
    
    # Core Content (Always Required)
    research_questions: List[str] = Field(description="Main research questions or objectives")
    methods_summary: str = Field(description="Summary of methods used")
    main_findings: List[str] = Field(description="Key findings and contributions")
    conclusions: str = Field(description="Main conclusions")
    
    # Extended Metadata (Optional)
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    arxiv_id: Optional[str] = Field(None, description="arXiv identifier")
    pmid: Optional[str] = Field(None, description="PubMed ID")
    funding_sources: Optional[List[str]] = Field(None, description="Funding sources")
    citations_count: Optional[int] = Field(None, description="Number of citations")
    
    # Detailed Content (Optional)
    background: Optional[str] = Field(None, description="Research background and context")
    hypotheses: Optional[List[str]] = Field(None, description="Research hypotheses")
    novelty_claims: Optional[List[str]] = Field(None, description="Claims of novelty")
    theoretical_framework: Optional[str] = Field(None, description="Theoretical framework or model")
    
    # Methodology Details (Optional)
    study_design: Optional[Dict[str, Any]] = Field(None, description="Detailed study design")
    data_collection: Optional[str] = Field(None, description="Data collection methods")
    data_analysis: Optional[str] = Field(None, description="Data analysis approach")
    statistical_methods: Optional[List[str]] = Field(None, description="Statistical methods")
    equipment_used: Optional[List[Equipment]] = Field(None, description="Equipment used")
    protocols: Optional[List[str]] = Field(None, description="Experimental protocols")
    
    # Results and Analysis (Optional)
    quantitative_results: Optional[Dict[str, Any]] = Field(None, description="Quantitative findings")
    qualitative_results: Optional[str] = Field(None, description="Qualitative findings")
    statistical_results: Optional[Dict[str, Any]] = Field(None, description="Statistical results")
    performance_metrics: Optional[List[ModelPerformance]] = Field(None, description="Performance metrics")
    
    # Validation and Limitations (Optional)
    validation_methods: Optional[List[str]] = Field(None, description="Validation approaches")
    limitations: Optional[List[str]] = Field(None, description="Study limitations")
    assumptions: Optional[List[str]] = Field(None, description="Key assumptions")
    
    # Impact and Applications (Optional)
    implications: Optional[List[str]] = Field(None, description="Research implications")
    applications: Optional[List[str]] = Field(None, description="Practical applications")
    societal_impact: Optional[str] = Field(None, description="Societal impact")
    future_work: Optional[List[str]] = Field(None, description="Future research directions")
    
    # Reproducibility Information (Optional)
    data_availability: Optional[str] = Field(None, description="Data availability")
    code_available: Optional[bool] = Field(None, description="Code availability")
    code_url: Optional[HttpUrl] = Field(None, description="Code repository URL")
    materials_available: Optional[bool] = Field(None, description="Materials availability")
    replication_instructions: Optional[str] = Field(None, description="Replication guide")
    
    # Domain-Specific Fields (Optional)
    # AI/ML Specific
    model_architectures: Optional[List[ModelArchitecture]] = Field(None, description="AI model architectures")
    datasets: Optional[List[Dataset]] = Field(None, description="Datasets used")
    compute_resources: Optional[List[ComputeResource]] = Field(None, description="Compute resources")
    experimental_setup: Optional[ExperimentalSetup] = Field(None, description="Experimental setup")
    
    # Clinical/Medical Specific
    ethical_approval: Optional[str] = Field(None, description="Ethics approval")
    clinical_relevance: Optional[str] = Field(None, description="Clinical relevance")
    patient_demographics: Optional[Dict[str, Any]] = Field(None, description="Patient demographics")    
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Attention Is All You Need",
                "authors": [
                    {
                        "name": "Ashish Vaswani",
                        "affiliation": "Google Research",
                        "is_corresponding": True,
                        "orcid": "0000-0002-1234-5678"
                    }
                ],
                "publication_date": "2017-12-06",
                "venue": "NeurIPS 2017",
                "doi": "10.48550/arXiv.1706.03762",
                "arxiv_id": "1706.03762",
                "fields": ["DEEP_LEARNING", "NATURAL_LANGUAGE_PROCESSING"],
                "keywords": ["attention mechanism", "transformer", "sequence-to-sequence", "neural networks"],
                "model_architectures": ["TRANSFORMER"],
                "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
                "main_contributions": [
                    "Introduction of the Transformer architecture",
                    "Self-attention mechanism",
                    "Multi-head attention"
                ],
                "novelty_claims": [
                    "First model relying entirely on attention mechanisms",
                    "Superior parallelization capabilities",
                    "State-of-the-art results on translation tasks"
                ],
                "datasets": [
                    {
                        "name": "WMT 2014 English-German",
                        "type": "TEXT",
                        "size": "4.5 GB",
                        "num_samples": 4500000,
                        "license": "CC BY-SA 4.0",
                        "preprocessing": ["tokenization", "BPE encoding"]
                    }
                ],
                "compute_resources": [
                    {
                        "hardware_type": "TPU",
                        "model": "TPU v2",
                        "quantity": 8,
                        "memory": "64 GB",
                        "provider": "Google Cloud",
                        "hours_used": 168.5,
                        "cost_estimate": 12000.0
                    }
                ],
                "experimental_setup": {
                    "batch_size": 4096,
                    "num_epochs": 100,
                    "optimizer": "Adam",
                    "learning_rate": 0.0001,
                    "loss_function": "cross_entropy",
                    "regularization": {
                        "dropout": 0.1,
                        "label_smoothing": 0.1
                    }
                },
                "performance_metrics": [
                    {
                        "metric_name": "BLEU",
                        "value": 28.4,
                        "confidence_interval": [28.1, 28.7],
                        "comparison_baseline": "GNMT",
                        "improvement": 2.0
                    }
                ],
                "ablation_studies": {
                    "num_attention_heads": {
                        "8_heads": 27.5,
                        "16_heads": 28.4,
                        "32_heads": 28.2
                    }
                },
                "limitations": [
                    "Quadratic memory complexity with sequence length",
                    "Requires large amounts of training data",
                    "High computational requirements"
                ],
                "citations": [
                    {
                        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                        "authors": ["Jacob Devlin", "Ming-Wei Chang"],
                        "year": 2019,
                        "venue": "NAACL",
                        "citations_count": 75000
                    }
                ],
                "code_available": True,
                "code_url": "https://github.com/tensorflow/tensor2tensor",
                "impact_score": 98.5,
                "environment_details": {
                    "tensorflow": "1.3.0",
                    "python": "3.6",
                    "cuda": "8.0"
                },
                "random_seeds": [42, 2017, 2018],
                "reproducibility_checklist": {
                    "code_released": True,
                    "datasets_available": True,
                    "hyperparameters_reported": True,
                    "training_hardware_specified": True
                }
            }
        }

def get_structured_output(prompt: str) -> ResearchPaper:
    """
    Extract structured information from scientific research papers using LangChain and Azure OpenAI.
    
    Args:
        prompt: Description or text from a research paper
        
    Returns:
        ResearchPaper: Comprehensive structured information about the paper
    """
    llm = ChatOpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        model="gpt-4o-mini", 
        temperature=0.3  # Lower temperature for more factual outputs
    )
    
    parser = JsonOutputParser(pydantic_object=ResearchPaper)
    
    prompt_template = """Your main task is to extract comprehensive information about the scientific research paper from the following text and output it in the defined JSON format.

    ### Paper text: 
    {user_input}
    
    ### Expected output format: 
    {format_instructions}
    """
    
    try:
        user_message = PromptTemplate(
            template=prompt_template,
            input_variables=["user_input"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = user_message | llm | parser

        with get_openai_callback() as cb:
            output = chain.invoke({"user_input": prompt})            
        
        
        logger.info(f"Successfully parsed paper")
        logger.info(f"Callback: {cb}")
        return output

    except Exception as e:
        logger.error(f"Error parsing paper data: {str(e)}")
        raise

def parse_pdf(pdf_path: str) -> str:
    """
    Parse a PDF file and return its text content.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        # Load PDF and extract text
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # Combine text from all pages
        full_text = ""
        for page in pages:
            full_text += page.page_content + "\n\n"
        
        return full_text
        
    except Exception as e:
        logger.error(f"Error parsing PDF: {e}")
        raise

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze scientific research papers from PDF files')
    parser.add_argument('pdf_paths', nargs='+', help='Paths to PDF files to analyze')
    args = parser.parse_args()
    
    try:
        # Process each PDF file
        for i, pdf_path in enumerate(args.pdf_paths):
            print(f"\n{'='*80}")
            print(f"Processing PDF {i+1}: {pdf_path}")
            print(f"{'='*80}")
            paper_name = os.path.basename(pdf_path).replace('.pdf', '')
            
            # Parse PDF content
            try:
                paper_text = parse_pdf(pdf_path)
                result = get_structured_output(paper_text)
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
                continue
                        
            output_file = f'parsed_{paper_name}_{i+1}.json'
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=4)

            print(f"\nFull analysis saved to {output_file}")
            
            print(f"\n{'='*80}\n")
            
    except Exception as e:
        logger.error(f"Error analyzing paper: {e}")
        raise

if __name__ == "__main__":
    main()
