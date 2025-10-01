"""
Company Data Augmentation AI Agent
This agent enriches company data with sector, funding, AI capabilities, and category information
using Azure OpenAI GPT-4o and Serp API through LangChain/LangGraph
"""

import os
import pandas as pd
import asyncio
from typing import Dict, List, Optional, TypedDict, Annotated
from datetime import datetime
import json
import time
from enum import Enum
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseMessage
from langchain.tools import Tool
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Configuration
class Config:
    """Configuration settings for the agent"""
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "your-api-key")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com/")
    AZURE_OPENAI_API_VERSION = "2023-07-01-preview"
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "your-serpapi-key")
    
    # Rate limiting
    REQUESTS_PER_MINUTE = 60
    BATCH_SIZE = 5
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 2

# Define Enums for standardized values
class Sector(str, Enum):
    BIOTECHNOLOGY = "Biotechnology"
    PHARMACEUTICALS = "Pharmaceuticals"
    OTHER = "Other"

class FundingStage(str, Enum):
    PRE_SEED = "Pre-Seed"
    SEED = "Seed"
    SERIES_A = "Series A"
    SERIES_B = "Series B"
    SERIES_C = "Series C"
    SERIES_D = "Series D"
    SERIES_E_PLUS = "Series E+"
    PRE_IPO = "Pre-IPO"
    IPO = "IPO"
    PUBLIC = "Public"
    ACQUIRED = "Acquired"
    UNKNOWN = "Unknown"

class AIEnabled(str, Enum):
    YES = "Yes"
    NO = "No"
    UNKNOWN = "Unknown"

class Category(str, Enum):
    EARLY_DRUG_DISCOVERY = "Early Drug Discovery"
    PRE_CLINICAL_DEVELOPMENT = "Pre Clinical Development"
    CLINICAL_TRIALS = "Clinical Trials"
    COMMERCIAL = "Commercial"
    PLATFORM_TECHNOLOGY = "Platform Technology"
    DIAGNOSTICS = "Diagnostics"
    MEDICAL_DEVICES = "Medical Devices"
    OTHER = "Other"

# Define the state for LangGraph
class CompanyState(TypedDict):
    """State definition for company augmentation workflow"""
    company_name: str
    search_results: Optional[str]
    sector: Optional[str]
    funding: Optional[str]
    ai_enabled: Optional[str]
    category: Optional[str]
    confidence_scores: Dict[str, float]
    error: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages]

# Company research tools
class CompanySearchTool:
    """Tool for searching company information"""
    
    def __init__(self, serpapi_key: str):
        self.search = SerpAPIWrapper(serpapi_api_key=serpapi_key)
    
    def search_company(self, company_name: str) -> str:
        """Search for comprehensive company information"""
        try:
            # Search for company information
            query = f"{company_name} biotechnology pharmaceutical funding AI drug discovery clinical trials"
            results = self.search.run(query)
            
            # Additional targeted search for funding information
            funding_query = f"{company_name} series funding round venture capital IPO"
            funding_results = self.search.run(funding_query)
            
            combined_results = f"General Info:\n{results}\n\nFunding Info:\n{funding_results}"
            return combined_results
        except Exception as e:
            return f"Search error: {str(e)}"

# Initialize Azure OpenAI
def create_llm():
    """Create and configure Azure OpenAI LLM"""
    return AzureChatOpenAI(
        azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        api_key=Config.AZURE_OPENAI_API_KEY,
        api_version=Config.AZURE_OPENAI_API_VERSION,
        deployment_name=Config.AZURE_OPENAI_DEPLOYMENT,
        temperature=0.1,
        max_tokens=1000
    )

# Prompts for information extraction
EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""
You are an expert analyst specializing in biotechnology and pharmaceutical companies.
Based on the search results provided, extract and classify the following information about the company.
Be precise and base your answers only on the information provided.

For each field, also provide a confidence score (0.0 to 1.0) based on how certain you are about the classification.

Fields to extract:
1. Sector: Classify as "Biotechnology", "Pharmaceuticals", or "Other"
2. Funding: Identify the latest funding stage (Series A, B, C, D, E+, Pre-IPO, IPO, Public, etc.)
3. AI Enabled: Determine if the company uses AI in their operations (Yes/No/Unknown)
4. Category: Classify the primary focus area:
   - Early Drug Discovery
   - Pre Clinical Development
   - Clinical Trials
   - Commercial
   - Platform Technology
   - Diagnostics
   - Medical Devices
   - Other

Return the results in JSON format:
{{
    "sector": "value",
    "funding": "value",
    "ai_enabled": "value",
    "category": "value",
    "confidence_scores": {{
        "sector": 0.0-1.0,
        "funding": 0.0-1.0,
        "ai_enabled": 0.0-1.0,
        "category": 0.0-1.0
    }},
    "reasoning": "brief explanation for classifications"
}}
    """),
    HumanMessagePromptTemplate.from_template("""
Company Name: {company_name}

Search Results:
{search_results}

Please analyze and extract the required information.
    """)
])

class CompanyAugmentationAgent:
    """Main agent for augmenting company data"""
    
    def __init__(self):
        self.llm = create_llm()
        self.search_tool = CompanySearchTool(Config.SERPAPI_API_KEY)
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(CompanyState)
        
        # Add nodes
        workflow.add_node("search", self._search_node)
        workflow.add_node("extract", self._extract_node)
        workflow.add_node("validate", self._validate_node)
        
        # Add edges
        workflow.add_edge("search", "extract")
        workflow.add_edge("extract", "validate")
        workflow.add_edge("validate", END)
        
        # Set entry point
        workflow.set_entry_point("search")
        
        # Compile with memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    async def _search_node(self, state: CompanyState) -> CompanyState:
        """Search for company information"""
        try:
            search_results = self.search_tool.search_company(state["company_name"])
            state["search_results"] = search_results
        except Exception as e:
            state["error"] = f"Search failed: {str(e)}"
        return state
    
    async def _extract_node(self, state: CompanyState) -> CompanyState:
        """Extract structured information from search results"""
        if state.get("error"):
            return state
        
        try:
            # Prepare the prompt
            messages = EXTRACTION_PROMPT.format_messages(
                company_name=state["company_name"],
                search_results=state["search_results"]
            )
            
            # Get LLM response
            response = await self.llm.ainvoke(messages)
            
            # Parse JSON response
            try:
                extracted_data = json.loads(response.content)
                state["sector"] = extracted_data.get("sector", "Other")
                state["funding"] = extracted_data.get("funding", "Unknown")
                state["ai_enabled"] = extracted_data.get("ai_enabled", "Unknown")
                state["category"] = extracted_data.get("category", "Other")
                state["confidence_scores"] = extracted_data.get("confidence_scores", {})
            except json.JSONDecodeError:
                state["error"] = "Failed to parse extraction results"
                
        except Exception as e:
            state["error"] = f"Extraction failed: {str(e)}"
        
        return state
    
    async def _validate_node(self, state: CompanyState) -> CompanyState:
        """Validate and standardize extracted information"""
        if state.get("error"):
            return state
        
        # Validate sector
        if state["sector"] not in [s.value for s in Sector]:
            state["sector"] = Sector.OTHER.value
        
        # Validate funding stage
        valid_funding = [f.value for f in FundingStage]
        if state["funding"] not in valid_funding:
            # Try to map common variations
            funding_map = {
                "series a": FundingStage.SERIES_A.value,
                "series b": FundingStage.SERIES_B.value,
                "series c": FundingStage.SERIES_C.value,
                "series d": FundingStage.SERIES_D.value,
                "ipo": FundingStage.IPO.value,
                "public": FundingStage.PUBLIC.value,
            }
            state["funding"] = funding_map.get(
                state["funding"].lower(), 
                FundingStage.UNKNOWN.value
            )
        
        # Validate AI enabled
        if state["ai_enabled"] not in [a.value for a in AIEnabled]:
            state["ai_enabled"] = AIEnabled.UNKNOWN.value
        
        # Validate category
        if state["category"] not in [c.value for c in Category]:
            state["category"] = Category.OTHER.value
        
        return state
    
    async def process_company(self, company_name: str) -> Dict:
        """Process a single company"""
        initial_state = CompanyState(
            company_name=company_name,
            search_results=None,
            sector=None,
            funding=None,
            ai_enabled=None,
            category=None,
            confidence_scores={},
            error=None,
            messages=[]
        )
        
        # Run the workflow
        config = {"configurable": {"thread_id": company_name}}
        result = await self.workflow.ainvoke(initial_state, config)
        
        return {
            "Lead_Company": company_name,
            "Sector": result.get("sector", "Other"),
            "Funding": result.get("funding", "Unknown"),
            "AI_Enabled": result.get("ai_enabled", "Unknown"),
            "Category": result.get("category", "Other"),
            "Confidence_Scores": result.get("confidence_scores", {}),
            "Error": result.get("error")
        }
    
    async def process_batch(self, companies: List[str], batch_size: int = 5) -> List[Dict]:
        """Process multiple companies in batches"""
        results = []
        
        for i in range(0, len(companies), batch_size):
            batch = companies[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1} ({len(batch)} companies)...")
            
            # Process batch concurrently
            batch_tasks = [self.process_company(company) for company in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            
            # Rate limiting
            if i + batch_size < len(companies):
                await asyncio.sleep(60 / Config.REQUESTS_PER_MINUTE * batch_size)
        
        return results

# Main execution function
async def augment_company_data(input_csv_path: str, output_csv_path: str):
    """Main function to augment company data from CSV"""
    
    print("Initializing Company Augmentation Agent...")
    agent = CompanyAugmentationAgent()
    
    # Read input CSV
    print(f"Reading companies from {input_csv_path}...")
    df = pd.read_csv(input_csv_path)
    companies = df['Lead_Company'].tolist()
    
    print(f"Found {len(companies)} companies to process")
    
    # Process companies
    print("Starting augmentation process...")
    start_time = time.time()
    
    results = await agent.process_batch(companies, batch_size=Config.BATCH_SIZE)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(output_csv_path, index=False)
    
    elapsed_time = time.time() - start_time
    print(f"\nAugmentation complete!")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"Results saved to {output_csv_path}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"- Successfully processed: {len(results_df[results_df['Error'].isna()])}")
    print(f"- Errors: {len(results_df[results_df['Error'].notna()])}")
    
    if 'Confidence_Scores' in results_df.columns:
        avg_confidence = {}
        for _, row in results_df.iterrows():
            if row['Confidence_Scores'] and isinstance(row['Confidence_Scores'], dict):
                for key, value in row['Confidence_Scores'].items():
                    if key not in avg_confidence:
                        avg_confidence[key] = []
                    avg_confidence[key].append(value)
        
        print("\nAverage Confidence Scores:")
        for key, values in avg_confidence.items():
            print(f"- {key}: {sum(values)/len(values):.2f}")
    
    return results_df

# Utility function for single company lookup
async def lookup_single_company(company_name: str):
    """Quick lookup for a single company"""
    agent = CompanyAugmentationAgent()
    result = await agent.process_company(company_name)
    return result

# Example usage
if __name__ == "__main__":
    # Set up environment variables
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = "2023-07-01-preview"
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "your-serpapi-key")
    
    # Run the augmentation
    asyncio.run(augment_company_data("serp_test.csv", "serp_augmented.csv"))
    
    # Or lookup a single company
    # result = asyncio.run(lookup_single_company("2seventy bio inc"))
    # print(json.dumps(result, indent=2))