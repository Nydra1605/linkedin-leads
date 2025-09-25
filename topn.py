"""
Company Prioritization Agent using Azure OpenAI
This agent prioritizes pharmaceutical companies based on:
1. Biotechnology sector preference
2. Lack of AI capabilities 
3. Focus on Early drug discovery (primary) or Pre-clinical development (secondary)
"""

import pandas as pd
import json
from typing import List, Dict, Optional
from openai import AzureOpenAI
import os
from datetime import datetime

# Azure OpenAI Configuration
class CompanyPrioritizationAgent:
    def __init__(self, api_key: str, endpoint: str, deployment_name: str, api_version: str = "2024-02-01"):
        """
        Initialize the agent with Azure OpenAI credentials
        
        Args:
            api_key: Azure OpenAI API key
            endpoint: Azure OpenAI endpoint URL
            deployment_name: Name of your deployed model (e.g., 'gpt-4')
            api_version: API version (default: "2024-02-01")
        """
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        self.deployment_name = deployment_name
        
    def create_system_prompt(self) -> str:
        """
        Creates a comprehensive system prompt with prompt engineering best practices
        """
        return """You are an expert pharmaceutical company analyst specializing in strategic investment prioritization. Your task is to evaluate and score companies based on specific criteria to identify the most promising investment opportunities.

## SCORING FRAMEWORK

You will use the following quantitative scoring system to evaluate each company:

### 1. SECTOR SCORE (0-40 points)
- **Biotechnology** (contains "Biotechnology" in Sector field): 40 points
- **Pharmaceuticals** (contains "Pharmaceuticals" but not "Biotechnology"): 20 points
- **Medical Devices**: 10 points
- **Other/Unknown**: 0 points

### 2. AI CAPABILITY SCORE (0-30 points)
- **No AI capabilities** (AI Enabled = "No"): 30 points
- **Unknown/N/A AI status**: 15 points
- **Has AI capabilities** (AI Enabled = "Yes"): 0 points

### 3. CATEGORY SCORE (0-30 points)
- **Early Drug Discovery** (contains keywords: "Early Drug Discovery", "Hit Discovery", "Lead Optimization", "Target Identification"): 30 points
- **Pre Clinical Development** (contains keywords: "Pre Clinical", "In Vitro", "In Vivo", "ADME", "Tox"): 20 points
- **Clinical Development** (contains "Clinical" but not "Pre Clinical"): 10 points
- **Other/Unknown**: 5 points

### TOTAL SCORE = SECTOR SCORE + AI CAPABILITY SCORE + CATEGORY SCORE (Maximum: 100 points)

## EVALUATION INSTRUCTIONS

1. **Parse each field carefully**: Look for partial matches and keywords, not just exact matches
2. **Handle missing data**: When data is "N/A" or missing, use the middle score for that category
3. **Be consistent**: Apply the same logic across all companies
4. **Ignore funding information**: Focus only on sector, AI capabilities, and category

## OUTPUT FORMAT

For each company, provide your analysis in the following JSON structure:
{
    "company_name": "Company Name",
    "sector_score": <number>,
    "sector_reason": "Brief explanation",
    "ai_score": <number>,
    "ai_reason": "Brief explanation", 
    "category_score": <number>,
    "category_reason": "Brief explanation",
    "total_score": <number>,
    "priority_rank": <number>
}

## ANALYSIS APPROACH

1. Read the company information carefully
2. Identify keywords and patterns in each field
3. Apply the scoring rubric systematically
4. Calculate the total score
5. Provide clear reasoning for each score

Remember: The goal is to identify companies that are:
- In the biotechnology sector (highest priority)
- Without existing AI capabilities (opportunity for AI integration)
- Focused on early-stage drug discovery (highest innovation potential)

Be analytical, precise, and consistent in your scoring."""

    def create_user_prompt(self, companies_data: pd.DataFrame, n_companies: int = 10) -> str:
        """
        Creates the user prompt with company data
        
        Args:
            companies_data: DataFrame with company information
            n_companies: Number of top companies to return
        """
        companies_json = companies_data.to_json(orient='records', indent=2)
        
        return f"""Please analyze the following {len(companies_data)} companies and identify the top {n_companies} companies based on the scoring criteria.

Here is the company data:

{companies_json}

Please:
1. Score each company according to the framework
2. Rank them by total score (highest to lowest)
3. Return the top {n_companies} companies with their complete scoring analysis
4. Provide a summary of why these companies were selected

Output the results as a JSON array with the top {n_companies} companies."""

    def score_companies_locally(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Local scoring function for validation and comparison
        
        Args:
            df: DataFrame with company data
            
        Returns:
            DataFrame with scores added
        """
        def calculate_sector_score(sector):
            if pd.isna(sector):
                return 0
            sector_lower = str(sector).lower()
            if 'biotechnology' in sector_lower:
                return 40
            elif 'pharmaceutical' in sector_lower:
                return 20
            elif 'medical device' in sector_lower:
                return 10
            else:
                return 0
        
        def calculate_ai_score(ai_enabled):
            if pd.isna(ai_enabled):
                return 15
            ai_lower = str(ai_enabled).lower()
            if 'no' in ai_lower:
                return 30
            elif 'yes' in ai_lower:
                return 0
            else:
                return 15
        
        def calculate_category_score(category):
            if pd.isna(category):
                return 5
            category_lower = str(category).lower()
            if any(term in category_lower for term in ['early drug discovery', 'hit discovery', 
                                                         'lead optimization', 'target identification']):
                return 30
            elif any(term in category_lower for term in ['pre clinical', 'pre-clinical', 'in vitro', 
                                                          'in vivo', 'adme', 'tox']):
                return 20
            elif 'clinical' in category_lower:
                return 10
            else:
                return 5
        
        # Calculate scores
        df['sector_score'] = df['Sector'].apply(calculate_sector_score)
        df['ai_score'] = df['AI Enabled'].apply(calculate_ai_score)
        df['category_score'] = df['Category'].apply(calculate_category_score)
        df['total_score'] = df['sector_score'] + df['ai_score'] + df['category_score']
        
        # Sort by total score
        df = df.sort_values('total_score', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        
        return df

    def prioritize_companies(self, csv_path: str, n_companies: int = 10, 
                            use_local_scoring: bool = False) -> Dict:
        """
        Main function to prioritize companies
        
        Args:
            csv_path: Path to the CSV file
            n_companies: Number of top companies to return
            use_local_scoring: If True, use local scoring instead of LLM
            
        Returns:
            Dictionary with prioritized companies and analysis
        """
        # Load data
        df = pd.read_csv(csv_path)
        
        if use_local_scoring:
            # Use local scoring for faster results or testing
            scored_df = self.score_companies_locally(df)
            top_companies = scored_df.head(n_companies)
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'total_companies_analyzed': len(df),
                'top_companies': top_companies[['Company Name', 'Sector', 'AI Enabled', 
                                               'Category', 'total_score', 'rank']].to_dict('records'),
                'scoring_method': 'local',
                'summary': f"Top {n_companies} companies identified using rule-based scoring."
            }
        else:
            # Use Azure OpenAI for scoring
            system_prompt = self.create_system_prompt()
            user_prompt = self.create_user_prompt(df, n_companies)
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent scoring
                max_tokens=4000,
                response_format={"type": "json_object"}  # Ensure JSON output
            )
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'total_companies_analyzed': len(df),
                'llm_response': json.loads(response.choices[0].message.content),
                'scoring_method': 'azure_openai',
                'model_used': self.deployment_name
            }
        
        return results

    def generate_report(self, results: Dict, output_path: Optional[str] = None) -> str:
        """
        Generate a formatted report from the results
        
        Args:
            results: Results dictionary from prioritize_companies
            output_path: Optional path to save the report
            
        Returns:
            Formatted report string
        """
        report = f"""
COMPANY PRIORITIZATION REPORT
Generated: {results['timestamp']}
=====================================

Total Companies Analyzed: {results['total_companies_analyzed']}
Scoring Method: {results['scoring_method']}

TOP PRIORITIZED COMPANIES
=====================================
"""
        
        if results['scoring_method'] == 'local':
            for i, company in enumerate(results['top_companies'], 1):
                report += f"""
{i}. {company['Company Name']}
   - Sector: {company['Sector']}
   - AI Enabled: {company['AI Enabled']}
   - Category: {company['Category']}
   - Total Score: {company['total_score']:.1f}
   - Rank: {company['rank']}
"""
        else:
            # Parse LLM response
            report += str(results['llm_response'])
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
        
        return report


def main():
    """
    Example usage of the Company Prioritization Agent
    """
    # Configure Azure OpenAI credentials
    # Replace these with your actual Azure OpenAI credentials
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "your-api-key-here")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com/")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
    
    # Initialize the agent
    agent = CompanyPrioritizationAgent(
        api_key=AZURE_OPENAI_API_KEY,
        endpoint=AZURE_OPENAI_ENDPOINT,
        deployment_name=AZURE_OPENAI_DEPLOYMENT
    )
    
    # Example 1: Use local scoring (no API calls, faster)
    print("Running local scoring...")
    results_local = agent.prioritize_companies(
        csv_path="augmented_companies.csv",
        n_companies=10,
        use_local_scoring=True
    )
    
    # Generate and print report
    report = agent.generate_report(results_local, output_path="prioritization_report_local.txt")
    print(report[:1000])  # Print first 1000 characters
    
    # Example 2: Use Azure OpenAI for scoring (requires valid credentials)
    if AZURE_OPENAI_API_KEY != "your-api-key-here":
        print("\nRunning Azure OpenAI scoring...")
        results_llm = agent.prioritize_companies(
            csv_path="augmented_companies.csv",
            n_companies=10,
            use_local_scoring=False
        )
        
        # Generate and save report
        report_llm = agent.generate_report(results_llm, output_path="prioritization_report_llm.txt")
        print("\nLLM-based prioritization completed!")
    
    # Example 3: Get top 5 companies with detailed scoring breakdown
    df = pd.read_csv("augmented_companies.csv")
    scored_df = agent.score_companies_locally(df)
    
    print("\n" + "="*80)
    print("TOP 5 COMPANIES - DETAILED SCORING BREAKDOWN")
    print("="*80)
    
    for idx, row in scored_df.head(5).iterrows():
        print(f"\n{row['rank']}. {row['Company Name']}")
        print(f"   Sector Score: {row['sector_score']}/40 - {row['Sector'][:50]}...")
        print(f"   AI Score: {row['ai_score']}/30 - AI Enabled: {row['AI Enabled']}")
        print(f"   Category Score: {row['category_score']}/30 - {row['Category'][:50]}...")
        print(f"   TOTAL SCORE: {row['total_score']}/100")
    
    # Save scored DataFrame to CSV for further analysis
    scored_df.to_csv("companies_with_scores.csv", index=False)
    print("\n✅ Scored companies saved to 'companies_with_scores.csv'")


if __name__ == "__main__":
    main()