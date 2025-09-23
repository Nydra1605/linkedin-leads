import pandas as pd
import time
from google import genai
from google.genai import types
import json
import re

# Configure the client
client = genai.Client(api_key="AIzaSyDpUD81GhgTihneibmypU0nsYbXr-bNWgk")

def clean_company_name(name):
    """Clean company name for better search results"""
    # Remove common suffixes that might interfere with search
    name = re.sub(r'\b(Inc\.?|LLC|Ltd\.?|Corp\.?|Corporation|Company|Co\.?)\b', '', name, flags=re.IGNORECASE)
    return name.strip()

def extract_funding_citations(response, funding_text):
    """
    Extract citations for funding information using grounding metadata.
    Returns a string with citation links.
    """
    try:
        # Check if response has grounding metadata
        if not hasattr(response, 'candidates') or not response.candidates:
            return "N/A"
        
        candidate = response.candidates[0]
        if not hasattr(candidate, 'grounding_metadata'):
            return "N/A"
        
        grounding_metadata = candidate.grounding_metadata
        
        # Get grounding supports and chunks
        supports = grounding_metadata.grounding_supports if hasattr(grounding_metadata, 'grounding_supports') else []
        chunks = grounding_metadata.grounding_chunks if hasattr(grounding_metadata, 'grounding_chunks') else []
        
        if not supports or not chunks:
            return "N/A"
        
        # Find the text segment that contains funding information
        response_text = response.text.lower()
        funding_citations = []
        
        for support in supports:
            # Get the segment text
            segment = support.segment
            start_index = segment.start_index if hasattr(segment, 'start_index') else 0
            end_index = segment.end_index if hasattr(segment, 'end_index') else len(response_text)
            
            segment_text = response_text[start_index:end_index]
            
            # Check if this segment contains funding-related keywords
            funding_keywords = ['funding', 'series', 'raised', 'investment', 'valuation', 'million', 'billion', 'round']
            if any(keyword in segment_text for keyword in funding_keywords):
                # Get the citation URLs for this segment
                if hasattr(support, 'grounding_chunk_indices') and support.grounding_chunk_indices:
                    for chunk_index in support.grounding_chunk_indices:
                        if chunk_index < len(chunks):
                            chunk = chunks[chunk_index]
                            if hasattr(chunk, 'web') and hasattr(chunk.web, 'uri'):
                                uri = chunk.web.uri
                                title = chunk.web.title if hasattr(chunk.web, 'title') else "Source"
                                # Add unique citations only
                                citation = f"{title}: {uri}"
                                if citation not in funding_citations:
                                    funding_citations.append(citation)
        
        # Return formatted citations
        if funding_citations:
            return " | ".join(funding_citations[:3])  # Limit to top 3 citations
        else:
            return "N/A"
            
    except Exception as e:
        print(f"Error extracting citations: {str(e)}")
        return "N/A"

def search_company_info(company_name, client):
    """
    Search for pharmaceutical company information using Google Search grounding.
    """
    # Define the grounding tool
    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )
    
    # Configure generation settings with grounding
    config = types.GenerateContentConfig(
        tools=[grounding_tool],
        response_modalities=["TEXT"]
    )
    
    # Create a detailed prompt to extract specific information
    prompt = f"""
    Search for information about the pharmaceutical/biotech company "{company_name}" and provide the following details:
    
    1. Sector: Categorize the company into these categories: Biotechnology, Pharmaceuticals, Diagnostics, Medical Devices, Digital Health, Research Tools, Contract Research Organization (CRO), Contract Manufacturing Organization (CMO), or Other.
    2. Funding: What type and level of funding has this company received? Include specific amounts, funding rounds (Series A, B, C, IPO), investors if available, and dates of funding rounds.
    3. AI Enabled: Is this company using AI in drug discovery or operations? Answer only "Yes" or "No"
    4. Category: Categorize the company into the following categories:
                    i. Early Drug Discovery : Hit Discovery, Hit-To-Lead, Lead Optimization, Molecule Design, Pre Clinical Discovery, Exploratory Research.
                    ii. Pre Clinical Development: In Vitro, In Vivo, ADME/Tox Profiling, Pharmacology, Pre Clinical Candidate Selection.
                    iii. Other: Anything not covered above.


    For funding information, be as specific as possible with amounts and dates.
    If any information is not available, respond with "N/A" for that field.
    Format the response as a clear list with labels.

    Example:
    Sector: Oncology
    Funding: Series B, $30M, 2022
    AI Enabled: Yes
    Category: Early Drug Discovery(Hit Discovery, Lead Optimization)
  
    """
    
    try:
        # Make the request with grounding
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config,
        )
        
        # Parse the response and extract funding citations
        company_info = parse_company_response(response.text, company_name)
        
        # Extract citations specifically for funding information
        funding_citations = extract_funding_citations(response, company_info.get('Funding', ''))
        company_info['Funding Citations'] = funding_citations
        
        return company_info
    
    except Exception as e:
        print(f"Error searching for {company_name}: {str(e)}")
        return create_empty_company_info(company_name)

def parse_company_response(response_text, company_name):
    """
    Parse the AI response to extract structured information.
    """
    info = {
        'Company Name': company_name,
        'Sector': 'N/A',
        'Funding': 'N/A',
        'AI Enabled': 'N/A',
        'Category': 'N/A'
    }
    
    # Parse the response text
    lines = response_text.split('\n')
    
    for line in lines:
        line_lower = line.lower()
        
        if 'sector:' in line_lower:
            info['Sector'] = extract_value(line)
        elif 'funding:' in line_lower:
            info['Funding'] = extract_value(line)
        elif 'ai enabled:' in line_lower or 'ai-enabled:' in line_lower:
            value = extract_value(line).lower()
            info['AI Enabled'] = 'Yes' if 'yes' in value else 'No' if 'no' in value else 'N/A'
        elif 'category:' in line_lower:
            info['Category'] = extract_value(line)
    return info

def extract_value(line):
    """Extract value from a line containing 'label: value' format"""
    parts = line.split(':', 1)
    if len(parts) > 1:
        value = parts[1].strip()
        # Clean up common formatting
        value = value.replace('*', '').strip()
        return value if value else 'N/A'
    return 'N/A'

def extract_linkedin_url(line):
    """Extract LinkedIn URL from a line"""
    # Look for LinkedIn URL pattern
    url_pattern = r'(https?://(?:www\.)?linkedin\.com/in/[^\s\)]+)'
    match = re.search(url_pattern, line)
    if match:
        return match.group(1)
    
    # If no URL found, return the extracted value
    value = extract_value(line)
    if 'linkedin.com' in value.lower():
        return value
    return 'N/A'

def create_empty_company_info(company_name):
    """Create an empty info dictionary for a company"""
    return {
        'Company Name': company_name,
        'Sector': 'N/A',
        'Funding': 'N/A',
        'Funding Citations': 'N/A',
        'AI Enabled': 'N/A',
        'Category': 'N/A'
    }

def process_companies_csv(input_file, output_file='augmented_companies.csv'):
    """
    Main function to process the CSV file with company names.
    """
    print(f"Loading CSV file: {input_file}")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Identify the company name column (assuming it's the first column or named 'Company')
    if 'Lead_Company' in df.columns:
        company_col = 'Lead_Company'
    else:
        # Use the first column if no standard name is found
        company_col = df.columns[0]
    
    print(f"Using column '{company_col}' as company name column")
    
    # Remove duplicates based on company name
    original_count = len(df)
    df = df.drop_duplicates(subset=[company_col], keep='first')
    deduplicated_count = len(df)
    
    print(f"Removed {original_count - deduplicated_count} duplicate companies")
    print(f"Processing {deduplicated_count} unique companies...")
    
    # Initialize list to store augmented data
    augmented_data = []
    
    # Process each company
    for idx, row in df.iterrows():
        company_name = row[company_col]
        
        if pd.isna(company_name) or str(company_name).strip() == '':
            print(f"Skipping empty company name at row {idx}")
            continue
        
        print(f"\nProcessing {idx + 1}/{deduplicated_count}: {company_name}")
        
        # Clean the company name for better search results
        clean_name = clean_company_name(str(company_name))
        
        # Search for company information (includes funding citations)
        company_info = search_company_info(clean_name, client)
        
        # Preserve any additional columns from the original data
        for col in df.columns:
            if col != company_col and col not in company_info:
                company_info[col] = row[col]
        
        augmented_data.append(company_info)
        
        # Add delay to avoid rate limiting
        time.sleep(2)  # 2 second delay between requests
    
    # Create DataFrame from augmented data
    augmented_df = pd.DataFrame(augmented_data)
    
    # Reorder columns to put the augmented fields first (including Funding Citations)
    desired_columns = [
        'Company Name', 'Sector', 'Funding', 'Funding Citations', 'AI Enabled',
        'Category'
    ]
    
    # Add any additional columns from original data
    other_columns = [col for col in augmented_df.columns if col not in desired_columns]
    final_columns = desired_columns + other_columns
    
    augmented_df = augmented_df[final_columns]

    # augmented_df = augmented_df[["AI Enabled"] == 'No']
    
    # Save to CSV
    augmented_df.to_csv(output_file, index=False)
    print(f"\n✅ Augmented data saved to: {output_file}")
    
    # Print summary statistics

    
    return augmented_df

# Example usage
if __name__ == "__main__":
    # Specify your input CSV file
    input_csv = "test.csv"  # Change this to your CSV file name
    output_csv = "augmented_pharma_companies.csv"  # Output file name
    
    # Process the companies
    result_df = process_companies_csv(input_csv, output_csv)
    
    # Display first few rows with funding citations
    print("\nFirst 5 augmented companies:")
    print(result_df[['Company Name', 'Funding', 'Funding Citations']].head())