import pandas as pd
import time
from google import genai
from google.genai import types
import json
import re
# --- ADDED: imports for safe appending / file checks
import os
import csv

# Configure the client
client = genai.Client(api_key="")

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

    # --- ADDED: Pre-compute output columns to keep order consistent for batch appends
    desired_columns = [
        'Company Name', 'Sector', 'Funding', 'Funding Citations', 'AI Enabled',
        'Category'
    ]
    other_columns = [col for col in df.columns if col != company_col]
    final_columns = desired_columns + [c for c in other_columns if c not in desired_columns]

    # --- ADDED: Helper to check if CSV exists and has a header (non-empty)
    def _file_exists_and_nonempty(path):
        return os.path.exists(path) and os.path.getsize(path) > 0

    # --- ADDED: Batch writing helper (append mode, header only if file doesn't exist or is empty)
    def _write_batch(rows):
        if not rows:
            return
        batch_df = pd.DataFrame(rows)
        # Ensure all expected columns exist; missing columns filled with N/A
        for col in final_columns:
            if col not in batch_df.columns:
                batch_df[col] = 'N/A'
        batch_df = batch_df[final_columns]

        write_header = not _file_exists_and_nonempty(output_file)
        batch_df.to_csv(output_file, index=False, mode='a', header=write_header, quoting=csv.QUOTE_MINIMAL)
        print(f"💾 Appended {len(batch_df)} rows to '{output_file}' (header written: {write_header})")

    # --- CHANGED: Buffer and batch size for periodic appends
    batch_size = 20  # append every 20 rows
    buffer_rows = []

    # Initialize list to store augmented data (we still return a DataFrame at the end)
    augmented_data = []

    # --- CHANGED: Optional resume logic (commented). Uncomment if you want to skip already written rows by Company Name.
    # existing_names = set()
    # if _file_exists_and_nonempty(output_file):
    #     try:
    #         existing_df = pd.read_csv(output_file, usecols=['Company Name'])
    #         existing_names = set(existing_df['Company Name'].dropna().astype(str).str.strip())
    #         print(f"Found {len(existing_names)} already processed companies in existing output; will skip them.")
    #     except Exception as e:
    #         print(f"Note: Could not load existing output for resume check: {e}")

    # Process each company
    for idx, row in df.iterrows():
        company_name = row[company_col]
        
        if pd.isna(company_name) or str(company_name).strip() == '':
            print(f"Skipping empty company name at row {idx}")
            continue

        # --- CHANGED: (If resume enabled) skip companies already present
        # if str(company_name).strip() in existing_names:
        #     print(f"Skipping already processed company: {company_name}")
        #     continue
        
        print(f"\nProcessing {idx + 1}/{deduplicated_count}: {company_name}")
        
        # Clean the company name for better search results
        clean_name = clean_company_name(str(company_name))
        
        # --- CHANGED: Wrap per-company processing in try/except to avoid total stop
        try:
            # Search for company information (includes funding citations)
            company_info = search_company_info(clean_name, client)
        except Exception as e:
            print(f"Error during search for '{company_name}': {e}")
            company_info = create_empty_company_info(clean_name)

        # Preserve any additional columns from the original data
        for col in df.columns:
            if col != company_col and col not in company_info:
                company_info[col] = row[col]
        
        augmented_data.append(company_info)

        # --- CHANGED: Add to buffer and append to file every 'batch_size' entries
        buffer_rows.append(company_info)
        if len(buffer_rows) >= batch_size:
            _write_batch(buffer_rows)
            buffer_rows = []  # clear buffer

        # Add delay to avoid rate limiting
        time.sleep(2)  # 2 second delay between requests
    
    # --- CHANGED: Flush any remaining rows at the end
    if buffer_rows:
        _write_batch(buffer_rows)
        buffer_rows = []

    # --- CHANGED: Return DataFrame by reading the final file (so it matches what's persisted)
    try:
        result_df = pd.read_csv(output_file)
    except Exception:
        # Fallback to in-memory if reading fails
        result_df = pd.DataFrame(augmented_data)

    print(f"\n✅ Augmented data saved (appended in batches) to: {output_file}")
    return result_df

# Example usage
if __name__ == "__main__":
    # Specify your input CSV file
    input_csv = "test.csv"  # Change this to your CSV file name
    output_csv = "augmented_companies.csv"  # Output file name
    
    # Process the companies
    result_df = process_companies_csv(input_csv, output_csv)
    
    # Display first few rows with funding citations
    print("\nFirst 5 augmented companies:")
    # --- CHANGED: Defensive check for missing columns if reading from persisted file
    cols_show = [c for c in ['Company Name', 'Funding', 'Funding Citations'] if c in result_df.columns]
    print(result_df[cols_show].head())
