import requests
import json
import csv
import pandas as pd
from time import sleep
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import AzureOpenAI
import os
load_dotenv()

# Configuration
API_KEY = os.getenv("SERPHOUSE_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")

def search_company(company_name):
    """Performs a search for a given company name using Serphouse API."""
    print(f"\n🔍 Searching for: {company_name}")
    
    payload = {
        "data": {
            "q": company_name,
            "domain": "google.com",
            "loc": "United States",
            "lang": "en",
            "device": "desktop",
            "serp_type": "web",
        }
    }
    
    url = "https://api.serphouse.com/serp/live"
    headers = {
        'accept': 'application/json',
        'content-type': 'application/json',
        'authorization': f"Bearer {API_KEY}"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()
        
        # Access the nested 'results' dictionary
        search_results = response_json.get('results', {}).get('results', {})
        
        # Extract from "top_stories"
        top_stories_data = []
        if 'top_stories' in search_results:
            for story in search_results['top_stories']:
                top_stories_data.append({
                    'title': story.get('title'),
                    'link': story.get('link')
                })
        
        # Extract from "organic" search results
        organic_data = []
        if 'organic' in search_results:
            for result in search_results['organic']:
                organic_data.append({
                    'title': result.get('title'),
                    'link': result.get('link'),
                    'snippet': result.get('snippet')
                })
        
        return organic_data, top_stories_data
        
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred for {company_name}: {http_err}")
        return [], []
    except Exception as err:
        print(f"An error occurred for {company_name}: {err}")
        return [], []

def scrape_url(url: str) -> str | None:
    """Scrapes the text content from a given URL."""
    print(f"  Scraping: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        return main_content.get_text(separator=' ', strip=True) if main_content else None
    except requests.RequestException as e:
        print(f"    Error scraping {url}: {e}")
        return None

def extract_company_info_with_llm(context: str, source_urls: list, company_name: str) -> dict | None:
    """Sends combined text to Azure OpenAI to extract structured company data."""
    print(f"\n💡 Analyzing combined text with Azure OpenAI for {company_name}...")
    if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_DEPLOYMENT_NAME]):
        print("❌ Error: Azure OpenAI environment variables are not set.")
        return None

    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version="2023-07-01-preview"
    )

    prompt = f"""
    Based on the provided text, which was scraped from the URLs listed below, analyze the company "{company_name}" and extract the following information.
    Format your response as a single, valid JSON object only, with no other text or explanations.

    Source URLs:
    {', '.join(source_urls)}

    Data Points to Extract:
    1. "sector": Categorize the companies into one of the following sectors: "Biotechnology", "Pharmaceuticals" or "Others".
    2. "funding": Provide the Funding Amount in USD, convert if required. Also provide the Funding Stage from the following options: "Seed", "Series A", "Series B", "Series C", "Series D", "Series E", "Series F", "IPO", "Acquired". The output should be in the following format: $10M, Series A. If no funding information is found, use "Not Found".
    3. "funding_citation": The exact URL from the source list that contains the funding information. If not found, use "Not Found".
    4. "ai_enabled": If the company is using AI in any capacity, return "Yes". If not, return "No". 

    --- START OF PROVIDED TEXT ---
    {context[:24000]}
    --- END OF PROVIDED TEXT ---

    Provide your response as a JSON object.
    """

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a precise pharmaceutical company data scraper that extracts data and returns it in JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        json_response = response.choices[0].message.content
        return json.loads(json_response)
    except Exception as e:
        print(f"❌ Error during LLM analysis for {company_name}: {e}")
        return None

def save_to_csv(data: dict, company_name: str, filename: str = "job5_output.csv"):
    """Appends the extracted company data to a CSV file."""
    fieldnames = ['company_name', 'sector', 'funding', 'funding_citation', 'ai_enabled']
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'company_name': company_name, **data})
    print(f"✅ Data for {company_name} appended successfully to {filename}")

def get_processed_companies(output_csv_file: str) -> set:
    """
    Reads the output CSV file and returns a set of already processed company names.
    
    Args:
        output_csv_file: Path to the output CSV file
        
    Returns:
        Set of company names that have already been processed
    """
    if not os.path.exists(output_csv_file):
        return set()
    
    try:
        df = pd.read_csv(output_csv_file)
        if 'company_name' in df.columns:
            processed = set(df['company_name'].dropna().unique())
            print(f"📋 Found {len(processed)} already processed companies in output file")
            return processed
        return set()
    except Exception as e:
        print(f"⚠️ Warning: Could not read existing output file: {e}")
        return set()

def process_company_data(company_name: str):
    """
    Main function to process search results for a single company.
    It searches, scrapes all links, analyzes the combined text, and saves the result.
    """
    print(f"\n{'='*50}\nProcessing analysis for: {company_name}\n{'='*50}")
    
    # 1. Search for the company
    organic_results, top_stories_results = search_company(company_name)
    
    # 2. Combine all sources and get a unique list of links
    all_sources = organic_results + top_stories_results
    unique_urls = list(set(item['link'] for item in all_sources if item.get('link')))

    if not unique_urls:
        print(f"No URLs found to process for {company_name}.")
        # Save a record with "Not Found" values
        default_data = {
            'sector': 'Not Found',
            'funding': 'Not Found',
            'funding_citation': 'Not Found',
            'ai_enabled': 'Unknown'
        }
        save_to_csv(default_data, company_name)
        return

    # 3. Scrape all unique links and combine their content
    scraped_content = []
    for url in unique_urls[:10]:  # Limit to first 10 URLs to avoid timeout
        content = scrape_url(url)
        if content:
            scraped_content.append(f"--- CONTENT FROM {url} ---\n{content}")

    full_context = "\n\n".join(scraped_content)

    if not full_context:
        print(f"Could not scrape any content from the provided URLs for {company_name}.")
        # Save a record with "Not Found" values
        default_data = {
            'sector': 'Not Found',
            'funding': 'Not Found',
            'funding_citation': 'Not Found',
            'ai_enabled': 'Unknown'
        }
        save_to_csv(default_data, company_name)
        return

    # 4. Analyze the combined content with the LLM
    extracted_data = extract_company_info_with_llm(full_context, unique_urls, company_name)

    if extracted_data:
        # 5. Save the final, consolidated results to a CSV file
        save_to_csv(extracted_data, company_name)
    else:
        print(f"❌ Failed to extract data for {company_name}.")
        # Save a record with "Not Found" values
        default_data = {
            'sector': 'Not Found',
            'funding': 'Not Found',
            'funding_citation': 'Not Found',
            'ai_enabled': 'Unknown'
        }
        save_to_csv(default_data, company_name)

def process_companies_from_csv(input_csv_file: str, output_csv_file: str = "job5_output.csv"):
    """
    Reads company names from a CSV file and processes each one.
    Automatically resumes from where it left off if the script was interrupted.
    
    Args:
        input_csv_file: Path to CSV file containing company names in 'Lead_Company' column
        output_csv_file: Path to output CSV file for augmented data
    """
    try:
        # Read the input CSV file
        df1 = pd.read_csv(input_csv_file)
        df = df1.drop_duplicates(subset=['Lead_Company'])
        
        if 'Lead_Company' not in df.columns:
            print(f"❌ Error: 'Lead_Company' column not found in {input_csv_file}")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Get unique company names
        all_companies = df['Lead_Company'].dropna().unique()
        
        # Get already processed companies
        processed_companies = get_processed_companies(output_csv_file)
        
        # Filter out already processed companies
        companies_to_process = [c for c in all_companies if c not in processed_companies]
        
        total_companies = len(all_companies)
        remaining_companies = len(companies_to_process)
        completed_companies = len(processed_companies)
        
        print(f"\n🚀 Processing Summary:")
        print(f"   Total companies in input: {total_companies}")
        print(f"   Already processed: {completed_companies}")
        print(f"   Remaining to process: {remaining_companies}")
        print(f"   Output file: {output_csv_file}")
        
        if remaining_companies == 0:
            print("\n✨ All companies have already been processed!")
            return
        
        # Process each remaining company
        for idx, company in enumerate(companies_to_process, 1):
            overall_progress = completed_companies + idx
            print(f"\n📊 Progress: {overall_progress}/{total_companies} (Processing {idx}/{remaining_companies} remaining)")
            
            try:
                process_company_data(company)
            except Exception as e:
                print(f"❌ Error processing {company}: {e}")
                # Save error entry to maintain progress tracking
                error_data = {
                    'sector': 'Error',
                    'funding': 'Error',
                    'funding_citation': 'Error',
                    'ai_enabled': 'Error'
                }
                save_to_csv(error_data, company, output_csv_file)
            
            # Add a small delay to avoid rate limiting
            if idx < remaining_companies:
                sleep(2)  # 2 second delay between companies
        
        print(f"\n✨ Processing complete! Results saved to {output_csv_file}")
        
        # Display summary
        if os.path.exists(output_csv_file):
            results_df = pd.read_csv(output_csv_file)
            print(f"\n📈 Final Summary:")
            print(f"   Total companies processed: {len(results_df)}")
            print(f"   Companies with funding data: {(results_df['funding'] != 'Not Found').sum()}")
            print(f"   AI-enabled companies: {(results_df['ai_enabled'] == 'Yes').sum()}")
            print(f"   Errors encountered: {(results_df['sector'] == 'Error').sum()}")
            
    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_csv_file}' not found.")
    except Exception as e:
        print(f"❌ Error processing CSV file: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Specify your input CSV file path
    INPUT_CSV_FILE = "job5.csv"  # Change this to your actual file path
    OUTPUT_CSV_FILE = "job5_output.csv"  # Output file name
    
    # Process all companies from the CSV
    process_companies_from_csv(INPUT_CSV_FILE, OUTPUT_CSV_FILE)