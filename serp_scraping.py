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
    1. "sector": A 1-2 word industry sector (e.g., 'Artificial Intelligence', 'Cloud Computing').
    2. "funding": The latest funding amount or stage (e.g., '$500M', 'Series C'). If not found, use "Not Found".
    3. "funding_citation": The exact URL from the source list that contains the funding information. If not found, use "Not Found".
    4. "ai_enabled": Answer "Yes" if the company's core product is AI-based, otherwise "No".

    --- START OF PROVIDED TEXT ---
    {context[:24000]}
    --- END OF PROVIDED TEXT ---

    Provide your response as a JSON object.
    """

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a precise financial and tech analyst that extracts data and returns it in JSON format."},
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

def save_to_csv(data: dict, company_name: str, filename: str = "serp_output.csv"):
    """Appends the extracted company data to a CSV file."""
    fieldnames = ['company_name', 'sector', 'funding', 'funding_citation', 'ai_enabled']
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'company_name': company_name, **data})
    print(f"✅ Data for {company_name} appended successfully to {filename}")

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

def process_companies_from_csv(input_csv_file: str, output_csv_file: str = "serp_output.csv"):
    """
    Reads company names from a CSV file and processes each one.
    
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
        companies = df['Lead_Company'].dropna().unique()
        total_companies = len(companies)
        
        print(f"\n🚀 Found {total_companies} unique companies to process")
        print(f"Output will be saved to: {output_csv_file}")
        
        # Clear the output file if it exists (optional - remove if you want to append)
        if os.path.exists(output_csv_file):
            os.remove(output_csv_file)
            print(f"Cleared existing output file: {output_csv_file}")
        
        # Process each company
        for idx, company in enumerate(companies, 1):
            print(f"\n📊 Progress: {idx}/{total_companies}")
            process_company_data(company)
            
            # Add a small delay to avoid rate limiting
            if idx < total_companies:
                sleep(2)  # 2 second delay between companies
        
        print(f"\n✨ Processing complete! Results saved to {output_csv_file}")
        
        # Display summary
        if os.path.exists(output_csv_file):
            results_df = pd.read_csv(output_csv_file)
            print(f"\n📈 Summary:")
            print(f"Total companies processed: {len(results_df)}")
            print(f"Companies with funding data: {(results_df['funding'] != 'Not Found').sum()}")
            print(f"AI-enabled companies: {(results_df['ai_enabled'] == 'Yes').sum()}")
            
    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_csv_file}' not found.")
    except Exception as e:
        print(f"❌ Error processing CSV file: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Specify your input CSV file path
    INPUT_CSV_FILE = "Customer360.csv"  # Change this to your actual file path
    OUTPUT_CSV_FILE = "serp_output.csv"  # Output file name
    
    # Process all companies from the CSV
    process_companies_from_csv(INPUT_CSV_FILE, OUTPUT_CSV_FILE)