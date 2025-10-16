import requests
import json
import csv
import pandas as pd
from time import sleep
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import AzureOpenAI
import os
from datetime import datetime
import re
load_dotenv()

# Configuration
API_KEY = os.getenv("SERPHOUSE_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Define therapeutic categories with their keywords
THERAPEUTIC_CATEGORIES = {
    "small_molecules": ["small molecule", "small molecules"],
    
    "peptides": [
        "peptide", "peptides", "linear peptide", "linear peptides", 
        "cyclic peptide", "cyclic peptides", "stapled peptide", "stapled peptides",
        "cell-penetrating peptide", "cell-penetrating peptides", "CPP", "CPPs",
        "bicyclic peptide", "bicyclic peptides", "peptidomimetic", "peptidomimetics"
    ],
    
    "proteins": [
        "therapeutic enzyme", "therapeutic enzymes", "asparaginase",
        "hormone", "hormones", "insulin", "growth hormone",
        "cytokine", "cytokines", "interferon", "interferons", 
        "interleukin", "interleukins", "glycoprotein", "glycoproteins",
        "clotting factor", "clotting factors", "factor VIII", "factor IX",
        "PEGylated protein", "PEGylated proteins"
    ],
    
    "antibodies": [
        "monoclonal antibody", "monoclonal antibodies", "mAb", "mAbs",
        "human antibody", "humanized antibody", "chimeric antibody",
        "bispecific antibody", "bispecific antibodies",
        "trispecific antibody", "trispecific antibodies",
        "nanobody", "nanobodies", "single-domain antibody", "single-domain antibodies",
        "antibody fragment", "antibody fragments", "Fab", "scFv", "F(ab')2"
    ],
    
    "antibody_conjugates": [
        "antibody-drug conjugate", "antibody-drug conjugates", "ADC", "ADCs",
        "radioimmunoconjugate", "radioimmunoconjugates",
        "immunocytokine", "immunocytokines", "antibody-cytokine fusion",
        "antibody-toxin conjugate", "antibody-toxin conjugates"
    ],
    
    "fusion_proteins": [
        "fusion protein", "fusion proteins", "Fc fusion protein", "Fc fusion proteins",
        "etanercept", "receptor-ligand fusion", "receptor-ligand fusion protein",
        "toxin fusion protein", "toxin fusion proteins",
        "multidomain fusion protein", "multidomain fusion proteins"
    ],
    
    "nucleic_acid_therapeutics": [
        "siRNA", "miRNA", "antisense oligonucleotide", "antisense oligonucleotides",
        "ASO", "ASOs", "mRNA therapeutic", "mRNA therapeutics",
        "aptamer", "aptamers", "ribozyme", "ribozymes"
    ],
    
    "gene_cell_therapies": [
        "CAR-T", "CAR-T cell", "CAR-T cells", "TCR-T", "TCR-T cell", "TCR-T cells",
        "NK cell therapy", "NK cell therapies", "AAV vector", "AAV vectors",
        "adeno-associated virus", "lentiviral vector", "lentiviral vectors",
        "CRISPR", "CRISPR/Cas9", "genome editing", "gene therapy", "cell therapy"
    ],
    
    "vaccines": [
        "protein subunit vaccine", "protein subunit vaccines",
        "mRNA vaccine", "mRNA vaccines", "DNA vaccine", "DNA vaccines",
        "viral vector vaccine", "viral vector vaccines", "adenovirus vaccine",
        "lentivirus vaccine", "conjugate vaccine", "conjugate vaccines",
        "live attenuated vaccine", "live attenuated vaccines",
        "inactivated vaccine", "inactivated vaccines"
    ],
    
    "protein_degraders": [
        "PROTAC", "PROTACs", "molecular glue", "molecular glues",
        "dTAG", "dTAG system", "dTAG systems", "protein degrader", "protein degraders"
    ],
    
    "other_modalities": [
        "exosome", "exosomes", "exosome-based therapeutic", "exosome-based therapeutics",
        "oncolytic virus", "oncolytic viruses", "scaffold protein", "scaffold proteins",
        "DARPin", "DARPins", "affibody", "affibodies",
        "immune cell engager", "immune cell engagers", "BiTE", "BiTEs",
        "nanoparticle", "nanoparticles", "drug delivery nanoparticle"
    ]
}

def search_company_pipeline(company_name):
    """Performs a search specifically for a company's pipeline/latest project using Serphouse API."""
    print(f"\n🔬 Searching for {company_name} pipeline/latest project...")
    
    # More targeted search query for pipeline information
    search_query = f"{company_name} pipeline latest drug candidate therapeutic"
    
    payload = {
        "data": {
            "q": search_query,
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
        print(f"HTTP error occurred for {company_name} pipeline search: {http_err}")
        return [], []
    except Exception as err:
        print(f"An error occurred for {company_name} pipeline search: {err}")
        return [], []

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

def scrape_url(url: str) -> tuple[str | None, str]:
    """
    Scrapes the text content from a given URL.
    Returns tuple of (content, url) for tracking source.
    """
    print(f"  Scraping: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        content = main_content.get_text(separator=' ', strip=True) if main_content else None
        return (content, url)
    except requests.RequestException as e:
        print(f"    Error scraping {url}: {e}")
        return (None, url)

def extract_website_timestamp(url: str) -> str:
    """
    Extracts timestamp/date information from a website and returns in standardized format.
    
    Returns timestamps in ISO 8601 format: YYYY-MM-DD HH:MM:SS
    If time is not available, returns: YYYY-MM-DD 00:00:00
    If no date found, returns: "No timestamp found"
    If error, returns: "Error: [description]"
    """
    print(f"  📅 Extracting timestamp from: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # List to store all found dates with their parsed datetime objects
        found_dates = []
        
        # 1. Check meta tags for published/modified dates
        meta_date_properties = [
            'article:published_time', 'article:modified_time', 
            'datePublished', 'dateModified', 'date',
            'DC.date.created', 'DC.date.modified',
            'og:updated_time', 'publish_date', 'lastmod',
            'created', 'modified', 'updated'
        ]
        
        for prop in meta_date_properties:
            meta_tag = soup.find('meta', property=prop) or soup.find('meta', attrs={'name': prop}) or soup.find('meta', attrs={'itemprop': prop})
            if meta_tag and meta_tag.get('content'):
                parsed = parse_date_string(meta_tag['content'])
                if parsed:
                    found_dates.append((parsed, 'meta_' + prop))
        
        # 2. Check for time tags
        time_tags = soup.find_all('time')
        for time_tag in time_tags[:5]:  # Limit to first 5 time tags
            date_str = time_tag.get('datetime') or time_tag.text.strip()
            if date_str:
                parsed = parse_date_string(date_str)
                if parsed:
                    found_dates.append((parsed, 'time_tag'))
        
        # 3. Check JSON-LD structured data
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                # Handle both single objects and arrays
                items = [data] if isinstance(data, dict) else (data if isinstance(data, list) else [])
                
                for item in items:
                    if isinstance(item, dict):
                        # Direct date fields
                        for date_field in ['datePublished', 'dateModified', 'dateCreated', 'uploadDate', 'datePosted']:
                            if date_field in item:
                                parsed = parse_date_string(item[date_field])
                                if parsed:
                                    found_dates.append((parsed, 'json_ld_' + date_field))
                        
                        # Check nested structures (common in Article, NewsArticle, etc.)
                        if '@graph' in item:
                            for graph_item in item['@graph']:
                                if isinstance(graph_item, dict):
                                    for date_field in ['datePublished', 'dateModified', 'dateCreated']:
                                        if date_field in graph_item:
                                            parsed = parse_date_string(graph_item[date_field])
                                            if parsed:
                                                found_dates.append((parsed, 'json_ld_graph_' + date_field))
            except (json.JSONDecodeError, TypeError):
                pass
        
        # 4. Check for dates in common class/id patterns
        date_selectors = [
            ('class', ['date', 'publish-date', 'post-date', 'entry-date', 'article-date', 
                      'created-date', 'modified-date', 'updated-date', 'timestamp']),
            ('id', ['date', 'publish-date', 'post-date', 'article-date']),
            ('itemprop', ['datePublished', 'dateModified', 'dateCreated'])
        ]
        
        for attr_name, attr_values in date_selectors:
            for attr_value in attr_values:
                elements = soup.find_all(attrs={attr_name: re.compile(attr_value, re.I)})
                for element in elements[:3]:  # Check first 3 matches
                    date_str = element.get('datetime') or element.get('content') or element.text.strip()
                    if date_str:
                        parsed = parse_date_string(date_str)
                        if parsed:
                            found_dates.append((parsed, f'{attr_name}_{attr_value}'))
        
        # 5. Look for date patterns in visible text (last resort)
        if not found_dates:
            # Common date patterns in text
            text_content = soup.get_text()[:10000]  # Check first 10000 chars
            
            # More comprehensive date patterns
            date_patterns = [
                (r'\b(\d{4})-(\d{1,2})-(\d{1,2})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?\b', 'iso'),  # ISO format
                (r'\b(\d{1,2})/(\d{1,2})/(\d{4})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?\b', 'us'),  # US format
                (r'\b(\d{1,2})\.(\d{1,2})\.(\d{4})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?\b', 'eu'),  # EU format
                (r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),?\s+(\d{4})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?\b', 'month_first'),
                (r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?\b', 'day_first'),
            ]
            
            for pattern, format_type in date_patterns:
                matches = re.findall(pattern, text_content, re.IGNORECASE)
                for match in matches[:2]:  # Take first 2 matches
                    parsed = parse_date_from_match(match, format_type)
                    if parsed:
                        found_dates.append((parsed, 'text_pattern'))
                        break
                if found_dates:
                    break
        
        # 6. Sort dates and select the most recent one (likely the last modified date)
        if found_dates:
            # Sort by date, most recent first
            found_dates.sort(key=lambda x: x[0], reverse=True)
            
            # Format the date in standardized format: YYYY-MM-DD HH:MM:SS
            best_date = found_dates[0][0]
            return best_date.strftime('%Y-%m-%d %H:%M:%S')
        
        # 7. If no date found in content, check HTTP headers as last resort
        if 'last-modified' in response.headers:
            parsed = parse_date_string(response.headers['last-modified'])
            if parsed:
                return parsed.strftime('%Y-%m-%d %H:%M:%S')
        
        return "No timestamp found"
        
    except requests.RequestException as e:
        return f"Error: Connection failed - {str(e)[:50]}"
    except Exception as e:
        return f"Error: {str(e)[:50]}"

def parse_date_string(date_str: str) -> datetime | None:
    """
    Attempts to parse a date string into a datetime object.
    Handles various common date formats.
    """
    if not date_str or not isinstance(date_str, str):
        return None
    
    # Clean the string
    date_str = date_str.strip()
    
    # Common date formats to try
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S%z',
        '%Y-%m-%d',
        '%Y/%m/%d',
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%Y.%m.%d',
        '%d.%m.%Y',
        '%B %d, %Y',
        '%b %d, %Y',
        '%d %B %Y',
        '%d %b %Y',
        '%Y-%m-%d %H:%M',
        '%Y/%m/%d %H:%M:%S',
        '%m/%d/%Y %H:%M:%S',
        '%a, %d %b %Y %H:%M:%S %Z',  # RFC 2822
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%S.%fZ',
    ]
    
    for fmt in formats:
        try:
            # Handle timezone-aware strings
            if '+' in date_str and fmt == '%Y-%m-%dT%H:%M:%S%z':
                # Fix timezone format if needed (+0000 -> +00:00)
                if re.match(r'.*\+\d{4}$', date_str):
                    date_str = date_str[:-2] + ':' + date_str[-2:]
            
            parsed = datetime.strptime(date_str, fmt)
            
            # If no time component was in the format, set to 00:00:00
            if '%H' not in fmt:
                parsed = parsed.replace(hour=0, minute=0, second=0)
                
            return parsed
        except ValueError:
            continue
    
    # Try ISO format with milliseconds and timezone
    try:
        # Handle ISO format with various timezone formats
        if 'T' in date_str:
            # Remove milliseconds if present
            if '.' in date_str:
                date_str = re.sub(r'\.\d+', '', date_str)
            
            # Handle Z timezone
            if date_str.endswith('Z'):
                date_str = date_str[:-1] + '+00:00'
            
            # Try parsing as ISO format
            if '+' in date_str or '-' in date_str[-6:]:
                return datetime.fromisoformat(date_str)
            else:
                return datetime.fromisoformat(date_str)
    except:
        pass
    
    return None

def parse_date_from_match(match: tuple, format_type: str) -> datetime | None:
    """
    Parse date from regex match based on format type.
    """
    try:
        if format_type == 'iso':
            # YYYY-MM-DD format
            year, month, day = int(match[0]), int(match[1]), int(match[2])
            hour = int(match[3]) if match[3] else 0
            minute = int(match[4]) if match[4] else 0
            second = int(match[5]) if match[5] else 0
            return datetime(year, month, day, hour, minute, second)
            
        elif format_type == 'us':
            # MM/DD/YYYY format
            month, day, year = int(match[0]), int(match[1]), int(match[2])
            hour = int(match[3]) if match[3] else 0
            minute = int(match[4]) if match[4] else 0
            second = int(match[5]) if match[5] else 0
            return datetime(year, month, day, hour, minute, second)
            
        elif format_type == 'eu':
            # DD.MM.YYYY format
            day, month, year = int(match[0]), int(match[1]), int(match[2])
            hour = int(match[3]) if match[3] else 0
            minute = int(match[4]) if match[4] else 0
            second = int(match[5]) if match[5] else 0
            return datetime(year, month, day, hour, minute, second)
            
        elif format_type == 'month_first':
            # Month DD, YYYY format
            month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
            month = month_map.get(match[0][:3].lower(), 1)
            day = int(match[1])
            year = int(match[2])
            hour = int(match[3]) if len(match) > 3 and match[3] else 0
            minute = int(match[4]) if len(match) > 4 and match[4] else 0
            second = int(match[5]) if len(match) > 5 and match[5] else 0
            return datetime(year, month, day, hour, minute, second)
            
        elif format_type == 'day_first':
            # DD Month YYYY format
            month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
            day = int(match[0])
            month = month_map.get(match[1][:3].lower(), 1)
            year = int(match[2])
            hour = int(match[3]) if len(match) > 3 and match[3] else 0
            minute = int(match[4]) if len(match) > 4 and match[4] else 0
            second = int(match[5]) if len(match) > 5 and match[5] else 0
            return datetime(year, month, day, hour, minute, second)
            
    except (ValueError, IndexError):
        pass
    
    return None

def extract_latest_project_with_llm(scraped_data_with_urls: list, company_name: str) -> dict | None:
    """
    Sends pipeline/project-specific scraped content to Azure OpenAI to extract 
    the latest drug candidate/project details.
    
    Args:
        scraped_data_with_urls: List of tuples (content, url) from pipeline search
        company_name: Name of the company
    
    Returns:
        Dictionary with molecule, target, and classification
    """
    print(f"\n💊 Extracting latest project details for {company_name} with Azure OpenAI...")
    if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_DEPLOYMENT_NAME]):
        print("❌ Error: Azure OpenAI environment variables are not set.")
        return None

    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version="2023-07-01-preview"
    )

    # Prepare context with URL markers
    context_parts = []
    for content, url in scraped_data_with_urls:
        if content:
            context_parts.append(f"--- SOURCE URL: {url} ---\n{content}")
    
    full_context = "\n\n".join(context_parts)

    prompt = f"""
    Based on the provided text about the company "{company_name}", extract information about their LATEST, MOST RECENT, or FLAGSHIP drug candidate/therapeutic project.
    
    Your task is to identify:
    1. MOLECULE: Compound/Inhibitor/Inducer/agonist/Antagonist/Modulator (e.g.: 'CT053', 'CT071')
    2. TARGET: Enzyme/Protein/Receptor (e.g.: DPP-4, Dipeptidyl Peptidase‑4)
    3. TARGET TYPE: Type of target (e.g., "kinases", "GPCR", "Ion Channels", "Nuclear Receptors")
    4. THERAPEUTIC AREA: Therapeutic area or indication (e.g., "oncology", "immunology", "metabolic diseases", "CNS")
    5. CLASSIFICATION: A brief classification of what type of therapeutic it is (e.g., "monoclonal antibody", "small molecule inhibitor", "CAR-T therapy", "mRNA vaccine")
    6. DEVELOPMENT STAGE: Development stage (e.g., "preclinical", "Phase 1", "Phase 2", "Phase 3", "approved")
    
    Look for:
    - Lead candidates or flagship products
    - Recently announced programs
    - Pipeline highlights
    - Products currently in development or recently approved
    - The most prominent or advanced therapeutic mentioned
    
    --- START OF PROVIDED TEXT ---
    {full_context[:24000]}
    --- END OF PROVIDED TEXT ---
    
    Format your response as a valid JSON object with the following structure:
    {{
        "molecule": "Compound/Inhibitor/Inducer/agonist/Antagonist/Modulator (e.g.: 'CT053', 'CT071'). If not found, say 'Not specified'",
        "target": "Enzyme/Protein/Receptor (e.g.: DPP-4, Dipeptidyl Peptidase‑4). If not found, say 'Not specified'",
        "target_type": "Type of target (e.g., 'kinases', 'GPCR', 'Ion Channels', 'Nuclear Receptors'). If not found, say 'Not specified'",
        "therapeutic_area": "Therapeutic area or indication (e.g., 'oncology', 'immunology', 'metabolic diseases', 'CNS'). If not found, say 'Not specified'",
        "classification": "Type of therapeutic (e.g., 'monoclonal antibody', 'small molecule'). If not found, say 'Not specified'"
        "development_stage": "Development stage (e.g., 'preclinical', 'Phase 1', 'Phase 2', 'Phase 3', 'approved'). If not found, say 'Not specified'"
    }}
    
    Important Guidelines:
    - Focus on the MOST RECENT or MOST PROMINENT drug candidate mentioned
    - Provide specific names, not generic descriptions
    - If multiple candidates exist, choose the one that appears most advanced or recently mentioned
    - Be concise but specific
    - If no clear drug candidate is found, return "Not specified" for all fields
    """

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a pharmaceutical intelligence analyst that extracts structured information about drug candidates and therapeutic projects from company information. You identify the drug molecule name, its target/indication, and classification. You always return valid JSON with all required fields."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        json_response = response.choices[0].message.content
        return json.loads(json_response)
    except Exception as e:
        print(f"❌ Error during latest project extraction for {company_name}: {e}")
        return None

def classify_company_with_llm(scraped_data_with_urls: list, company_name: str) -> dict | None:
    """
    Sends combined text to Azure OpenAI to classify company into therapeutic categories
    and detect preclinical development activities.
    
    Args:
        scraped_data_with_urls: List of tuples (content, url)
        company_name: Name of the company
    
    Returns:
        Dictionary with classification results including keywords, URLs, and preclinical stage
    """
    print(f"\n💡 Classifying {company_name} with Azure OpenAI...")
    if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_DEPLOYMENT_NAME]):
        print("❌ Error: Azure OpenAI environment variables are not set.")
        return None

    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version="2023-07-01-preview"
    )

    # Prepare context with URL markers
    context_parts = []
    source_urls = []
    for content, url in scraped_data_with_urls:
        if content:
            context_parts.append(f"--- SOURCE URL: {url} ---\n{content}")
            source_urls.append(url)
    
    full_context = "\n\n".join(context_parts)

    # Create the categories description for the prompt
    categories_description = """
    1. small_molecules: Small molecule drugs
    2. peptides: Linear peptides, Cyclic peptides, Stapled peptides, Cell-penetrating peptides (CPPs), Bicyclic peptides, Peptidomimetics
    3. proteins: Enzymes (therapeutic enzymes like asparaginase), Hormones (insulin, growth hormone), Cytokines (interferons, interleukins), Glycoproteins, Clotting factors (Factor VIII, IX), PEGylated proteins
    4. antibodies: Monoclonal antibodies (mAbs - human, humanized, chimeric), Bispecific antibodies, Trispecific antibodies, Nanobodies (single-domain antibodies), Antibody fragments (Fab, scFv, F(ab')2)
    5. antibody_conjugates: Antibody–Drug Conjugates (ADCs), Radioimmunoconjugates, Immunocytokines (antibody–cytokine fusions), Antibody–Toxin conjugates
    6. fusion_proteins: Fc fusion proteins (e.g., etanercept), Receptor-ligand fusion proteins, Toxin fusion proteins, Multidomain fusion proteins
    7. nucleic_acid_therapeutics: siRNA, miRNA, Antisense oligonucleotides (ASOs), mRNA therapeutics, Aptamers, Ribozymes
    8. gene_cell_therapies: CAR-T cells, TCR-T cells, NK cell therapies, AAV vectors, Lentiviral vectors, CRISPR/Cas9 genome editing systems
    9. vaccines: Protein subunit vaccines, mRNA vaccines, DNA vaccines, Viral vector vaccines (adenovirus, lentivirus), Conjugate vaccines, Live attenuated, Inactivated vaccines
    10. protein_degraders: PROTACs, Molecular Glues, dTAG systems
    11. other_modalities: Exosome-based therapeutics, Oncolytic viruses, Scaffold proteins (DARPins, Affibodies), Immune cell engagers (BiTEs), Nanoparticles for drug delivery
    """

    # Preclinical keywords list
    preclinical_keywords = """
    Early Drug Discovery, Hit Discovery, Lead Optimization, Target Discovery, Assay Development,
    Pre Clinical, Preclinical, In Vitro Biology, In Vivo Pharmacology, ADME/DMPK, ADME, DMPK,
    Binding Assays, Toxicity, Toxicology, Safety, Cell Based Assays, Biochemical Assays,
    Hit-To-Lead, Hit to Lead, Medicinal Chemistry, Structure Activity Relationships, SAR,
    Analytical & Biophysical Analysis, Analytical Analysis, Biophysical Analysis,
    Molecular Synthesis, Biologics, Gene-to-protein, Gene to protein,
    Antibody Discovery, Immuno Oncology, Immunooncology, Cell Engineering
    """

    prompt = f"""
    Based on the provided text about the company "{company_name}", perform the following tasks:
    
    1. CLASSIFY THERAPEUTIC CATEGORIES: For each category, determine if the company is working on it.
    2. IDENTIFY KEYWORDS: If "Yes", provide the exact keyword/phrase from the text that proves this.
    3. IDENTIFY SOURCE URL: If "Yes", provide the URL where this keyword was found (look for "--- SOURCE URL:" markers in the text).
    4. DETECT PRECLINICAL STAGE: Determine if the company is involved in preclinical or early-stage drug discovery activities (Yes/No only).
    5. RETRIEVE DEVELOPMENT STAGE: If available, provide the development stage (e.g., Preclinical, Phase 1, Phase 2, Phase 3, Approved). If not found, return "Not specified".
    6. RETRIEVE THERAPEUTIC AREAS: If available, provide the therapeutic areas (e.g., Oncology, Immunology, Rare Diseases). If not found, return "Not specified".
    
    Therapeutic Categories:
    {categories_description}
    
    Preclinical/Early Drug Discovery Keywords to Look For:
    {preclinical_keywords}
    
    For the preclinical_stage field, respond with "Yes" ONLY if you find ANY of the preclinical keywords or clear evidence of early-stage drug discovery, hit discovery, lead optimization, target discovery, assay development, preclinical testing, in vitro/in vivo studies, ADME/DMPK, toxicology, safety studies, medicinal chemistry, or related early-stage R&D activities. Otherwise respond with "No".
    
    --- START OF PROVIDED TEXT ---
    {full_context[:24000]}
    --- END OF PROVIDED TEXT ---
    
    Format your response as a valid JSON object with the following structure:
    {{
        "preclinical_stage": "Yes" or "No",
        "categories": {{
            "small_molecules": {{
                "classification": "Yes" or "No",
                "keyword": "exact keyword found" or "",
                "source_url": "URL where keyword was found" or ""
            }},
            "peptides": {{
                "classification": "Yes" or "No",
                "keyword": "exact keyword found" or "",
                "source_url": "URL where keyword was found" or ""
            }},
            "proteins": {{
                "classification": "Yes" or "No",
                "keyword": "exact keyword found" or "",
                "source_url": "URL where keyword was found" or ""
            }},
            "antibodies": {{
                "classification": "Yes" or "No",
                "keyword": "exact keyword found" or "",
                "source_url": "URL where keyword was found" or ""
            }},
            "antibody_conjugates": {{
                "classification": "Yes" or "No",
                "keyword": "exact keyword found" or "",
                "source_url": "URL where keyword was found" or ""
            }},
            "fusion_proteins": {{
                "classification": "Yes" or "No",
                "keyword": "exact keyword found" or "",
                "source_url": "URL where keyword was found" or ""
            }},
            "nucleic_acid_therapeutics": {{
                "classification": "Yes" or "No",
                "keyword": "exact keyword found" or "",
                "source_url": "URL where keyword was found" or ""
            }},
            "gene_cell_therapies": {{
                "classification": "Yes" or "No",
                "keyword": "exact keyword found" or "",
                "source_url": "URL where keyword was found" or ""
            }},
            "vaccines": {{
                "classification": "Yes" or "No",
                "keyword": "exact keyword found" or "",
                "source_url": "URL where keyword was found" or ""
            }},
            "protein_degraders": {{
                "classification": "Yes" or "No",
                "keyword": "exact keyword found" or "",
                "source_url": "URL where keyword was found" or ""
            }},
            "other_modalities": {{
                "classification": "Yes" or "No",
                "keyword": "exact keyword found" or "",
                "source_url": "URL where keyword was found" or ""
            }}
        }}
    }}
    
    Important Guidelines:
    - Only return "Yes" if there is clear evidence the company is actively developing or has developed therapeutics in that category
    - When "Yes", MUST provide the actual keyword/phrase found in the text
    - When "Yes", MUST provide the source URL where the keyword was found (extract from "--- SOURCE URL:" marker before the content)
    - If uncertain or no information is found, return "No" with empty keyword and source_url
    - For preclinical_stage, return "Yes" if ANY preclinical or early drug discovery activities are mentioned, otherwise "No"
    """

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a precise pharmaceutical company classifier that analyzes companies and determines which therapeutic modalities they are working on, identifies the exact keywords that prove this, notes the source URLs, and determines if they are involved in preclinical/early-stage drug discovery (Yes/No only). You always return valid JSON with all required fields."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        json_response = response.choices[0].message.content
        return json.loads(json_response)
    except Exception as e:
        print(f"❌ Error during LLM classification for {company_name}: {e}")
        return None

def save_to_csv(classification_data: dict, company_name: str, website_timestamp: str, project_data: dict, filename: str = "therapeutic_classification_output.csv"):
    """
    Appends the classified company data to a CSV file with enhanced columns.
    
    New CSV structure:
    - company_name
    - website_timestamp
    - preclinical_stage
    - molecule
    - development_stage
    - target
    - target_type
    - theraputic_area
    - classification
    - For each category: [category], [category]_keyword, [category]_source_url
    
    Args:
        classification_data: Dictionary with therapeutic category classifications
        company_name: Name of the company
        website_timestamp: Website timestamp
        project_data: Dictionary with molecule, target, target_type, theraputic_area and classification
        filename: Output CSV filename
    """
    # Build fieldnames dynamically
    fieldnames = ['company_name', 'website_timestamp', 'preclinical_stage', 'molecule', 'target', 'target_type', 'theraputic_area', 'classification']
    
    for category in THERAPEUTIC_CATEGORIES.keys():
        fieldnames.extend([
            category,
            f"{category}_keyword",
            f"{category}_source_url"
        ])
    
    file_exists = os.path.isfile(filename)
    
    # Prepare row data
    row_data = {
        'company_name': company_name,
        'website_timestamp': website_timestamp,
        'preclinical_stage': classification_data.get('preclinical_stage', 'No'),
        'molecule': project_data.get('molecule', 'Not specified'),
        'target': project_data.get('target', 'Not specified'),
        'target_type': project_data.get('target_type', 'Not specified'),
        'theraputic_area': project_data.get('theraputic_area', 'Not specified'),
        'classification': project_data.get('classification', 'Not specified'),
        'development_stage': project_data.get('development_stage', 'Not specified')
    }
    
    # Add category data
    categories_data = classification_data.get('categories', {})
    for category in THERAPEUTIC_CATEGORIES.keys():
        category_info = categories_data.get(category, {})
        
        if isinstance(category_info, dict):
            row_data[category] = category_info.get('classification', 'No')
            row_data[f"{category}_keyword"] = category_info.get('keyword', '')
            row_data[f"{category}_source_url"] = category_info.get('source_url', '')
        else:
            # Fallback for old format
            row_data[category] = category_info if category_info in ['Yes', 'No'] else 'No'
            row_data[f"{category}_keyword"] = ''
            row_data[f"{category}_source_url"] = ''

    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)
    
    print(f"✅ Classification for {company_name} appended successfully to {filename}")

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

def process_company_data(company_name: str, company_website: str, output_csv_file: str):
    """
    Main function to process search results for a single company.
    Enhanced to:
    1. Search company general information for therapeutic classification
    2. Search specifically for pipeline/latest project information
    3. Extract drug molecule, target, and classification from project data
    """
    print(f"\n{'='*50}\nProcessing classification for: {company_name}\n{'='*50}")
    
    # Extract timestamp from company website if provided
    website_timestamp = "Not provided"
    if company_website and company_website.strip() and company_website.lower() not in ['nan', 'none', 'n/a', '']:
        # Ensure URL has protocol
        if not company_website.startswith(('http://', 'https://')):
            company_website = 'https://' + company_website
        website_timestamp = extract_website_timestamp(company_website)
        print(f"   Timestamp extracted: {website_timestamp}")
    
    # === PART 1: General Company Search for Therapeutic Classification ===
    print(f"\n📋 STEP 1: Searching for general company information...")
    organic_results, top_stories_results = search_company(company_name)
    
    # Combine all sources and get a unique list of links
    all_sources = organic_results + top_stories_results
    unique_urls = list(set(item['link'] for item in all_sources if item.get('link')))
    
    # Add company website to scraping list if provided and not already in results
    if company_website and company_website.strip() and company_website.lower() not in ['nan', 'none', 'n/a', '']:
        if company_website not in unique_urls:
            unique_urls.insert(0, company_website)

    # Default project data
    default_project_data = {
        'molecule': 'Not specified',
        'target': 'Not specified',
        'target_type': 'Not specified',
        'theraputic_area': 'Not specified',
        'classification': 'Not specified',
        'development_stage': 'Not specified'
    }

    if not unique_urls:
        print(f"No URLs found to process for {company_name}.")
        default_data = {
            'preclinical_stage': 'No',
            'categories': {category: {'classification': 'No', 'keyword': '', 'source_url': ''} 
                          for category in THERAPEUTIC_CATEGORIES.keys()}
        }
        save_to_csv(default_data, company_name, website_timestamp, default_project_data, output_csv_file)
        return

    # Scrape all unique links for classification
    scraped_data_with_urls = []
    for url in unique_urls[:10]:  # Limit to first 10 URLs
        content, source_url = scrape_url(url)
        if content:
            scraped_data_with_urls.append((content, source_url))

    if not scraped_data_with_urls:
        print(f"Could not scrape any content from the provided URLs for {company_name}.")
        default_data = {
            'preclinical_stage': 'No',
            'categories': {category: {'classification': 'No', 'keyword': '', 'source_url': ''} 
                          for category in THERAPEUTIC_CATEGORIES.keys()}
        }
        save_to_csv(default_data, company_name, website_timestamp, default_project_data, output_csv_file)
        return

    # Classify the company using LLM
    print(f"\n🔬 STEP 2: Classifying therapeutic categories...")
    classification_data = classify_company_with_llm(scraped_data_with_urls, company_name)

    if not classification_data:
        print(f"❌ Failed to classify {company_name}.")
        default_data = {
            'preclinical_stage': 'No',
            'categories': {category: {'classification': 'No', 'keyword': '', 'source_url': ''} 
                          for category in THERAPEUTIC_CATEGORIES.keys()}
        }
        save_to_csv(default_data, company_name, website_timestamp, default_project_data, output_csv_file)
        return

    # === PART 2: Pipeline-Specific Search for Latest Project ===
    print(f"\n💊 STEP 3: Searching for pipeline/latest project information...")
    pipeline_organic, pipeline_stories = search_company_pipeline(company_name)
    
    # Combine pipeline search results
    pipeline_sources = pipeline_organic + pipeline_stories
    pipeline_urls = list(set(item['link'] for item in pipeline_sources if item.get('link')))
    
    # Scrape pipeline-specific URLs
    pipeline_scraped_data = []
    if pipeline_urls:
        for url in pipeline_urls[:10]:  # Limit to first 10 URLs
            content, source_url = scrape_url(url)
            if content:
                pipeline_scraped_data.append((content, source_url))
    
    # Extract latest project information using dedicated LLM call
    project_data = default_project_data
    if pipeline_scraped_data:
        print(f"\n📊 STEP 4: Extracting drug molecule, target, and classification...")
        extracted_project = extract_latest_project_with_llm(pipeline_scraped_data, company_name)
        if extracted_project:
            project_data = extracted_project
            print(f"   ✅ Drug Molecule: {project_data.get('molecule', 'Not specified')}")
            print(f"   ✅ Target: {project_data.get('target', 'Not specified')}")
            print(f"   ✅ Target Type: {project_data.get('target_type', 'Not specified')}")
            print(f"   ✅ Therapeutic Area: {project_data.get('theraputic_area', 'Not specified')}")
            print(f"   ✅ Classification: {project_data.get('classification', 'Not specified')}")
            print(f"   ✅ Development Stage: {project_data.get('development_stage', 'Not specified')}")
        else:
            print(f"   ⚠️ Could not extract project details")
    else:
        print(f"   ⚠️ No pipeline content found to analyze")

    # Save all results to CSV
    save_to_csv(classification_data, company_name, website_timestamp, project_data, output_csv_file)

def process_companies_from_csv(input_csv_file: str, output_csv_file: str = "therapeutic_classification_output.csv"):
    """
    Reads company names and websites from a CSV file and processes each one for classification.
    Automatically resumes from where it left off if the script was interrupted.
    
    Args:
        input_csv_file: Path to CSV file containing company names in 'company_name' column 
                       and optionally websites in 'Website' or 'Company_Website' column
        output_csv_file: Path to output CSV file for classification data
    """
    try:
        # Read the input CSV file
        df1 = pd.read_csv(input_csv_file)
        df = df1.drop_duplicates(subset=['company_name'])
        
        if 'company_name' not in df.columns:
            print(f"❌ Error: 'company_name' column not found in {input_csv_file}")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Check for website column (try different possible column names)
        website_column = None
        possible_website_columns = ['Website', 'website', 'Company_Website', 'company_website', 
                                   'URL', 'url', 'funding_citation', 'Funding_Citation']
        
        for col in possible_website_columns:
            if col in df.columns:
                website_column = col
                break
        
        if not website_column:
            print("⚠️ No website column found. Will proceed without company websites.")
            df['Website'] = ''  # Add empty website column
            website_column = 'Website'
        
        # Get unique company names with their websites
        company_data = df[['company_name', website_column]].dropna(subset=['company_name'])
        company_data = company_data.rename(columns={website_column: 'Website'})
        
        # Get already processed companies
        processed_companies = get_processed_companies(output_csv_file)
        
        # Filter out already processed companies
        companies_to_process = []
        for _, row in company_data.iterrows():
            if row['company_name'] not in processed_companies:
                companies_to_process.append((row['company_name'], row.get('Website', '')))
        
        total_companies = len(company_data)
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
        for idx, (company, website) in enumerate(companies_to_process, 1):
            overall_progress = completed_companies + idx
            print(f"\n📊 Progress: {overall_progress}/{total_companies} (Processing {idx}/{remaining_companies} remaining)")
            if website and str(website).strip() and str(website).lower() not in ['nan', 'none', 'n/a']:
                print(f"   Company website: {website}")
            
            try:
                process_company_data(company, website if pd.notna(website) else '', output_csv_file)
            except Exception as e:
                print(f"❌ Error processing {company}: {e}")
                # Save error entry to maintain progress tracking
                error_data = {
                    'preclinical_stage': 'Error',
                    'categories': {category: {'classification': 'Error', 'keyword': '', 'source_url': ''} 
                                  for category in THERAPEUTIC_CATEGORIES.keys()}
                }
                error_project_data = {
                    'molecule': 'Error',
                    'target': 'Error',
                    'classification': 'Error'
                }
                save_to_csv(error_data, company, "Error", error_project_data, output_csv_file)
            
            # Add a small delay to avoid rate limiting
            if idx < remaining_companies:
                sleep(2)  # 2 second delay between companies
        
        print(f"\n✨ Processing complete! Results saved to {output_csv_file}")
        
        # Display summary
        if os.path.exists(output_csv_file):
            results_df = pd.read_csv(output_csv_file)
            print(f"\n📈 Classification Summary:")
            print(f"   Total companies processed: {len(results_df)}")
            
            # Count companies with timestamps
            if 'website_timestamp' in results_df.columns:
                valid_timestamps = results_df[
                    (results_df['website_timestamp'] != 'Not provided') & 
                    (results_df['website_timestamp'] != 'No timestamp found') &
                    (results_df['website_timestamp'] != 'Error') &
                    (~results_df['website_timestamp'].str.startswith('Error:', na=False))
                ]
                print(f"   Companies with valid timestamps: {len(valid_timestamps)}")
            
            # Count companies by preclinical stage
            if 'preclinical_stage' in results_df.columns:
                print(f"\n   Preclinical Stage Distribution:")
                preclinical_yes = (results_df['preclinical_stage'] == 'Yes').sum()
                preclinical_no = (results_df['preclinical_stage'] == 'No').sum()
                print(f"      Preclinical/Early Discovery Activities: {preclinical_yes}")
                print(f"      No Preclinical Activities Found: {preclinical_no}")
            
            # Count companies in each category
            print(f"\n   Therapeutic Category Distribution:")
            for category in THERAPEUTIC_CATEGORIES.keys():
                if category in results_df.columns:
                    count = (results_df[category] == 'Yes').sum()
                    print(f"      {category.replace('_', ' ').title()}: {count} companies")
            
            # Count errors
            error_count = 0
            for category in THERAPEUTIC_CATEGORIES.keys():
                if category in results_df.columns:
                    error_count += (results_df[category] == 'Error').sum()
            if error_count > 0:
                print(f"\n   Errors encountered: {error_count}")
            
    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_csv_file}' not found.")
    except Exception as e:
        print(f"❌ Error processing CSV file: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Specify your input CSV file path
    INPUT_CSV_FILE = "seg_test.csv"  # Change this to your actual file path
    OUTPUT_CSV_FILE = "final3.csv"  # Enhanced output file name
    
    # Process all companies from the CSV
    process_companies_from_csv(INPUT_CSV_FILE, OUTPUT_CSV_FILE)