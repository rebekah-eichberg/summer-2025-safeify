import time
import undetected_chromedriver as uc
import rapidfuzz as rf
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def duckduckgo_search(query, max_results=10, headless=False):
    """
    Performs a DuckDuckGo search using Selenium and returns the parsed HTML of the results page.

    This function automates a headless (or optionally visible) Chrome browser to open DuckDuckGo,
    perform a search for the given query, and return the resulting page content as a BeautifulSoup object.

    Args:
        query (str): The search term to query DuckDuckGo with.
        max_results (int, optional): Unused in current implementation; reserved for future use to limit results. Defaults to 10.
        headless (bool, optional): If True, runs Chrome in headless mode (no browser window). Defaults to True.

    Returns:
        BeautifulSoup or list: Parsed HTML of the search results page as a BeautifulSoup object.
        Returns an empty list if an error occurs during the search process.

    Notes:
        - Requires undetected-chromedriver (uc), Selenium, BeautifulSoup4, and ChromeDriver v136.
        - The function waits up to 12 seconds for the search box to appear.
        - The function currently does not paginate or limit results using `max_results`.
        - May print errors to the console during scraping.
    """
    options = uc.ChromeOptions()
    
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--remote-debugging-port=9222")

    driver = uc.Chrome(options=options, version_main=136)

    try:
        driver.get("https://duckduckgo.com/")

        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "searchbox_input"))
        )
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)


        time.sleep(2)  # give results a moment to fully load

        source=driver.page_source
        return BeautifulSoup(source,"html.parser")

    except Exception as e:
        print("Error during scraping:", e)
        return []

    finally:
        driver.quit()



def amazon_asin(query:str):
    """
    Searches Amazon for a given product query and attempts to extract ASINs from search results.

    This function uses DuckDuckGo search (via duckduckgo_search) to find product pages on Amazon 
    related to the input query. It parses the search results to identify Amazon product URLs containing 
    ASINs (Amazon Standard Identification Numbers), extracts the product name and ASIN, and computes a 
    fuzzy match score between the query and the extracted product name.

    Args:
        query (str): The product search query (e.g., product description or name). Truncated to 200 characters.

    Returns:
        list[dict] or None: A list of dictionaries, each containing:
            - 'name' (str): Extracted product name from the URL.
            - 'asin' (str): The ASIN string extracted from the Amazon URL.
            - 'score' (int): A fuzzy match score (0–100) indicating relevance to the original query.
        
        Returns None if the query is empty, invalid, or if no ASINs were found.
    
    Note:
        - Relies on an external `duckduckgo_search` function for search and a fuzzy matching tool `rf.fuzz`.
        - This function does not verify if ASINs are still valid on Amazon.
    """
    try:
        # Ensure query is a string and truncate to 200 characters
        query=str(query)[0:200]
        
        # Remove any newline, carriage return, or tab characters
        query=query.replace('\r', '').replace('\n', '').replace('\t','')
        if query=='' or query=='nan':
            return None
        
        # Build the DuckDuckGo search query to limit results to amazon.com
        search_query="site:amazon.com "+str(query)
        
        # Perform the search using DuckDuckGo (browser-based)
        page= duckduckgo_search(search_query)
        
        # Extract all Amazon product links from the search result page
        links=[link.attrs['href'] for link in page.find_all('a',class_='eVNpHGjtxRBq_gLOfGDr LQNqh2U1kzYxREs65IJu')]
        
        # Loop through all extracted links, saving the product name, asin and fuzzscore to results.
        results=[]
        for link in links:
            index=link.find('/dp/')
            if index!=-1:
                asin=link[index+4:]
                name=link[23:index].replace('-',' ')
                score=rf.fuzz.token_set_ratio(name,query)
                results.append({'name': name,'asin': asin,'score':score})
        
        # Return None if no valid ASINs were found
        if results==list():
            return None
    
        return results
    except Exception as error:
        print("An error occurred:", error)
        return None


