import json
from openai import OpenAI
from config import get_config
import os
import re
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from typing import List
import requests

config = get_config()

class MetricExtractor:
    """Class for extracting financial metrics from 10-K text using OpenAI, with text preprocessing and downloading."""

    def __init__(self):
        self.metrics_with_descriptions = {
            "Depreciation": "depreciation expense related to PPE or property plant equipment",
            "DepreciationAndAmortization": "depreciation amortization expense",
            "OperatingLeaseCost": "operating lease cost or cash flow or cash payment",
            "VariableLeaseCost": "variable lease cost or cash flow or cash payment",
            "StockBasedCompensation": "Share or stock based compensation",
            "OperatingLeaseAssets": "operating ROU right of use lease assets both short-term/current and long-term/noncurrent combined",
            "OperatingLeaseLiabilities": "operating lease liabilities both short-term/current and long-term/noncurrent combined",
            "OperatingLeaseNewAssetsObtained": "operating lease new assets obtained",
            "OperatingLeaseDiscountRate": "operating lease discount rate or interest rate %",
            "FinanceLeaseAssets": "finance or financing lease assets both short-term/current and long-term/noncurrent combined",
            "FinanceLeaseLiabilities": "finance lease liabilities total both short-term/current and long-term/noncurrent",
            "FinanceLeaseNewAssetsObtained": "finance lease new assets obtained",
            "FinanceLeaseDiscountRate": "finance lease discount rate or interest rate %",
            "FinanceLeaseTerm": "finance lease term remaining years",
            "Goodwill": "total goodwill asset",
            "DeferredTaxAssets": "deferred income tax asset",
            "DeferredTaxLiabilities": "deferred income tax liability",
            "InterestExpense": "interest expense from debt and leasing",
            "InterestIncome": "interest income"
        }
        self.headers = {"User-Agent": "YourName your@email.com"}

    def download_10k_html(self, ticker: str, year: str) -> str:
        """
        Download 10-K HTML from SEC EDGAR for given ticker and year.
        
        Args:
            ticker: Company ticker symbol
            year: Year as string (e.g., "2023")
            
        Returns:
            Path to downloaded HTML file
        """
        print(f"Downloading 10-K for {ticker} {year}...")
        
        # 1) Ticker → CIK
        ticker_url = "https://www.sec.gov/files/company_tickers.json"
        ticker_data = requests.get(ticker_url, headers=self.headers).json()

        cik = None
        for item in ticker_data.values():
            if item["ticker"].upper() == ticker.upper():
                cik = str(item["cik_str"]).zfill(10)
                break

        if not cik:
            raise ValueError(f"Ticker {ticker} not found")

        print(f"CIK for {ticker}: {cik}")

        # 2) Get company filings
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        data = requests.get(submissions_url, headers=self.headers).json()

        forms = data["filings"]["recent"]["form"]
        dates = data["filings"]["recent"]["filingDate"]
        docs = data["filings"]["recent"]["primaryDocument"]
        accessions = data["filings"]["recent"]["accessionNumber"]

        # 3) Find 10-K for the requested year
        html_url = None
        for i in range(len(forms)):
            if forms[i] == "10-K" and dates[i].startswith(year):
                acc = accessions[i].replace("-", "")
                doc = docs[i]
                html_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
                print(f"10-K {year} URL: {html_url}")
                break

        if not html_url:
            raise ValueError(f"No 10-K found for {ticker} in {year}")

        # 4) Download HTML
        html = requests.get(html_url, headers=self.headers).text
        
        # Save to file
        filename = f"{ticker}_{year}_10K.html"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html)
        
        print(f"Downloaded and saved to {filename}")
        return filename

    def is_number(self, s: str) -> bool:
        """Check if a string is a number (with optional commas and decimals)."""
        s = s.replace(',', '').replace('$', '').strip()
        try:
            float(s)
            return True
        except ValueError:
            return False

    def clean_markdown_table(self, table_content: str) -> str:
        """
        Cleans up a single markdown table by removing empty columns and merging related cells.
        """
        lines = table_content.strip().split('\n')
        
        if not lines:
            return table_content
        
        # Extract table rows
        table_rows = []
        for line in lines:
            if '|' in line:
                cells = [cell.strip() for cell in line.split('|')]
                if cells and cells[0] == '':
                    cells = cells[1:]
                if cells and cells[-1] == '':
                    cells = cells[:-1]
                table_rows.append(cells)
        
        if not table_rows:
            return table_content
        
        # Pad rows to have the same number of columns
        num_cols = max(len(row) for row in table_rows)
        for row in table_rows:
            while len(row) < num_cols:
                row.append('')
        
        # Merge cells: if a cell is just "$" and the next cell is a number, merge them
        for row in table_rows:
            i = 0
            while i < len(row) - 1:
                current = row[i].strip()
                next_cell = row[i + 1].strip()
                
                # Check if current cell is $ and next is a number
                if current == '$' and next_cell and self.is_number(next_cell):
                    row[i] = f"$ {next_cell}"
                    row[i + 1] = ''
                    i += 1
                else:
                    i += 1
        
        # Remove completely empty columns
        non_empty_cols = []
        for col_idx in range(num_cols):
            has_content = False
            for row in table_rows:
                if col_idx < len(row) and row[col_idx].strip() and row[col_idx].strip() != '---':
                    has_content = True
                    break
            if has_content:
                non_empty_cols.append(col_idx)
        
        # Filter out empty columns
        filtered_rows = []
        for row in table_rows:
            filtered_row = [row[i] for i in non_empty_cols if i < len(row)]
            filtered_rows.append(filtered_row)
        
        # Remove empty rows
        filtered_rows = [row for row in filtered_rows if any(cell.strip() and cell.strip() != '---' for cell in row)]
        
        if not filtered_rows or len(filtered_rows) < 1:
            return table_content
        
        # Determine column widths
        num_final_cols = len(filtered_rows[0])
        col_widths = [0] * num_final_cols
        
        for row in filtered_rows:
            for idx in range(min(len(row), num_final_cols)):
                col_widths[idx] = max(col_widths[idx], len(row[idx]))
        
        # Format the table with proper newlines
        formatted_lines = []
        
        # Add header row (first non-separator row)
        header_row = filtered_rows[0]
        formatted_cells = []
        for idx in range(num_final_cols):
            cell = header_row[idx] if idx < len(header_row) else ''
            formatted_cells.append(cell.ljust(col_widths[idx]))
        formatted_lines.append('| ' + ' | '.join(formatted_cells) + ' |')
        
        # Add separator row
        separator_cells = ['-' * width for width in col_widths]
        formatted_lines.append('| ' + ' | '.join(separator_cells) + ' |')
        
        # Add data rows
        for row_idx in range(1, len(filtered_rows)):
            row = filtered_rows[row_idx]
            formatted_cells = []
            for idx in range(num_final_cols):
                cell = row[idx] if idx < len(row) else ''
                formatted_cells.append(cell.ljust(col_widths[idx]))
            formatted_lines.append('| ' + ' | '.join(formatted_cells) + ' |')
        
        return '\n'.join(formatted_lines)

    def format_markdown_content(self, markdown_content: str) -> str:
        """
        Process markdown content and clean up all tables while preserving text.
        """
        lines = markdown_content.split('\n')
        result_lines = []
        table_buffer = []
        in_table = False
        
        for line in lines:
            # Check if line contains table marker
            if '|' in line:
                table_buffer.append(line)
                in_table = True
            else:
                # If we were in a table and now we're not, process the table
                if in_table and table_buffer:
                    table_content = '\n'.join(table_buffer)
                    try:
                        cleaned_table = self.clean_markdown_table(table_content)
                        result_lines.append(cleaned_table)
                    except Exception as e:
                        # If cleaning fails, keep original table
                        print(f"Warning: Failed to clean table: {e}")
                        result_lines.extend(table_buffer)
                    
                    table_buffer = []
                    in_table = False
                
                # Add non-table line
                result_lines.append(line)
        
        # Handle case where file ends with a table
        if table_buffer:
            table_content = '\n'.join(table_buffer)
            try:
                cleaned_table = self.clean_markdown_table(table_content)
                result_lines.append(cleaned_table)
            except Exception as e:
                print(f"Warning: Failed to clean table: {e}")
                result_lines.extend(table_buffer)
        
        return '\n'.join(result_lines)

    def remove_lines_up_to_table_of_contents(self, markdown_content: str) -> str:
        """
        Removes all lines up to and including the line that contains 'TABLE OF CONTENTS'.
        """
        lines = markdown_content.split('\n')
        for idx, line in enumerate(lines):
            if 'TABLE OF CONTENTS' in line.upper():
                # Return everything after this line
                return '\n'.join(lines[idx + 1:])
        # If 'TABLE OF CONTENTS' not found, return original content
        return markdown_content

    def preprocess_text(self, html_file_path: str) -> str:
        """
        Preprocess HTML file to clean text with formatted tables.
        
        Args:
            html_file_path: Path to HTML file
            
        Returns:
            Cleaned text content
        """
        if not os.path.exists(html_file_path):
            raise FileNotFoundError(f"HTML file not found: {html_file_path}")
        
        with open(html_file_path, "r", encoding="utf-8") as f:
            html = f.read()
        
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        
        markdown = md(str(soup), heading_style="ATX")
        markdown = self.remove_lines_up_to_table_of_contents(markdown)
        cleaned_text = self.format_markdown_content(markdown)
        
        return cleaned_text

    def extract_financial_metrics(self, full_text_content: str):
        """
        Extract financial metrics from preprocessed 10-K text using OpenAI.
        Verifies metric presence, correctness, and prioritizes context order.

        Args:
            full_text_content (str): Full preprocessed 10-K text content

        Returns:
            dict: Extracted metrics with cost/infra tracking
        """
        prompt = "You are a financial data extraction assistant.\n\n"
        prompt += "Extract the numerical values for the following metrics from the text below. Include metric names and values in JSON format. Use null if a metric is not found.\n\n"
        prompt += "IMPORTANT: Return values as clean numbers without commas, dollar signs, or other formatting. For example, extract '2,500,000' as 2500000, not as '2,500,000'.\n\n"
        prompt += "Metrics:\n"
        for metric, desc in self.metrics_with_descriptions.items():
            prompt += f"- {metric}: {desc}\n"

        # Prioritize context order: Financial Statements > MD&A > Notes
        context_hierarchy = ["Consolidated Financial Statements", "Management Discussion & Analysis", "Notes to Financial Statements"]
        prompt += "\nPrioritize extraction from: " + " > ".join(context_hierarchy)
        prompt += "\n\nText content:\n\"\"\"\n" + full_text_content + "\n\"\"\""

        extracted_metrics = {}
        cost_info = {"backend": "openai", "total_cost_usd": 0.0, "tokens_used": 0, "infra_estimate": {"type": "cloud", "notes": "OpenAI API usage"}}

        try:
            client = OpenAI(api_key=config.OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=config.TENK_DATA_EXTRACTOR_OPENAI_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in extracting financial metrics from text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            result_text = response.choices[0].message.content
            usage = response.usage
            cost_info["total_cost_usd"] = (usage.prompt_tokens / 1000 * 0.03) + (usage.completion_tokens / 1000 * 0.06)
            cost_info["tokens_used"] = usage.total_tokens

            # Parse JSON response
            try:
                extracted_metrics = json.loads(result_text.strip("```json").strip("```").strip())
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse LLM response as JSON: {result_text}")
                extracted_metrics = {"error": "Invalid JSON response"}

        except Exception as e:
            print(f"Error extracting metrics with OpenAI: {e}")
            extracted_metrics = {"error": str(e)}

        return {"metrics": extracted_metrics, "cost_info": cost_info}


    def extract_combined_data(self, full_text_content: str):
        """
        Extract financial metrics from preprocessed 10-K text.
        
        Args:
            full_text_content (str): Full preprocessed 10-K text content
            
        Returns:
            dict: Extracted metrics with cost/infra tracking
        """
        # Extract metrics from the full text (with chunking if needed)
        metrics_data = self._extract_metrics_from_full_text(full_text_content)
        
        return {"data": {"metrics": metrics_data["metrics"]}, "cost_info": metrics_data["cost_info"]}

    def _chunk_text(self, text: str, max_chunk_tokens: int = 12000) -> list:
        """
        Split text into chunks that fit within token limits.
        Uses approximate token counting (1 token ≈ 4 characters).
        """
        # Rough token estimation: 1 token ≈ 4 characters
        max_chunk_chars = max_chunk_tokens * 4
        
        if len(text) <= max_chunk_chars:
            return [text]
        
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > max_chunk_chars and current_chunk:
                # Start new chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def _extract_metrics_from_full_text(self, full_text_content: str):
        """
        Extract financial metrics from the full text, with chunking support for large documents.
        """
        # Check if text needs chunking (rough token estimate)
        estimated_tokens = len(full_text_content) / 4  # 1 token ≈ 4 chars
        
        # Use different limits based on model
        model = config.TENK_DATA_EXTRACTOR_OPENAI_MODEL_NAME
        if model == "gpt-4o-mini":
            max_context = 128000  # 128k tokens
            safe_limit = 100000   # Leave room for prompt and response
        elif model == "gpt-4o":
            max_context = 128000
            safe_limit = 100000
        else:  # gpt-3.5-turbo
            max_context = 16385
            safe_limit = 12000
        
        if estimated_tokens > safe_limit:
            print(f"Text too long ({estimated_tokens:.0f} tokens), chunking into smaller parts...")
            return self._extract_metrics_from_chunks(full_text_content)
        else:
            return self._extract_metrics_from_single_chunk(full_text_content)

    def _extract_metrics_from_single_chunk(self, text_content: str):
        """
        Extract metrics from a single chunk of text.
        """
        prompt = "You are a financial data extraction assistant.\n\n"
        prompt += "Extract the numerical values for the following metrics from the text below. Include metric names and values in JSON format. Use null if a metric is not found.\n\n"
        prompt += "IMPORTANT: Return values as clean numbers without commas, dollar signs, or other formatting. For example, extract '2,500,000' as 2500000, not as '2,500,000'.\n\n"
        prompt += "Metrics:\n"
        for metric, desc in self.metrics_with_descriptions.items():
            prompt += f"- {metric}: {desc}\n"

        prompt += "\nPrioritize extraction from: Financial Statements > Management Discussion & Analysis > Notes\n\n"
        prompt += "Text content:\n\"\"\"\n" + text_content + "\n\"\"\""

        extracted_metrics = {}
        cost_info = {"backend": "openai", "total_cost_usd": 0.0, "tokens_used": 0}

        try:
            client = OpenAI(api_key=config.OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=config.TENK_DATA_EXTRACTOR_OPENAI_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in extracting financial metrics from text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            result_text = response.choices[0].message.content
            usage = response.usage
            
            # Calculate cost based on model
            model = config.TENK_DATA_EXTRACTOR_OPENAI_MODEL_NAME
            if model == "gpt-4o":
                input_rate = 0.0025
                output_rate = 0.01
            elif model == "gpt-3.5-turbo":
                input_rate = 0.0005
                output_rate = 0.0015
            elif model == "gpt-4o-mini":
                input_rate = 0.00015
                output_rate = 0.0006
            else:
                input_rate = 0.03
                output_rate = 0.06
            
            cost_info["total_cost_usd"] = (usage.prompt_tokens / 1000 * input_rate) + (usage.completion_tokens / 1000 * output_rate)
            cost_info["tokens_used"] = usage.total_tokens

            try:
                extracted_metrics = json.loads(result_text.strip("```json").strip("```").strip())
            except json.JSONDecodeError:
                extracted_metrics = {"error": "Invalid JSON response"}

        except Exception as e:
            extracted_metrics = {"error": str(e)}

        return {"metrics": extracted_metrics, "cost_info": cost_info}

    def _extract_metrics_from_chunks(self, full_text_content: str):
        """
        Extract metrics by chunking the text and processing each chunk separately.
        """
        chunks = self._chunk_text(full_text_content)
        print(f"Split into {len(chunks)} chunks")
        
        all_metrics = {}
        total_cost = 0.0
        total_tokens = 0
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            chunk_result = self._extract_metrics_from_single_chunk(chunk)
            
            # Merge metrics (later chunks can override earlier ones if they have better data)
            if "error" not in chunk_result["metrics"]:
                for metric, value in chunk_result["metrics"].items():
                    if value is not None:  # Only override if we found a non-null value
                        all_metrics[metric] = value
            
            total_cost += chunk_result["cost_info"]["total_cost_usd"]
            total_tokens += chunk_result["cost_info"]["tokens_used"]
        
        # Fill in any missing metrics with null
        for metric in self.metrics_with_descriptions.keys():
            if metric not in all_metrics:
                all_metrics[metric] = None
        
        cost_info = {
            "backend": "openai", 
            "total_cost_usd": total_cost, 
            "tokens_used": total_tokens,
            "chunks_processed": len(chunks)
        }
        
        return {"metrics": all_metrics, "cost_info": cost_info}



# Example usage
if __name__ == "__main__":
    extractor = MetricExtractor()
    # Preprocess HTML to text
    # cleaned_text = extractor.preprocess_text("path/to/10k.html")
    # result = extractor.extract_financial_metrics(cleaned_text)
    # print(json.dumps(result, indent=4))
    print("MetricExtractor ready. Use preprocess_text() then extract_financial_metrics() or extract_combined_data().")