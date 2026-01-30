# MicroService-Extraction

A microservice for extracting financial metrics from SEC 10-K filings using OpenAI. 

## How to Run

1. Install dependencies: `pip install -r requirements.txt`
2. Set up your OpenAI API key in a `.env` file: `OPENAI_API_KEY=your_key_here`
3. Run the server: `python run.py`
4. The API will be available at `http://localhost:8000`
5. Give the Tciker name and year range 
6. Relevent data will store inside the file_data folder

Use the `/process-tenk` endpoint to extract metrics from 10-K filings.
