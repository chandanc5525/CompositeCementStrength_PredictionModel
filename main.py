from src.data_ingestion import data_ingestion

# Define Orchestron Model: Entry Point 
def main(): # Function Definition
  # Step1: Data Ingestion
  df = data_ingestion()
  print(df.head())



main()    # Function Calling