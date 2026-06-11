# main.py
import os
import pandas as pd
from preprocessing import TextPipeline

def run_single_text(pipeline):
    print("\n---  Single Text Processing ---")
    user_input = input("Enter raw text to preprocess: ")
    
    print("\n---  Live Pipeline Execution Logs ---")
    # Turn on logging temporarily to show steps
    pipeline.use_logging = True
    
    # Run both methods to show the bonus comparison feature
    tokens_lemma = pipeline.process(user_input, method='lemma')
    pipeline.use_logging = False # Turn off logging to keep output clean
    
    tokens_stem = pipeline.process(user_input, method='stem')
    
    print("\n---  Final Comparison Results ---")
    print(f" Raw Input:  \"{user_input}\"")
    print(f" Lemmatized: {tokens_lemma}")
    print(f" Stemmed:    {tokens_stem}")

def run_batch_processing(pipeline):
    print("\n---  Batch CSV Processing ---")
    file_path = input("Enter path to CSV file (e.g., dataset.csv): ").strip()
    
    if not os.path.exists(file_path):
        print(f" Error: File '{file_path}' not found. Creating a mock file for demonstration...")
        # Create dummy file if it doesn't exist
        df_mock = pd.DataFrame({
            'raw_text': [
                "AI Internships are amazing! I am learning so much.",
                "The computational models are running faster than yesterday.",
                "Data science requires clean data; text preprocessing is crucial!!"
            ]
        })
        df_mock.to_csv('dataset.csv', index=False)
        file_path = 'dataset.csv'
        print(f"Mock file generated and saved as '{file_path}'!")

    column_name = input("Enter the text column name to process (Default: 'raw_text'): ").strip() or 'raw_text'
    
    try:
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            print(f" Column '{column_name}' not found in CSV. Available columns: {list(df.columns)}")
            return
        
        print("🔄 Processing dataset columns...")
        df['processed_lemmatized'] = df[column_name].apply(lambda x: pipeline.process(str(x), method='lemma'))
        df['processed_stemmed'] = df[column_name].apply(lambda x: pipeline.process(str(x), method='stem'))
        
        output_file = "processed_dataset.csv"
        df.to_csv(output_file, index=False)
        print(f" Success! Saved fully processed dataset to: '{output_file}'")
        print(df.head())
        
    except Exception as e:
        print(f" An error occurred during file parsing: {e}")

def main():
    # Instantiate pipeline with configurations
    pipeline = TextPipeline(remove_numbers=True, use_logging=False)
    
    while True:
        print("\n=============================================")
        print(" AI INTERN: TEXT PREPROCESSING PIPELINE ENGINE")
        print("=============================================")
        print("1️ Process a Single Text String")
        print("2️ Batch Process a CSV Dataset")
        print("3 Exit Program")
        
        choice = input("Select an option (1-3): ").strip()
        
        if choice == '1':
            run_single_text(pipeline)
        elif choice == '2':
            run_batch_processing(pipeline)
        elif choice == '3':
            print(" Shutting down text pipeline workspace. Happy modeling!")
            break
        else:
            print(" Invalid choice. Please pick a number from 1 to 3.")

if __name__ == "__main__":
    main()