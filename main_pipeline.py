import subprocess

# List of scripts to run in the desired order
scripts = [
    "main_analysis.py",                    # Step 1: Main analysis
    # "preprocessing_to_conllu.py",          # Step 2: Preprocessing
    # "naive_bayes.py",                      # Step 3: Naive Bayes model
    # "BERTScore_fine_tuned.py",             # Step 4: BERTScore fine-tuning
    # "BERTScore_semantic_similarity.py",    # Step 5: BERTScore semantic similarity
    # "create_features.py"                 # Step 6: Create Features for feature classifier
    # "feature_classifier.py"              # Step 7: Feature Classifier
    "naive_bayes_analysis.py",             # Step 8: Naive Bayes analysis
    "bertscore_fine_tuned_analysis.py",    # Step 9: Analysis for BERT fine-tuned model
    "bertscore_semantic_similarity_analysis.py"  # Step 8: Analysis for semantic similarity
]

# Execute each script sequentially
for script in scripts:
    print(f"\nRunning {script}...")
    try:
        subprocess.run(["python", script], check=True)
        print(f"Finished running {script}.")
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script}: {e}")
        break  # Stop the pipeline if an error occurs
