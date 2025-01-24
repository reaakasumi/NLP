import subprocess

# List of scripts to run in the desired order
scripts = [
    "main_analysis.py",                    # Step 1: Main analysis
    # "preprocessing_to_conllu.py",          # Step 2: Preprocessing
    "naive_bayes.py",                        # Step 3: Naive Bayes model
    "naive_bayes_analysis.py",                #further Analysis on Naive Bayes
    "vectara_improved.py",                    #Step 4: Vectara
    # "BERTScore_fine_tuned.py",             # Step 5: BERTScore fine-tuning
    # "BERTScore_semantic_similarity.py",    # Step 6: BERTScore semantic similarity
    # "create_features.py"                 # Step 7: Create Features for feature classifier
    # "feature_classifier.py"              # Step 8: Feature Classifier
    "bertscore_fine_tuned_analysis.py",    # Step 9: Analysis for BERT fine-tuned model
    "bertscore_semantic_similarity_analysis.py",  # Step 10: Analysis for semantic similarity
    "majority_model_with_weights.py"         #Majority Approach
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
