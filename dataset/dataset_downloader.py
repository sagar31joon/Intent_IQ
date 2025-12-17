#dataset/dataset_downlaoder.py
from datasets import load_dataset
import pandas as pd
import os

#dataset1 = "elenigkove/Email_Intent_Classification"
#dataset2 = "Bhuvaneshwari/intent_classification"
#dataset3 = "Vinitrajputt/jarvis_intent"
dataset_loaded= load_dataset("Bhuvaneshwari/intent_classification") #downloading dataset from hugging face / loading
print(dataset_loaded)

test_df = dataset_loaded["test"].to_pandas() #loading test data
train_df = dataset_loaded["train"].to_pandas() #loading train data
#validation_df = dataset_loaded["validation"].to_pandas() #loading validaton split
df = pd.concat([train_df, test_df], ignore_index = True) #joing both splits in dataframe

print(df.head(25))
print(df.shape)
intent_counts = df['intent'].value_counts()

print("Total intents:", df['intent'].nunique())
print("\nTexts per intent:\n")
print(intent_counts)


#df = test_split.to_pandas()
"""output_filename = "dataset/Email_Intent_Classification.csv" 
df.to_csv(output_filename, index = False) #saving dataset as csv file

print(f"Dataset successfully downloaded and saved to {os.path.abspath(output_filename)}") # data set successfullly saved"""