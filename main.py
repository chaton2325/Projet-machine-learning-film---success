import pandas as pd

df=pd.read_csv('movie_dataset_cleaned_final.csv')

#Afficher la moyenne du budget
print("Moyenne du budget:", df['budget'].mean())
#Moyenne des revenus
print("Moyenne des revenus:", df['revenue'].mean())
#Moyenne des revenues moins le budget 
print("Moyenne des revenus moins le budget:", (df['revenue'] - df['budget']).mean())


