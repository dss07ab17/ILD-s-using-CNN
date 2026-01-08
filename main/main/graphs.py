import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv(r"C:\Users\saisa\OneDrive\Desktop\New Microsoft Excel Worksheet.csv")

print(df)

plt.plot(df['F1 Score'], df['Accuracy'])
plt.title('F1 score vs Accuracy')
plt.xlabel('F1 score')
plt.ylabel('Accuracy')