### Step 1
Run the pip install command below:
```
pip install -r requirements.txt
```
### Step 2
Run these python files in **crawl** folder to crawl data on admission website
```
py crawl_diemchuan_2023.py
py crawl_diemchuan_2024.py
py crawl-nganh.py
```
### Step 3
Run **data-preprocessing.py** to processing data from *dataset* folder into a vector database: 
```
py data-preprocessing.py
```
### Step 4
Run **app.py** to operate the chatbot (CUDA require for faster answering): 
```
py app.py
```