# %%
import os
os.environ["GEMINI_API_KEY"]="AIzaSyC3xzwKpOkR9N7K944bZTkGhkEf6QGF9BM"

# %%
!pip install chromadb pypdf rouge_score

# %%

from pypdf import PdfReader

def load_pdf(file_path):  #خواندن فایل و برگرداندن آن به عنوان یک رشته

    # خواندن فایل pdf
    reader = PdfReader(file_path)

    # ذخیره
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    return text


pdf_text = load_pdf(file_path="./متن قانون.pdf")




# %%


pdf_text = load_pdf(file_path="./متن قانون.pdf")
#csv_text = load_csv(file_path="./datasetchatbot162.csv")

# ترکیب متن پی دی اف و فایل پرسش و پاسخ
combined_text = pdf_text# + "\n" + csv_text


print(combined_text)

# %%
import re
def split_text(text: str):
   #جدا سازی متن به زیر رشته هایی درون یک آرایه
    split_text = re.split('\n \n', text)
    return [i for i in split_text if i != ""]

chunked_text = split_text(text=combined_text)

# %%
import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
import os

class GeminiEmbeddingFunction(EmbeddingFunction):



    def __call__(self, input: Documents) -> Embeddings:             #تولید امبدینگ برای متن داده شده
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(model=model,
                                   content=input,
                                   task_type="retrieval_document",
                                   title=title)["embedding"]

# %%
import chromadb
from typing import List
def create_chroma_db(documents:List, path:str, name:str):         #ساخت کروما دیتابیس برای ذخیره امبدینگ ها

    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())

    for i, d in enumerate(documents):
        db.add(documents=d, ids=str(i))

    return db, name

db,name =create_chroma_db(documents=chunked_text,
                          path="./chromadb",
                          name="rag_experiment7")

# %%
def load_chroma_collection(path, name):             #بارگیری امبدینگ از کروما دیتابیس

    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())

    return db

db=load_chroma_collection(path="./chromadb", name="rag_experiment7")

# %%
def get_relevant_passage(query, db, n_results):
  passage = db.query(query_texts=[query], n_results=n_results)['documents'][0]
  return passage

#Example usage
relevant_text = get_relevant_passage(query="قانون حمایت خانواده چیست؟",db=db,n_results=3)
print(relevant_text)

# %%
def make_rag_prompt(query, relevant_passage):       #Prompt Engineering
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = ("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
  strike a friendly and converstional tone. \
  If the passage is irrelevant to the answer, you may ignore it.
  QUESTION: '{query}'
  PASSAGE: '{relevant_passage}'

  ANSWER:
  """).format(query=query, relevant_passage=escaped)

  return prompt

# %%
import google.generativeai as genai
def generate_answer1(prompt):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(prompt)
    return answer.text

# %%
def generate_answer(db,query):
    #برگرداندن 3 قطعه متن که بیشترین ارتباط را به سوال دارند.
    relevant_text = get_relevant_passage(query,db,n_results=3)
    prompt = make_rag_prompt(query,
                             relevant_passage="".join(relevant_text)) # یکی کردن قطعه متون برگردانده شده
    answer = generate_answer1(prompt)

    return answer

# %%
import ipywidgets as widgets
from IPython.display import display


def on_generate_answer(b):
    query = text_box.value
    answer = generate_answer(db, query)
    result_box.value = answer

# Load database
db = load_chroma_collection(path="./chromadb", name="rag_experiment7")

# Create a text box for input
text_box = widgets.Textarea(
    value='',
    placeholder='Enter your query here',
    description='Query:',
    disabled=False,
    layout=widgets.Layout(width='100%', height='100px')
)

# Create a button to generate the answer
button = widgets.Button(
    description='Generate Answer',
    disabled=False,
    button_style='',
    tooltip='Click to generate answer',
    icon='check'
)

# Create a text box to display the result
result_box = widgets.Textarea(
    value='',
    placeholder='Answer will appear here',
    description='Answer:',
    disabled=True,
    layout=widgets.Layout(width='100%', height='100px')
)

# Set up the button click event
button.on_click(on_generate_answer)

# Display the widgets
display(text_box)
display(button)
display(result_box)


# %%
db=load_chroma_collection(path="./chromadb",
                          name="rag_experiment7")

query = """قانون تعیین مدت
اعتبار گواهی عدم امکان سازش مصوب در چه تاریخی تصویب شد؟"""
answer = generate_answer(db,query)
print(answer)

# %%
from rouge_score import rouge_scorer

def evaluate_rouge(generated_texts, reference_texts):
    """
    Evaluates ROUGE scores for generated texts against reference texts.

    Parameters:
    - generated_texts (List[str]): A list of generated responses from the model.
    - reference_texts (List[str]): A list of reference responses.

    Returns:
    - Dict: ROUGE scores for ROUGE-1, ROUGE-2, and ROUGE-L.
    """
    assert len(generated_texts) == len(reference_texts), "The number of generated texts and reference texts must be the same."

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    for gen_text, ref_text in zip(generated_texts, reference_texts):
        score = scorer.score(ref_text, gen_text)
        for key in scores.keys():
            scores[key].append(score[key].fmeasure)

    # Calculate average scores
    avg_scores = {key: sum(values) / len(values) for key, values in scores.items()}

    return avg_scores

# Example usage
generated_texts = [
    "Pursuant to Article 64 of the Family Support Act, the presence of children under the age of 15 in family court hearings is prohibited, except in necessary cases that the court may authorize."
]

reference_texts = [
    """Article 46: The presence of children under fifteen years of age in the hearing of family lawsuits is prohibited, except in cases required by the court."""
]

rouge_scores = evaluate_rouge(generated_texts, reference_texts)
print("ROUGE Scores:")
print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")


# %%



