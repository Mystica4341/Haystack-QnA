from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import FARMReader,DensePassageRetriever, DocxToTextConverter,TextConverter, PreProcessor
from haystack.pipelines import ExtractiveQAPipeline
from fastapi import APIRouter
from pydantic import BaseModel
import os

#Init Router
HaystackRouter = APIRouter()

#Init Model
class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str

#Create an InMemoryDocumentStore
document_store = InMemoryDocumentStore()

#Converter: convert file to text
converter = DocxToTextConverter(valid_languages=["en"])

#Preprocessor: clean and split text
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="word",
    split_length=200,
    split_respect_sentence_boundary=True,
)

## Read file in folder
document_dir = "data"
ProcessedDocs = []
for file in os.listdir(document_dir):
    if file.endswith(".docx"):
        file_path = os.path.join(document_dir, file)
        convertedFiles = converter.convert(file_path, meta=None)
        processedFiles = preprocessor.process(convertedFiles)
        ProcessedDocs.extend(processedFiles)

## Static path
# doc = converter.convert(file_path="data/ChatAI.docx", meta=None)
# ProcessedDocs = preprocessor.process(doc)

#Write docs to InMemoryDocumentStore
document_store.write_documents(ProcessedDocs)

#Init Retriever
retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
)

#Update embeddings
document_store.update_embeddings(retriever)

#Init Reader
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

#Init Pipeline
pipeline = ExtractiveQAPipeline(reader, retriever)

## Command Test
# while True:
#     question = input("Question: ")
#     result = reader.predict(
#         query=question,
#         documents=ProcessedDocs,
#         top_k=1
#     )
#     print(result["answers"][0].answer)

## API Test
#Add router
@HaystackRouter.post("/ask")
async def ask(q: Question):
    result = reader.predict(
        query=q.question,
        documents=ProcessedDocs,
        top_k=1
    )
    return {"Answer": result["answers"][0].answer}
    