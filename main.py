import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
from os import listdir
import shutil
from models import lesson
from services import lesson_service, file_service, db_service, llm

app = FastAPI(title = "MyAPI", version = "2.1.0", description = "PROJECT_DESCRIPTION")

origins = [
    "http://localhost",
    "http://localhost:4200",
    "http://0.0.0.0:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/upload-file')
def upload_file(uploaded_file: UploadFile = File(..., alias="file")):
    """
    upload new file to files directory.

    Args:
        uploaded_file (file): the pdf input file.

    Returns:
        result (object): the file data and process result.
    """

    """
    Implement your logic here to save the uploaded file to the desired directory. For example, you can use the shutil library to save the file to a specific folder.
    """
    uploadFolder = "Files"
    os.makedirs(uploadFolder, exist_ok=True)
    
    filePath = os.path.join(uploadFolder, uploaded_file.filename)
    
    with open(filePath, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)
    
    return {
        'uploaded': True,
        'message' : f'File "{uploaded_file.filename}" uploaded successfully'
    }

@app.get('/process-batch-files')
def process_batch_files():
    """
    process, and save a new pdf file into db.

    Returns:
        result (object): the file data and process result.
    """
    
    temp_folder_path = "Files"
    if not os.path.exists(temp_folder_path):
        return {'uploaded': False, 'message': 'No files directory found!'}

    files = listdir(temp_folder_path)
    if not files:
        return {
            'uploaded': False,
            'message': 'No files to be processed!'
        }

    total_chunks = []
    
    for file_name in files:
        file_path = os.path.join(temp_folder_path, file_name)
        if os.path.isfile(file_path):
            # Process file to extract chunks
            chunks = file_service.preprocess_file(file_path)
            total_chunks.extend(chunks)

    # Add all extracted chunks to the database in one operation
    if total_chunks:
        db_service.add_docs(total_chunks)

    return {
        'uploaded': True,
        'message': f'Successfully processed {len(files)} file(s) into {len(total_chunks)} chunk(s).'
    }

@app.post('/process-lesson')
def process_lesson(lesson: lesson):
    """
    process, and save a new lesson into db, update, delete from db.

    Args:
        lesson (lesson): lesson body.

    Returns:
        state (state): the lesson data and process result.
    """
    
    chunks = lesson_service.preprocess_lesson(lesson)
    
    match lesson.status.lower():
        case "new":
            db_service.add_docs(chunks)
            return {
                'processed': True,
                'message': 'Lesson processed successfully'
            }
          
        case "update":
            db_service.update_docs(chunks)
            return {
                'processed': True,
                'message': 'Lesson updated successfully'
            }
            
        case "delete":
            db_service.delete_docs(chunks)
            return {
                'processed': True,
                'message': 'Lesson deleted successfully'
            }
            
        case _:
            raise HTTPException(status_code=400, detail=f"Invalid status: '{lesson.status}'")

@app.post('/get-chunks')
def get_chunks(query: str):
    """
    get chunks from db.

    Returns:
        result list(objects): the chunks data.
    """
    """
    Implement your logic here to retrieve chunks from the database based on the provided query. For example, you can call a function from the files_service to fetch the relevant chunks.
    """
    
    results = db_service.similarity_search(query)
    
    return {
        'chunks': results,
        'message': 'Chunks retrieved successfully'
    }

@app.post('/chatbot')
def chatbot(query: str):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    res = llm.getResponse(query)
    
    return {
        "answer": res["answer"],
        "source_documents": res["source_documents"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)