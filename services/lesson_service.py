from models import lesson
from langchain_core.documents import Document

def get_lesson_data(lesson: lesson):
    """
    Get lesson data based on its status.
    
    Args:
        lesson (lesson.Lesson): The lesson object to retrieve data from.
    
    Returns:
        dict: A dictionary containing the lesson data.
    """
    data = lesson.model_dump(exclude={"id", "status", "category"})
    
    return "\n".join(f"{key}: {value}" for key, value in data.items())
     

def preprocess_lesson(lesson: lesson):
    """
    Preprocess a lesson based on its status.
    
    Args:
        lesson (lesson.Lesson): The lesson object to preprocess.
    
    Returns:
        None
    """
    lesson_data = get_lesson_data(lesson) 
    doc_id = str(lesson.id)
    
    doc = Document(
        page_content=lesson_data,
        id=doc_id,
        metadata={"category": lesson.category, "title": lesson.title}
    )
    
    return [doc]