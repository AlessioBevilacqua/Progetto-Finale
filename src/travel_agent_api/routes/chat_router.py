from fastapi import APIRouter
from pydantic import BaseModel  

from travel_agent_api.services.agent_service import Agent

router = APIRouter()

class ChatCompletionRequest(BaseModel):
    messages: list

    model_config = {
        "json_schema_extra": {
            "example": { 
                "messages": [
                    {
                        "role": "user",
                        "content": "Vorrei organizzare un viaggio a Roma"  
                    }
                ]
            }
        }
    }

@router.post("/travel-agent")
def chat_completion(request: ChatCompletionRequest):
    """
    Endpoint per la gestione delle richieste di chat.
    Processa i messaggi ricevuti e restituisce una risposta dall'agente di viaggio.
    
    Args:
        request (ChatComplentionRequest): La richiesta contenente i messaggi della conversazione
        
    Returns:
        dict: La risposta elaborata dall'agente di viaggio
        
    Raises:
        HTTPException: In caso di errori durante l'elaborazione della richiesta
    """
    agent = Agent()
    
    response = agent.run(messages=request.messages)
    # Tracing
    print("*" * 80)
    print("chat_completion")
    print("*" * 80)
    # print("request messages: ", request.messages)
    # print("response: ", response)
    # Restituzione della risposta
    return response