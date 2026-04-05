from disease_knowledge import DISEASE_INFO
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")
client = genai.Client(api_key=API_KEY)

def normalize_disease_name(name):
    name = name.lower()
    if "curl" in name:
        return "Cotton Leaf Curl Disease"
    return name.title()

class DiseaseChatbot:
    def get_response(self, disease, conversation):
        try:
            # 1. Safety check for conversation history
            if not conversation or not isinstance(conversation, list):
                last_question = ""
                history = ""
            else:
                last_question = conversation[-1].get("content", "").lower()
                
                # Build history safely
                history = ""
                # Get last 3 messages, handle potential missing keys
                for msg in conversation[-3:]:
                    role = "User" if msg.get("role") == "user" else "Assistant"
                    content = msg.get("content", "")
                    history += f"{role}: {content}\n"

            normalized_disease = normalize_disease_name(disease)

            # 2. Check Local Knowledge Base First
            if normalized_disease in DISEASE_INFO:
                info = DISEASE_INFO[normalized_disease]
                
                if "cause" in last_question:
                    return info["causes"]
                if "symptom" in last_question:
                    return info["symptoms"]
                if "treat" in last_question:
                    return info["treatment"]
                if "prevent" in last_question:
                    return info["prevention"]

            # 3. Fallback to Gemini API
            prompt = f"""
            You are an agricultural expert.
            
            Disease: {normalized_disease}
            
            Conversation History:
            {history}
            
            User Question: {last_question}
            
            Answer clearly and practically.
            """

            # UPDATED: Use a standard model name (Check valid models in Google AI Studio)
            response = client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=prompt
            )
            
            return response.text.strip()

        except Exception as e:
            # This prints the actual error to your terminal so you can see it
            print(f"\n ------------- ERROR LOG -------------")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {str(e)}")
            print(f"-------------------------------------\n")
            
            return (
                "⚠️ AI is temporarily unavailable.\n\n"
                "Error details have been logged. "
                "Please check your API Key and internet connection."
            )