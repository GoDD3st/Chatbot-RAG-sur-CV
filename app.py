import streamlit as st
import re
import docx
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

def load_cv(file):
    text = ""
    if file.name.endswith('.pdf'):
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() # extraction du text de chaque page du PDF et ajout au text global
    elif file.name.endswith('.docx'):
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n" # extraction du text de chaque paragraphe du DOCX et ajout au text global
    return text #retourne le text extrait pouor chaque type de fichier soit PDF ou DOCX

def clean_text(text):
    text = re.sub(r'\s+', ' ', text) #remplace les séquences de caractères d'espacement par un seul espace
    return text.strip() #supprime les espaces en début et en fin de chaîne de caractères

def chunk_document(text, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter( # pour decouper le texte en plusieurs chunks en utilisant pusieur separateurs comme les paragraphes, lignes etc
        chunk_size=chunk_size,
        chunk_overlap=overlap, 
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

@st.cache_resource # j'ai utilise le cache de streamlit pour stocker le modèle d'embedding en mémoire et éviter de le recharger à chaque fois que la fonction est appelée, ce qui améliore les performances de l'application.
def build_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # j'ai choisi ce modèle d'embedding car il est léger et rapide tout en offrant de bonnes performances pour les tâches de similarité de texte.

def build_embeddings(chunks, model):
    return model.embed_documents(chunks)

def create_vector_store(chunks, embeddings_model):
    vector_store = FAISS.from_texts(chunks, embeddings_model) # indexation de chaque chunk et le stocker daans la BD vectorielle faiss
    return vector_store

def retrieve_relevant_chunks(question, store, k=5): # j'ai choisi k = 5 pour recuperer les 5 chunks les plus proches dde laa question
    return store.similarity_search(question, k=k)

def build_prompt(question, context):
    return f"""
    Tu es un assistant virtuel expert en recrutement. 
    SERS-TOI UNIQUEMENT DU CONTEXTE CI-DESSOUS POUR RÉPONDRE.
    
    CONTRÔLE DE SÉCURITÉ :
    Si la question n'a aucun rapport avec le CV ou si la réponse n'est pas dans le texte, 
    réponds EXACTEMENT : "Désolé, je ne peux répondre qu'aux questions concernant ce CV."
    
    CONTEXTE EXTRAIT DU CV :
    {context}
    
    QUESTION : {question}
    RÉPONSE :"""

load_dotenv() # charger les variables d'environnement depuis .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # chargement de api de gemini utilise dans ce projet
def generate_answer(prompt):
    try: # le try catch pour gerer les erreurs qui peuvent surveniir loors de l'appel de api pour eviter que l'app se plante
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", # le modele de gemini choisie est le plus rapide et gratuit
            google_api_key=GEMINI_API_KEY,
            temperature=0, # eviter l'hallucination du modele
        )
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Erreur lors de la génération : {str(e)}"

def answer_cv_only(question, store):
    relevant_docs = retrieve_relevant_chunks(question, store) # on recupere seulement les chunks les plus proches de laa questiioion
    context = "\n".join([doc.page_content for doc in relevant_docs]) # on lie les chunks choisie pouor construire le contexte de la quesstion
    full_prompt = build_prompt(question, context) # on construit le prompt depuis le contexte et la question
    return generate_answer(full_prompt)

def reload_vector_store(file):
    raw_text = load_cv(file)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_document(cleaned_text)
    model = build_embeddings_model()
    return create_vector_store(chunks, model)

def run_streamlit_app():
    st.set_page_config(page_title="CV Chatbot", layout="centered")
    st.title("CV Chatbot")
    
    file = st.file_uploader("Upload your CV (PDF or DOCX)", type=["pdf", "docx"])

    if file is not None:
        st.info(f"CV Actif : {file.name}")

        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != file.name:
            st.session_state.last_uploaded_file = file.name
            with st.spinner("Analyse du document en cours..."):
                st.session_state.vector_store = reload_vector_store(file)
                st.session_state.messages = []
            st.success("Base vectorielle mise à jour !")

        for message in st.session_state.get("messages", []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Posez une question sur le CV"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
        
            with st.chat_message("assistant"):
                if "vector_store" in st.session_state:
                    response = answer_cv_only(prompt, st.session_state.vector_store)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error("Veuillez charger un CV d'abord.")

if __name__ == "__main__":
    run_streamlit_app()