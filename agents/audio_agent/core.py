import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
import shutil

# Try to import groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

class AudioAgent:
    def __init__(self, groq_api_key):
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        if GROQ_AVAILABLE:
            self.groq_client = Groq(api_key=groq_api_key)
        else:
            self.groq_client = None
            
        self.vector_stores = {}
        self.chat_histories = {}
        self.uploaded_files = {}

    def process_audio(self, audio_path, session_id, original_filename, language_mode="Auto-Detect Language"):
        """Process audio file using Groq Cloud API"""
        if not self.groq_client:
            return "❌ Groq library not installed. Please run: pip install groq"

        try:
            # Prepare file for upload
            with open(audio_path, "rb") as file:
                # Map language mode to ISO codes
                lang_code = None
                if "English" in language_mode:
                    lang_code = "en"
                elif "Arabic" in language_mode:
                    lang_code = "ar"
                elif "French" in language_mode:
                    lang_code = "fr"
                elif "Spanish" in language_mode:
                    lang_code = "es"
                
                # Call Groq Audio API
                completion = self.groq_client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo", 
                    file=(original_filename, file.read()), 
                    response_format="text",
                    language=lang_code 
                )
            
            transcript = completion
            
            # Create document
            doc = Document(page_content=transcript, metadata={"source": original_filename})
            
            # Split text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents([doc])
            
            # Create vector store
            persist_directory = f"data/audio_vector_stores/{session_id}"
            os.makedirs(persist_directory, exist_ok=True)
            
            if session_id in self.vector_stores:
                self.vector_stores[session_id].add_documents(splits)
            else:
                vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings,
                    persist_directory=persist_directory
                )
                self.vector_stores[session_id] = vectorstore
            
            # Track file
            if session_id not in self.uploaded_files:
                self.uploaded_files[session_id] = []
            if original_filename not in self.uploaded_files[session_id]:
                self.uploaded_files[session_id].append(original_filename)
            
            return f"✅ Audio '{original_filename}' transcribed successfully!"
        
        except Exception as e:
            return f"❌ Error processing audio: {e}"

    def get_uploaded_files(self, session_id):
        """Return list of uploaded audio files"""
        return self.uploaded_files.get(session_id, [])

    def format_docs(self, docs):
        """Format documents for context"""
        return "\n\n".join(doc.page_content for doc in docs)

    def get_response(self, query, session_id):
        """Get response based on audio transcript"""
        if session_id not in self.vector_stores:
            return "⚠️ No audio uploaded yet. Please upload an audio file first."
        
        try:
            vectorstore = self.vector_stores[session_id]
            retriever = vectorstore.as_retriever()
            
            if session_id not in self.chat_histories:
                self.chat_histories[session_id] = []
            
            template = """Answer the question based only on the following audio transcript:
{context}

Question: {question}

Answer: """
            
            prompt = ChatPromptTemplate.from_template(template)
            
            rag_chain = (
                {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            response = rag_chain.invoke(query)
            
            self.chat_histories[session_id].append(HumanMessage(content=query))
            self.chat_histories[session_id].append(AIMessage(content=response))
            
            return response
        
        except Exception as e:
            return f"❌ Error generating response: {e}"

    def clear_context(self, session_id):
        """Clear context for session"""
        if session_id in self.vector_stores:
            del self.vector_stores[session_id]
        
        if session_id in self.chat_histories:
            del self.chat_histories[session_id]
        
        if session_id in self.uploaded_files:
            del self.uploaded_files[session_id]
        
        persist_directory = f"data/audio_vector_stores/{session_id}"
        if os.path.exists(persist_directory):
            try:
                shutil.rmtree(persist_directory)
            except Exception as e:
                print(f"Warning: Could not delete directory {persist_directory}: {e}")