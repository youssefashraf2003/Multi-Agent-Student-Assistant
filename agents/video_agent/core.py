import os
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

class VideoAgent:
    def __init__(self, groq_api_key):
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

    def extract_video_id(self, url: str) -> str:
        """
        Extract the YouTube video ID from a URL.
        """
        parsed = urlparse(url)
        if parsed.hostname == 'youtu.be':
            return parsed.path[1:]
        if parsed.hostname in ('www.youtube.com', 'youtube.com'):
            if parsed.path == '/watch':
                p = parse_qs(parsed.query)
                if 'v' in p:
                    return p['v'][0]
            if parsed.path[:7] == '/embed/':
                return parsed.path.split('/')[2]
            if parsed.path[:3] == '/v/':
                return parsed.path.split('/')[2]
        
        # Fallback/Check query param again
        qs = parse_qs(parsed.query)
        if 'v' in qs:
            return qs['v'][0]

        raise ValueError(f"No video id found in URL: {url}")

    def get_transcript(self, video_id):
        try:
            transcript_list = None
            # Try static method first
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            except AttributeError:
                pass
            
            # If static failed, try instance method
            if transcript_list is None:
                api = YouTubeTranscriptApi()
                if hasattr(api, 'get_transcript'):
                    transcript_list = api.get_transcript(video_id)
                elif hasattr(api, 'fetch'):
                    transcript_list = api.fetch(video_id)
                else:
                    # Fallback: maybe it's list_transcripts?
                    # But let's raise for now
                    raise Exception("YouTubeTranscriptApi has no get_transcript or fetch method")

            # Handle the list
            text_parts = []
            for item in transcript_list:
                if isinstance(item, dict):
                    text_parts.append(item.get('text', ''))
                elif hasattr(item, 'text'):
                    text_parts.append(item.text)
                else:
                    # Fallback
                    text_parts.append(str(item))
            
            transcript_text = " ".join(text_parts)
            return transcript_text
        except Exception as e:
            raise Exception(f"Could not fetch transcript: {str(e)}")

    def summarize(self, url):
        try:
            video_id = self.extract_video_id(url)
            transcript = self.get_transcript(video_id)
            
            # Summarize using LLM
            # Truncate transcript if too long (approx 30k chars for safety with 8k context, though Llama 3.1 has 128k context, 
            # but let's be safe and also cost-effective/fast. 8b-instant might have smaller limits on Groq free tier?)
            # Groq Llama 3.1 8b instant usually has 128k context. But let's limit to reasonable size.
            max_chars = 32000 
            if len(transcript) > max_chars:
                transcript = transcript[:max_chars] + "... (truncated)"

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that summarizes YouTube videos based on their transcript. Provide a comprehensive summary of the following video transcript. Capture the main points and key details."),
                ("human", "{transcript}")
            ])
            
            chain = prompt | self.llm
            response = chain.invoke({"transcript": transcript})
            return response.content
            
        except Exception as e:
            return f"Error processing video: {str(e)}"
