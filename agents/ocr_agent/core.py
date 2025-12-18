import cv2
import numpy as np
import tempfile
import os
import base64
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

def preprocess_image(
        img_path: str,
        op: str = "threshold",
        target_width: int = 1600) -> str:
    """
    Performs image processing (upscaling, thresholding, deskewing, or denoise) 
    to improve OCR quality, saving the result to a temporary file.
    
    Available operations (op): "threshold", "deskew", "denoise".
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")

    # --- 1. Common Preprocessing: Upscaling ---
    if target_width is not None and img.shape[1] < target_width:
        scale = target_width / img.shape[1]
        # Use INTER_CUBIC for quality upscaling
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # --- 2. Specific Operation ---
    if op == "threshold":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Gentle denoise (Bilateral Filter)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        # Adaptive Thresholding for clear text
        out = cv2.adaptiveThreshold(gray,
                                    255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY,
                                    31,
                                    10)
    
    elif op == "deskew":
        # --- Full Deskewing Logic (Re-added) ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        # Otsu thresholding to find text pixels
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(bw > 0))
        angle = 0.0
        
        # Calculate the rotation angle
        if coords.size > 0:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
        
        # Rotate the image around its center
        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
    elif op == "denoise":
        # --- Non-Local Means Denoising (New Preprocess Step) ---
        # Highly effective noise reduction, especially for color images/photos
        out = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    else: 
        # Default: return the upscaled image
        out = img

    # --- 3. Output Preparation ---
    # Ensure 3-channel PNG for the vision model
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    # Write to a temporary PNG and return its path
    tmpdir = tempfile.gettempdir()
    base = os.path.splitext(os.path.basename(img_path))[0]
    # Use the operation name in the path for clarity
    out_path = os.path.join(tmpdir, f"{base}_proc_{op}.png")
    cv2.imwrite(out_path, out)
    return out_path

class OCRAgent:
    def __init__(self, groq_api_key: str):
        """
        Initializes the OCR Agent with the Groq API key.
        """
        if not groq_api_key:
            raise ValueError("Groq API Key is required to initialize OCRAgent.")
            
        # Initialize the ChatGroq model with Vision capabilities
        self.llm = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name="meta-llama/llama-4-scout-17b-16e-instruct" 
        )

    def extract_text(self, image_path: str) -> str:
        """
        Extracts text from an image, applying the single threshold preprocessing step,
        then sending it to the LLM to return ONLY the extracted text.
        """
        path_to_encode = image_path
        
        # --- 1. Single-Step Preprocessing (Thresholding) ---
        try:
            # Using op="threshold" as the default single-step process
            enhanced_image_path = preprocess_image(image_path, op="threshold")
            path_to_encode = enhanced_image_path
        except Exception as e:
            # Fallback to original image if preprocessing fails
            print(f"Warning: Preprocessing failed ({e}). Using original image.")
            path_to_encode = image_path

        # --- 2. Encoding & Inference ---
        try:
            # Encode image to Base64
            with open(path_to_encode, "rb") as image_file:
                image_bytes = image_file.read()
            
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            # Construct the prompt
            message = [
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": (
                                "Extract all the text from this image perfectly. "
                                "Return ONLY the extracted text, no introductory or concluding remarks. "
                                "Preserve the formatting as much as possible.\n\n"
                                "MATH FORMATTING RULES:\n"
                                "1. Use standard LaTeX syntax for all mathematical equations.\n"
                                "2. Enclose inline equations in single dollar signs (e.g., $E=mc^2$).\n"
                                "3. Enclose block equations in double dollar signs (e.g., $$x=y$$).\n"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            },
                        },
                    ]
                )
            ]

            # Call the LLM
            response = self.llm.invoke(message)
            return response.content.strip()

        except Exception as e:
            return f"Error extracting text: {str(e)}"

        finally:
            # --- 3. Cleanup ---
            # Remove the temporary preprocessed file if it was created
            if path_to_encode != image_path:
                try:
                    if os.path.exists(path_to_encode):
                        os.unlink(path_to_encode)
                except Exception:
                    pass