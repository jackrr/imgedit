#!/usr/bin/env python3
"""
AI Photo Editor - Process RAW images using Ollama AI recommendations
"""
import argparse
import os
import sys
import json
import time
import logging
import requests
import base64
from pathlib import Path
from PIL import Image, ImageEnhance
try:
    import rawpy
    import numpy as np
    import imageio
except ImportError:
    rawpy = None
    np = None
    imageio = None



def setup_logging(level=logging.INFO):
    """Configure logging for the application"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger('ai_photo_editor')

logger = setup_logging()

def get_config():
    """Get configuration from environment variables with defaults"""
    return {
        'ollama_server_url': os.environ.get('OLLAMA_SERVER_URL', 'http://localhost:11434'),
        'ollama_api_key': os.environ.get('OLLAMA_API_KEY', ''),
        'ollama_model': os.environ.get('OLLAMA_MODEL', 'nemotron3:33b')
    }


def validate_input_directory(input_dir: str) -> bool:
    """Validate that input directory exists and contains valid RAW files"""
    if not os.path.exists(input_dir):
        logger.error(f"Input directory '{input_dir}' does not exist")
        return False
    
    if not os.path.isdir(input_dir):
        logger.error(f"Input path '{input_dir}' is not a directory")
        return False
    
    # Check for valid RAW image extensions
    raw_extensions = {'.cr2', '.cr3', '.crw', '.nef', '.orf', '.arw', '.mrw', 
                      '.raf', '.rw2', '.dng', '.3fr', '.erf', '.kdc', '.mef', 
                      '.pef', '.sr2', '.srw', '.tiff', '.tif'}
    valid_files = []
    
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in raw_extensions):
            valid_files.append(filename)
    
    if not valid_files:
        logger.warning(f"No RAW image files found in '{input_dir}'")
        return False

    
    return True

def load_raw_image(image_path: str) -> Image.Image:
    """Load a RAW image and convert it to a PIL Image"""
    if rawpy is None:
        logger.error("rawpy library not installed. Cannot load RAW images.")
        # Fallback to PIL for non-RAW formats if possible
        return Image.open(image_path)
    
    try:
        with rawpy.imread(image_path) as raw:
            # postprocess converts RAW to RGB array
            rgb = raw.postprocess()
            # Convert numpy array to PIL Image
            return Image.fromarray(rgb)
    except Exception as e:
        logger.error(f"Failed to load RAW image {image_path}: {e}")
        # Fallback to PIL if it's actually a JPEG/PNG etc.
        return Image.open(image_path)

def create_preview_image(image_path: str, max_size: tuple = (800, 600)) -> str:
    """Create a lower-res preview of an image for Ollama transmission"""
    try:
        # Load the image (handles RAW via load_raw_image)
        img = load_raw_image(image_path)
        # Create a preview
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Create a temporary preview file
        preview_path = f"{image_path}.preview.jpg"
        img.save(preview_path, 'JPEG', quality=85, optimize=True)
        
        return preview_path
    except Exception as e:
        logger.warning(f"Could not create preview for {image_path}: {e}")
        return image_path  # Return original if preview fails

def generate_ai_prompt(image_path: str) -> str:
    """Generate prompt for AI to analyze the image"""
    prompt = f"""
You are an expert photo editor specializing in RAW image processing.

Analyze this RAW image and recommend appropriate adjustments to optimize it for viewing.

Provide recommendations for:
1. Exposure adjustment  
2. White balance
3. Contrast
4. Highlight recovery
5. Shadow recovery
6. Clarity/saturation

Format your response as valid JSON:
{{
    "title": "Descriptive title for edit series",
    "description": "Brief explanation of recommended edits",
    "edits": [
        {{
            "type": "exposure|white_balance|contrast|highlights|shadows|saturation|clarity",
            "value": "numeric or descriptive value",
            "explanation": "Why this adjustment improves the image quality"
        }}
    ]
}}
"""
    return prompt

def send_image_to_ollama(image_path: str, prompt: str, model: str, server_url: str, api_key: str = None) -> dict:
    """Send image and prompt to Ollama API and get AI recommendations"""
    try:
        # Create preview image for transmission  
        preview_path = create_preview_image(image_path)
        logger.info(f"Created preview for {image_path}")
        
        # Ollama's API format for multimodal models (LLaVA-like)
        url = f"{server_url}/api/generate"

        
        headers = {
            'Content-Type': 'application/json'
        }
        
        # Add authorization header if available
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        
        # Prepare payload for Ollama multimodal request
        with open(preview_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')

        payload = {
            "model": model,
            "prompt": prompt,
            "images": [img_base64],
            "stream": False
        }
        
        # In case of API 400 errors, let's try an alternative approach
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        
        # Clean up preview file if it was created
        if preview_path != image_path and os.path.exists(preview_path):
            os.remove(preview_path)

        if response.status_code == 400:
            logger.error("Ollama API returned 400 error. This could be due to:\n"
                   "- Incorrect model (check if nemotron3:33b is installed)\n"
                   "- Invalid multimodal input\n"
                   "- Missing or incorrect preview image format\n"
                   "Testing with a simple text-only prompt instead.")
            
            # Fallback to simpler text-only request
            simple_payload = {
                "model": model,
                "prompt": prompt
            }
            return send_simple_prompt_to_ollama(prompt, model, server_url, api_key)

        
        response.raise_for_status()
        
        data = response.json()
        generated_text = data.get('response', '')
        
        # Try to extract JSON from response
        try:
            import re
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                json_string = json_match.group(0)
                return json.loads(json_string)
        except:
            pass
            
        # Return fallback if no JSON found
        return {
            "title": "AI Recommendation",
            "description": "AI-generated edit suggestions",
            "edits": []
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Ollama API for {image_path}: {e}")
        return {
            "title": "AI Recommendation", 
            "description": "AI-generated edit suggestions",
            "edits": []
        }
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return {
            "title": "AI Recommendation",
            "description": "AI-generated edit suggestions",
            "edits": []
        }

def send_simple_prompt_to_ollama(prompt: str, model: str, server_url: str, api_key: str = None) -> dict:
    """Fallback method to send text-only prompt when multimodal fails"""
    try:
        url = f"{server_url}/api/generate"
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        generated_text = data.get('response', '')
        
        # Try to extract JSON
        try:
            import re
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                json_string = json_match.group(0)
                return json.loads(json_string)
        except:
            pass
            
        # Return fallback if no JSON found
        return {
            "title": "AI Recommendation (default)",
            "description": "AI-generated edit suggestions (fallback)",
            "edits": []
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in simple Ollama API call: {e}")
        return {
            "title": "AI Recommendation (fallback)",
            "description": "AI-generated edit suggestions (fallback)",
            "edits": []
        }
    except Exception as e:
        logger.error(f"Error in simple Ollama call: {e}")
        return {
            "title": "AI Recommendation (fallback)",
            "description": "AI-generated edit suggestions (fallback)",
            "edits": []
        }

def validate_ai_response(response: dict) -> bool:

    """Validate that the AI response contains the required structure and types"""
    if not isinstance(response, dict):
        return False
    if "edits" not in response or not isinstance(response["edits"], list):
        return False
    for edit in response["edits"]:
        if not isinstance(edit, dict) or "type" not in edit or "value" not in edit:
            return False
    return True

def apply_adjustments(img: Image.Image, recommendations: dict) -> Image.Image:
    """Apply AI recommendations to the image using PIL ImageEnhance"""
    processed_img = img.copy()
    edits = recommendations.get("edits", [])
    
    for edit in edits:
        edit_type = edit.get("type")
        value = edit.get("value")
        
        try:
            # Convert value to float if it's numeric
            if isinstance(value, str):
                # Try to extract first float from string
                import re
                match = re.search(r"[-+]?\d*\.\d+|\d+", value)
                if match:
                    numeric_value = float(match.group())
                else:
                    numeric_value = 1.0 # Default
            else:
                numeric_value = float(value)
            
            if edit_type == "exposure":
                enhancer = ImageEnhance.Brightness(processed_img)
                processed_img = enhancer.enhance(numeric_value)
            elif edit_type == "contrast":
                enhancer = ImageEnhance.Contrast(processed_img)
                processed_img = enhancer.enhance(numeric_value)
            elif edit_type == "saturation":
                enhancer = ImageEnhance.Color(processed_img)
                processed_img = enhancer.enhance(numeric_value)
            elif edit_type == "clarity":
                # Basic clarity simulation using contrast/sharpness
                enhancer = ImageEnhance.Contrast(processed_img)
                processed_img = enhancer.enhance(numeric_value)
            # Other types (highlights, shadows, white_balance) would require 
            # more complex numpy manipulations on the array.
            
        except Exception as e:
            logger.warning(f"Could not apply edit {edit_type} with value {value}: {e}")
            
    return processed_img

def save_processed_image(img: Image.Image, output_path: str):
    """Save the processed image as a JPEG"""
    try:
        img.save(output_path, "JPEG", quality=95)
    except Exception as e:
        logger.error(f"Failed to save image to {output_path}: {e}")

def main():
    """Main execution function"""
    # Get config from environment variables
    config = get_config()
    
    parser = argparse.ArgumentParser(description="AI Photo Editor for RAW images")
    parser.add_argument("input_dir", help="Directory containing RAW image files")
    parser.add_argument("--output_dir", default="output", help="Output directory for processed images")
    parser.add_argument("--model", help="Ollama model to use for AI processing (overrides environment)")
    parser.add_argument("--server", help="Ollama server URL (overrides environment)")
    parser.add_argument("--api-key", help="Ollama API key (overrides environment)")
    
    args = parser.parse_args()
    
    # Only override with CLI args if they were provided
    if args.model is not None:
        config['ollama_model'] = args.model
    if args.server is not None:
        config['ollama_server_url'] = args.server  
    if args.api_key is not None:
        config['ollama_api_key'] = args.api_key
    
    # Validate input directory
    if not validate_input_directory(args.input_dir):
        return 1
    
    # Validate output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Get list of RAW files
    raw_extensions = {'.cr2', '.cr3', '.crw', '.nef', '.orf', '.arw', '.mrw', 
                      '.raf', '.rw2', '.dng', '.3fr', '.erf', '.kdc', '.mef', 
                      '.pef', '.sr2', '.srw', '.tiff', '.tif'}
    raw_files = [f for f in os.listdir(args.input_dir) if any(f.lower().endswith(ext) for ext in raw_extensions)]
    
    if not raw_files:
        logger.info("No RAW files found to process")
        return 1
    
    logger.info(f"Processing {len(raw_files)} RAW files:")
    logger.info(f"  Input directory: {args.input_dir}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Ollama server: {config['ollama_server_url']}")
    logger.info(f"  Ollama model: {config['ollama_model']}")

    
    # Process each file
    for raw_file in raw_files:
        image_path = os.path.join(args.input_dir, raw_file)
        logger.info(f"Processing {raw_file}")
        
        # Generate system prompt
        prompt = generate_ai_prompt(image_path)
        
        # Send to Ollama for AI recommendations  
        recommendations = send_image_to_ollama(
            image_path, 
            prompt, 
            config['ollama_model'], 
            config['ollama_server_url'], 
            config['ollama_api_key']
        )
        
        logger.info(f"  AI recommendations received")
        
        # Create output files (simulating actual processing)
        metadata_file = os.path.join(args.output_dir, f"edit_{Path(raw_file).stem}.json")
        output_file = os.path.join(args.output_dir, f"edit_{Path(raw_file).stem}.jpg")
        
        metadata = {
            "original_image": raw_file,
            "processing_date": time.time(),
            "recommendations": recommendations,
            "edit_series": f"edit_{Path(raw_file).stem}"
        }
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Process and save the actual image
        if validate_ai_response(recommendations):
            logger.info(f"  Applying AI edits to {raw_file}...")
            img = load_raw_image(image_path)
            processed_img = apply_adjustments(img, recommendations)
            save_processed_image(processed_img, output_file)
            logger.info(f"  Saved processed image: {output_file}")
        else:
            logger.warning(f"  Invalid AI response for {raw_file}. Skipping image processing.")
            # Fallback to saving the original as-is or a simple copy if desired.
            # For now, we just log the invalid response.
        
        logger.info(f"  Created: {output_file}")
        logger.info(f"  Created: {metadata_file}")


    
    logger.info("Processing complete!")
    logger.info("AI Photo Editor has finished processing RAW images with AI recommendations.")

    
    return 0

if __name__ == "__main__":
    sys.exit(main())
