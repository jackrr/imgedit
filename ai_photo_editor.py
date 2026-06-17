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



RAW_EXTENSIONS = {'.cr2', '.cr3', '.crw', '.nef', '.orf', '.arw', '.mrw', 
                  '.raf', '.rw2', '.dng', '.3fr', '.erf', '.kdc', '.mef', 
                  '.pef', '.sr2', '.srw', '.tiff', '.tif'}

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
    valid_files = []
    
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in RAW_EXTENSIONS):
            valid_files.append(filename)
    
    if not valid_files:
        logger.warning(f"No RAW image files found in '{input_dir}'")
        return False

    
    return True

def load_image(image_path: str) -> Image.Image:
    """Load an image, using rawpy for RAW formats and PIL for others"""
    ext = Path(image_path).suffix.lower()
    
    if ext in RAW_EXTENSIONS:
        if rawpy is None:
            logger.error("rawpy library not installed. Cannot load RAW images.")
        else:
            try:
                with rawpy.imread(image_path) as raw:
                    # postprocess converts RAW to RGB array
                    rgb = raw.postprocess()
                    # Convert numpy array to PIL Image
                    return Image.fromarray(rgb)
            except Exception as e:
                logger.warning(f"rawpy failed to load {image_path}: {e}. Trying PIL fallback...")
    
    try:
        return Image.open(image_path)
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        raise

def create_preview_image(image_path: str, max_size: tuple = (800, 600)) -> str:
    """Create a lower-res preview of an image for Ollama transmission"""
    try:
        # Load the image (handles RAW via load_image)
        img = load_image(image_path)
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

def generate_refinement_prompt(previous_recommendations: dict) -> str:
    """Generate prompt for AI to refine the results of edits"""
    recs_text = json.dumps(previous_recommendations, indent=2)
    prompt = f"""
You are an expert photo editor. You are provided with two images:
1. The original RAW image.
2. The edited version resulting from the previous recommendations.

The recommendations that were applied are:
{recs_text}

Please perform the following:
1. Provide a professional critique of the edited image. 
   - Did the edits achieve the intended goal?
   - Are there any over-corrections or under-corrections?
2. Provide UPDATED recommendations to "dial in" the final look.

Format your response as valid JSON:
{{
    "critique": "Your professional critique of the current result",
    "recommendations": {{
        "title": "Updated title for edit series",
        "description": "Explanation of the refinements made",
        "edits": [
            {{
                "type": "exposure|white_balance|contrast|highlights|shadows|saturation|clarity",
                "value": "numeric or descriptive value",
                "explanation": "Why this updated value is better"
            }}
        ]
    }}
}}
"""
    return prompt

def generate_critique_prompt(original_recommendations: dict) -> str:
    """Generate prompt for AI to critique the results of edits"""
    recs_text = json.dumps(original_recommendations, indent=2)
    prompt = f"""
You are an expert photo critic. You are provided with two images: 
1. The original RAW image.
2. The edited version based on specific recommendations.

The recommended edits were:
{recs_text}

Please critique the edited image. 
- Did the edits achieve the intended goal?
- Is the image improved? 
- Are there any over-corrections (e.g., too much saturation, too bright)?
- What would you further improve?

Provide a detailed, professional critique.
"""
    return prompt

def send_images_to_ollama(image_paths: list, prompt: str, model: str, server_url: str, api_key: str = None) -> str:
    """Send multiple images and a prompt to Ollama API and get the AI response"""
    try:
        # Create preview images for all input paths
        img_base64_list = []
        preview_files_to_cleanup = []
        
        for path in image_paths:
            preview_path = create_preview_image(path)
            if preview_path != path:
                preview_files_to_cleanup.append(preview_path)
            
            with open(preview_path, "rb") as f:
                img_base64_list.append(base64.b64encode(f.read()).decode('utf-8'))
        
        # Ollama's API format for multimodal models
        url = f"{server_url}/api/generate"
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        
        payload = {
            "model": model,
            "prompt": prompt,
            "images": img_base64_list,
            "stream": False
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        
        # Clean up preview files
        for p in preview_files_to_cleanup:
            if os.path.exists(p):
                os.remove(p)

        if response.status_code == 400:
            logger.error("Ollama API returned 400 error. Fallback to text-only.")
            return send_simple_prompt_to_ollama(prompt, model, server_url, api_key).get('response', 'Error: AI failed to respond.')

        response.raise_for_status()
        return response.json().get('response', '')
        
    except Exception as e:
        logger.error(f"Error calling Ollama API for images {image_paths}: {e}")
        return f"Error during AI processing: {e}"

def send_image_to_ollama(image_path: str, prompt: str, model: str, server_url: str, api_key: str = None) -> dict:
    """Send image and prompt to Ollama API and get AI recommendations"""
    try:
        # Use the generalized function to get the raw response
        response_text = send_images_to_ollama([image_path], prompt, model, server_url, api_key)
        
        # Try to extract JSON from response
        try:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
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
                # AI provides offsets like +0.2 or -0.1. Factor = 1.0 + offset.
                factor = max(0.0, 1.0 + numeric_value)
                enhancer = ImageEnhance.Brightness(processed_img)
                processed_img = enhancer.enhance(factor)
            elif edit_type == "contrast":
                # AI provides percentages like +15 or -5. Factor = 1.0 + (perc/100).
                factor = max(0.0, 1.0 + (numeric_value / 100.0))
                enhancer = ImageEnhance.Contrast(processed_img)
                processed_img = enhancer.enhance(factor)
            elif edit_type == "saturation":
                # AI provides percentages like +5 or -5. Factor = 1.0 + (perc/100).
                factor = max(0.0, 1.0 + (numeric_value / 100.0))
                enhancer = ImageEnhance.Color(processed_img)
                processed_img = enhancer.enhance(factor)
            elif edit_type == "clarity":
                # AI provides percentages like +2 or +10. Factor = 1.0 + (perc/100).
                factor = max(0.0, 1.0 + (numeric_value / 100.0))
                enhancer = ImageEnhance.Contrast(processed_img)
                processed_img = enhancer.enhance(factor)
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

def get_ai_critique(original_path: str, edited_path: str, recommendations: dict, config: dict) -> str:
    """Get a critique from AI about the applied edits"""
    logger.info(f"  Requesting AI critique for {edited_path}...")
    prompt = generate_critique_prompt(recommendations)
    critique = send_images_to_ollama(
        [original_path, edited_path],
        prompt,
        config['ollama_model'],
        config['ollama_server_url'],
        config['ollama_api_key']
    )
    return critique

def get_ai_refinement(original_path: str, edited_path: str, recommendations: dict, config: dict) -> tuple:
    """
    Get a critique and updated recommendations from AI to 'dial in' the image.
    Returns: (refined_recs, critique)
    """
    logger.info(f"  Requesting AI refinement for {edited_path}...")
    prompt = generate_refinement_prompt(recommendations)
    response_text = send_images_to_ollama(
        [original_path, edited_path],
        prompt,
        config['ollama_model'],
        config['ollama_server_url'],
        config['ollama_api_key']
    )

    try:
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            refined_recs = data.get("recommendations")
            critique = data.get("critique", "No critique provided.")
            
            if refined_recs and validate_ai_response(refined_recs):
                return refined_recs, critique
            
    except Exception as e:
        logger.warning(f"  Failed to parse refinement JSON: {e}")

    return None, response_text

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
    raw_files = [f for f in os.listdir(args.input_dir) if any(f.lower().endswith(ext) for ext in RAW_EXTENSIONS)]
    
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
        
        stem = Path(raw_file).stem
        # Output files
        metadata_file = os.path.join(args.output_dir, f"edit_{stem}.json")
        output_file = os.path.join(args.output_dir, f"edit_{stem}.jpg")
        pre_edit_file = os.path.join(args.output_dir, f"pre_{stem}.jpg")
        
        # Save pre-edit version for comparison
        try:
            img = load_image(image_path)
            save_processed_image(img, pre_edit_file)
            logger.info(f"  Saved pre-edit image: {pre_edit_file}")
        except Exception as e:
            logger.error(f"  Failed to save pre-edit image for {raw_file}: {e}")
            img = None

        iterations_history = []
        current_recs = None
        best_recs = None
        
        # Iterative dial-in process (up to 3 rounds)
        for round_num in range(1, 4):
            logger.info(f"  Round {round_num}/3")
            
            if round_num == 1:
                # Initial prompt and recommendations
                prompt = generate_ai_prompt(image_path)
                current_recs = send_image_to_ollama(
                    image_path, 
                    prompt, 
                    config['ollama_model'], 
                    config['ollama_server_url'], 
                    config['ollama_api_key']
                )
                critique = "Initial recommendation pass."
            else:
                # Refinement: use previous result and recs to improve
                if not os.path.exists(output_file):
                    logger.warning(f"    No image from previous round. Skipping round {round_num}.")
                    break
                
                refined_recs, critique = get_ai_refinement(
                    image_path, output_file, current_recs, config
                )
                
                if refined_recs:
                    current_recs = refined_recs
                else:
                    logger.warning(f"    Failed to get valid refined recommendations in round {round_num}. Stopping iterations.")
                    break

            if not validate_ai_response(current_recs):
                logger.warning(f"    Invalid AI response in round {round_num}. Skipping this pass.")
                continue

            # Apply recommendations to the original image
            logger.info(f"    Applying edits for round {round_num}...")
            try:
                if img is None:
                    img = load_image(image_path)
                processed_img = apply_adjustments(img, current_recs)
                save_processed_image(processed_img, output_file)
                
                best_recs = current_recs
                iterations_history.append({
                    "round": round_num,
                    "recommendations": current_recs,
                    "critique": critique
                })
                logger.info(f"    Saved image for round {round_num}")
            except Exception as e:
                logger.error(f"    Error applying adjustments in round {round_num}: {e}")
                break
        
        # Save final metadata with the history of the dial-in process
        metadata = {
            "original_image": raw_file,
            "processing_date": time.time(),
            "final_recommendations": best_recs,
            "iterations": iterations_history,
            "edit_series": f"edit_{Path(raw_file).stem}"
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"  Created final output: {output_file}")
        logger.info(f"  Created final metadata: {metadata_file}")


    
    logger.info("Processing complete!")
    logger.info("AI Photo Editor has finished processing RAW images with AI recommendations.")

    
    return 0

if __name__ == "__main__":
    sys.exit(main())
