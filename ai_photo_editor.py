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
7. Curves (point-based mapping)

Format your response as valid JSON:
{{
    "title": "Descriptive title for edit series",
    "description": "Brief explanation of recommended edits",
    "edits": [
        {{
            "type": "exposure|white_balance|contrast|highlights|shadows|saturation|clarity|curves",
            "value": "numeric value or JSON array for curves",
            "explanation": "Why this adjustment improves the image quality"
        }}
    ]
}}
Note: For 'curves', value should be a list of [input, output] pairs from 0.0 to 1.0, 
e.g., [[0.0, 0.0], [0.25, 0.3], [0.75, 0.7], [1.0, 1.0]] to lift shadows and compress highlights.
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
                "type": "exposure|white_balance|contrast|highlights|shadows|saturation|clarity|curves",
                "value": "numeric value or JSON array for curves",
                "explanation": "Why this updated value is better"
            }}
        ]
    }}
}}
Note: For 'curves', value should be a list of [input, output] pairs from 0.0 to 1.0.
"""
    return prompt

def generate_interactive_prompt(user_feedback: str, is_initial: bool = False) -> str:
    """Generate a prompt for AI to suggest edits based on user feedback"""
    if is_initial:
        prompt = f"""
You are an expert photo editor specializing in RAW image processing.
Analyze this image and recommend initial adjustments to optimize it for viewing.

Provide recommendations for:
1. Exposure adjustment  
2. White balance
3. Contrast
4. Highlight recovery
5. Shadow recovery
6. Clarity/saturation
7. Curves (point-based mapping)

Format your response as valid JSON:
{{
    "title": "Initial AI Assessment",
    "description": "Initial recommendations for image improvement",
    "edits": [
        {{
            "type": "exposure|white_balance|contrast|highlights|shadows|saturation|clarity|curves",
            "value": "numeric value or JSON array for curves",
            "explanation": "Why this adjustment improves the image quality"
        }}
    ]
}}
Note: For 'curves', value should be a list of [input, output] pairs from 0.0 to 1.0.
"""
    else:
        prompt = f"""
You are an expert photo editor. The user has reviewed the current edit and requested the following changes:
"{user_feedback}"

Analyze the provided image and suggest specific adjustments to fulfill the user's request.
Ensure your recommendations are complementary to the current look but move in the direction requested by the user.

Format your response as valid JSON:
{{
    "title": "Interactive Refinement",
    "description": "Refinements based on feedback: {user_feedback}",
    "edits": [
        {{
            "type": "exposure|white_balance|contrast|highlights|shadows|saturation|clarity|curves",
            "value": "numeric value or JSON array for curves",
            "explanation": "How this change addresses the user's request"
        }}
    ]
}}
Note: For 'curves', value should be a list of [input, output] pairs from 0.0 to 1.0.
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

def interpolate_curve(points):
    """
    Create a 256-entry lookup table from a set of [input, output] points.
    points: List of [x, y] where x, y are 0.0 to 1.0
    """
    # Ensure points are sorted by x
    sorted_points = sorted(points, key=lambda p: p[0])
    
    # Add boundary points if not present
    if not sorted_points or sorted_points[0][0] > 0:
        sorted_points.insert(0, [0.0, sorted_points[0][1] if sorted_points else 0.0])
    if sorted_points[-1][0] < 1.0:
        sorted_points.append([1.0, sorted_points[-1][1] if sorted_points else 1.0])
    
    lut = []
    for i in range(256):
        x = i / 255.0
        # Find the segment [p1, p2] that contains x
        for j in range(len(sorted_points) - 1):
            p1 = sorted_points[j]
            p2 = sorted_points[j+1]
            if p1[0] <= x <= p2[0]:
                # Linear interpolation
                t = (x - p1[0]) / (p2[0] - p1[0]) if p2[0] != p1[0] else 0
                y = p1[1] + t * (p2[1] - p1[1])
                lut.append(int(max(0, min(1, y)) * 255))
                break
        else:
            # This should not happen if boundary points are added
            lut.append(i)
            
    return lut

def apply_adjustments(img: Image.Image, recommendations: dict) -> Image.Image:
    """Apply AI recommendations to the image using PIL ImageEnhance"""
    processed_img = img.copy()
    edits = recommendations.get("edits", [])
    
    for edit in edits:
        edit_type = edit.get("type")
        value = edit.get("value")
        
        try:
            if edit_type == "curves":
                # AI provides a list of [input, output] pairs
                if isinstance(value, list):
                    lut = interpolate_curve(value)
                    # Apply to each channel
                    if processed_img.mode == 'RGB':
                        r, g, b = processed_img.split()
                        r = r.point(lut)
                        g = g.point(lut)
                        b = b.point(lut)
                        processed_img = Image.merge('RGB', (r, g, b))
                    else:
                        processed_img = processed_img.point(lut)
                continue

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
            current_img = img.copy()
        except Exception as e:
            logger.error(f"  Failed to load image {raw_file}: {e}")
            continue

        iterations_history = []
        round_num = 1
        
        # Interactive dial-in process
        while True:
            logger.info(f"  Iteration {round_num}")
            
            # Save current state for display and AI input
            preview_path = os.path.join(args.output_dir, f"temp_preview_{stem}.jpg")
            save_processed_image(current_img, preview_path)
            
            # Display the current image to the user
            try:
                current_img.show()
            except Exception as e:
                logger.warning(f"    Could not display image: {e}")
            
            # Get user feedback
            user_feedback = input(f"  [{raw_file}] Enter stylistic changes/requests, or 'done' to finish: ").strip()
            
            if user_feedback.lower() == 'done':
                logger.info(f"  User marked {raw_file} as done.")
                break
            
            # AI recommendations based on feedback
            is_initial = (round_num == 1)
            prompt = generate_interactive_prompt(user_feedback, is_initial=is_initial)
            
            logger.info(f"    Requesting AI edits based on: {user_feedback if not is_initial else 'initial analysis'}...")
            current_recs = send_image_to_ollama(
                preview_path, 
                prompt, 
                config['ollama_model'], 
                config['ollama_server_url'], 
                config['ollama_api_key']
            )
            
            if not validate_ai_response(current_recs):
                logger.warning(f"    Invalid AI response in iteration {round_num}. Skipping this pass.")
                iterations_history.append({
                    "round": round_num,
                    "user_feedback": user_feedback,
                    "recommendations": current_recs,
                    "status": "invalid_response"
                })
                round_num += 1
                continue

            # Apply recommendations cumulatively
            try:
                current_img = apply_adjustments(current_img, current_recs)
                save_processed_image(current_img, output_file)
                
                iterations_history.append({
                    "round": round_num,
                    "user_feedback": user_feedback,
                    "recommendations": current_recs,
                    "status": "applied"
                })
                logger.info(f"    Applied edits for iteration {round_num}")
            except Exception as e:
                logger.error(f"    Error applying adjustments in iteration {round_num}: {e}")
                iterations_history.append({
                    "round": round_num,
                    "user_feedback": user_feedback,
                    "recommendations": current_recs,
                    "status": "error",
                    "error": str(e)
                })
            
            round_num += 1
        
        # Cleanup temp preview
        if os.path.exists(preview_path):
            os.remove(preview_path)
        
        # Save final image and metadata
        save_processed_image(current_img, output_file)
        
        metadata = {
            "original_image": raw_file,
            "processing_date": time.time(),
            "iterations": iterations_history,
            "edit_series": f"edit_{stem}"
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
