# Implementation Plan: Interactive Human-in-the-Loop Image Editing

## Goal
Transform the static 3-iteration AI editing process into an interactive loop where a human user provides feedback and the AI suggests refinements until the user is satisfied.

## Tasks

1. **Implement Image Display Utility**
   - Modify `ai_photo_editor.py` to include a way to open the image for the user.
   - Action: Use `PIL.Image.show()` as the primary mechanism for platform-independent image opening.
   - Acceptance: The script can successfully trigger the OS default image viewer to display a JPEG.

2. **Create Interactive AI Prompt Template**
   - Add a new function `generate_interactive_prompt(user_feedback: str, is_initial: bool = False) -> str`.
   - The prompt should instruct the AI to:
     - Act as an expert photo editor.
     - Consider the provided image and the user's specific stylistic requests.
     - Provide recommendations in the existing JSON format (`type`, `value`, `explanation`).
     - If `is_initial` is True, provide a general optimization; otherwise, focus on the provided feedback.
   - Acceptance: The prompt clearly directs the AI to incorporate user feedback into the JSON response.

3. **Refactor the Main Processing Loop**
   - File: `ai_photo_editor.py`
   - Changes:
     - Remove `for round_num in range(1, 4):`.
     - Implement `while True:` loop for each image.
     - **Inside the loop**:
       - Save the current state of the image to a temporary preview file.
       - Call the image display utility to show the preview to the user.
       - Use `input()` to capture user feedback.
       - If input is `"done"`, break the loop.
       - Call `generate_interactive_prompt` with the user input.
       - Send the current preview image and the prompt to Ollama.
       - Parse the AI response and apply adjustments using `apply_adjustments`.
       - Update the working image object.
   - Acceptance: The script successfully iterates based on user input and terminates when "done" is entered.

4. **Update Image State Management**
   - Ensure that `apply_adjustments` is called on the *currently edited* image rather than the original image in every iteration.
   - This allows the user to "nudge" the image iteratively.
   - Acceptance: Subsequent AI edits build upon previous edits rather than resetting to the original.

5. **Modify Metadata Logging**
   - Update the metadata saved to `edit_{stem}.json`.
   - Instead of `iterations` (1-3), store a list of `interaction_history` containing:
     - User feedback.
     - AI recommendations.
     - Timestamp of the change.
   - Acceptance: The final JSON metadata accurately reflects the interactive session.

6. **Cleanup and Testing**
   - Ensure temporary preview files are managed (either deleted or overwritten).
   - Verify that the flow handles cases where AI returns invalid JSON (keep the previous state and warn the user).

## Files to Modify
- `ai_photo_editor.py`: All logic changes (prompts, loop, input, display).

## New Files
- None.

## Dependencies
- Task 2 (Prompting) must be complete before Task 3 (Loop implementation).
- Task 1 (Display) must be complete before Task 3.

## Risks
- **Blocking UI**: `PIL.Image.show()` is non-blocking on most systems. The script will proceed to `input()` immediately after opening the image. The user must be aware they need to look at the image then return to the terminal.
- **AI Hallucinations**: If the user requests a change the AI cannot express via the `exposure|contrast|saturation|clarity` types, the AI might provide invalid types. The existing `apply_adjustments` handles unknowns via `logger.warning`, which is acceptable.
- **State Drift**: Because edits are cumulative, multiple iterations of "increase exposure" might lead to over-exposure. This is the intended behavior of a human-in-the-loop system.
