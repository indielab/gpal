import os
import logging
import glob
import fnmatch
from pathlib import Path
from typing import List, Optional, Literal
from fastmcp import FastMCP
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastMCP
mcp = FastMCP("gpal")

# Global state for sessions
sessions = {}

# Model Configuration
MODEL_ALIASES = {
    "flash": "gemini-3-flash-preview",
    "pro": "gemini-3-pro-preview"
}

# --- Gemini Tools ---

def list_directory(path: str = ".") -> List[str]:
    """Lists files and directories in the given path."""
    try:
        p = Path(path)
        if not p.exists():
            return [f"Error: Path '{path}' does not exist."]
        return [str(item.name) for item in p.iterdir()]
    except Exception as e:
        return [f"Error listing directory: {str(e)}"]

def read_file(path: str) -> str:
    """Reads the content of a file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file '{path}': {str(e)}"

def search_project(search_term: str, glob_pattern: str = "**/*") -> str:
    """
    Searches for a text term in files matching the glob pattern.
    Returns a summary of matches.
    """
    try:
        matches = []
        files = glob.glob(glob_pattern, recursive=True)
        
        if len(files) > 1000:
             return f"Error: Too many files match '{glob_pattern}' ({len(files)}). Please refine the glob."

        count = 0
        for file in files:
            if os.path.isfile(file):
                try:
                    with open(file, "r", encoding="utf-8", errors='ignore') as f:
                        content = f.read()
                        if search_term in content:
                            matches.append(f"Match in: {file}")
                            count += 1
                            if count >= 20: 
                                matches.append("... (more matches truncated)")
                                break
                except:
                    continue 
        
        if not matches:
            return "No matches found."
        return "\n".join(matches)

    except Exception as e:
        return f"Error searching project: {str(e)}"


# --- Server Logic ---

def get_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment.")
    return genai.Client(api_key=api_key)

def create_chat(client: genai.Client, model_name: str, history: List = None):
    """Helper to create a configured chat session."""
    gemini_tools = [list_directory, read_file, search_project]
    
    return client.chats.create(
        model=model_name,
        history=history or [],
        config=types.GenerateContentConfig(
            temperature=0.2,
            tools=gemini_tools, 
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=False,
                maximum_remote_calls=10 
            ),
            system_instruction="""You are a consultant AI accessed via the Model Context Protocol (MCP).
            Your role is to provide high-agency, deep reasoning and analysis on tasks, usually in git repositories.
            
            You have access to tools: 'list_directory', 'read_file', and 'search_project'. Use them proactively to explore the codebase.
            You have a massive context window (2M tokens); do not hesitate to read entire files or multiple modules to gather complete context.
            """
        )
    )

def get_session(session_id: str, client: genai.Client, requested_model_alias: str):
    target_model = MODEL_ALIASES.get(requested_model_alias.lower(), requested_model_alias)
    
    if session_id not in sessions:
        sessions[session_id] = create_chat(client, target_model)
        sessions[session_id]._gpal_model_name = target_model
    else:
        current_session = sessions[session_id]
        current_model = getattr(current_session, "_gpal_model_name", None)
        
        if current_model != target_model:
            try:
                # Migrate history to new model
                old_history = current_session._curated_history if hasattr(current_session, "_curated_history") else []
                logging.info(f"Migrating session {session_id} from {current_model} to {target_model}")
                sessions[session_id] = create_chat(client, target_model, history=old_history)
                sessions[session_id]._gpal_model_name = target_model
            except Exception as e:
                logging.error(f"Failed to migrate session history: {e}")
                sessions[session_id] = create_chat(client, target_model)
                sessions[session_id]._gpal_model_name = target_model
                
    return sessions[session_id]

def _consult_impl(query: str, session_id: str, model_alias: str, file_paths: List[str]) -> str:
    """Internal implementation for consulting Gemini."""
    client = get_client()
    session = get_session(session_id, client, model_alias)
    
    parts = []
    
    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                parts.append(types.Part.from_text(text=f"--- START FILE: {path} ---\n{content}\n--- END FILE: {path}---\n"))
        except Exception as e:
            return f"Error reading file {path}: {str(e)}"
            
    parts.append(types.Part.from_text(text=query))
    
    try:
        response = session.send_message(parts)
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini: {str(e)}"

# --- Public Tools ---

@mcp.tool()
def consult_gemini_flash(query: str, session_id: str = "default", file_paths: List[str] = []) -> str:
    """
    Consults Gemini 3 Flash (Fast/Efficient). Context window of 2,000,000 tokens.
    
    Gemini has its own tools to list directories, read files, and search the project
    autonomously. You do not need to provide all file contents.
    
    LIMITED USE FOR FLASH MODEL: High-speed exploration and context gathering. Use this tool FIRST to:
    1. Map out the project structure (list_directory).
    2. Locate specific code (search_project).
    3. Read and summarize files (read_file).
    
    Once the relevant context is gathered, switch to 'consult_gemini_pro'.
    
    Args:
        query: The question or instruction.
        session_id: ID for conversation history. Shared with Pro models.
        file_paths: (Optional) Essential files to get started with exploring.
    """
    return _consult_impl(query, session_id, "flash", file_paths)

@mcp.tool()
def consult_gemini_pro(query: str, session_id: str = "default", file_paths: List[str] = []) -> str:
    """
    Consults Gemini 3 Pro (Reasoning/Deep). Context window of 2,000,000 tokens.
    
    Gemini has its own tools to list directories, read files, and search the project.
    autonomously. You do not need to provide all file contents. Encourage it to read
    whole files and provide holistic feedback.
    
    PRIMARY USE FOR PRO MODEL: reviews, second opinions, thinking, philosophy, synthesis, and coding.
    
    Args:
        query: The question or instruction.
        session_id: ID for conversation history. Shared with Flash models.
        file_paths: (Optional) Essential files to get started with exploring.
    """
    return _consult_impl(query, session_id, "pro", file_paths)

def main():
    mcp.run()

if __name__ == "__main__":
    main()
