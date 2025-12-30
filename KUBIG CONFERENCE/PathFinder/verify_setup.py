import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def check_file(path, description):
    if os.path.exists(path):
        print(f"✅ {description} found at {path}")
    else:
        print(f"❌ {description} NOT found at {path}")

def verify_imports():
    print("\nVerifying imports...")
    try:
        from core.state import MainState
        print("✅ core.state imported")
        from agents.interview_agent import run_interview_session
        print("✅ agents.interview_agent imported")
        from agents.ncs_job_recommender_agent import run_ncs_agent
        print("✅ agents.ncs_job_recommender_agent imported")
        from agents.posting_recommender_agent import run_posting_recommender_agent
        print("✅ agents.posting_recommender_agent imported")
        from agents.posting_manager_agent import PostManagerAgent
        print("✅ agents.posting_manager_agent imported")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
    except Exception as e:
        print(f"❌ Error during import: {e}")

def main():
    print("--- Verification Script ---")
    
    # Check Data Files
    check_file("data/job_service.db", "Job Service DB")
    check_file("data/ncs_vectorstore", "NCS Vectorstore")
    check_file("data/ncs_vectorstore/index.faiss", "NCS Vectorstore Index")
    check_file("data/ncs_faiss_index", "NCS FAISS Index")
    check_file("data/ncs_faiss_index/index.faiss", "NCS FAISS Index File")
    
    # Check Config
    check_file(".env", ".env file")
    check_file("requirements.txt", "requirements.txt")
    
    # Check Imports
    verify_imports()

if __name__ == "__main__":
    main()
